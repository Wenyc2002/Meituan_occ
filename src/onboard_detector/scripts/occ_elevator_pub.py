#!/usr/bin/env python3
"""ROS1 elevator free-space evaluator.

This node evaluates elevator free space on the ROS1 robot runtime.
It subscribes to LiDAR, tracked persons, the global map, and /elevator_area,
then publishes local occupancy, free-space polygon, enterable flag, and metrics.
"""

from __future__ import annotations

import json
import math
from collections import deque

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point, Point32, PolygonStamped
from nav_msgs.msg import MapMetaData, OccupancyGrid
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Bool, Float32, Int32, String
from visualization_msgs.msg import Marker, MarkerArray

try:
    import sensor_msgs.point_cloud2 as pc2
except Exception:  # pragma: no cover - available after sourcing ROS.
    pc2 = None

try:
    from spencer_tracking_msgs.msg import TrackedPersons
except Exception:  # pragma: no cover - optional at import time.
    TrackedPersons = None


def normalize_frame_id(frame_id):
    return (frame_id or "").lstrip("/")


def quaternion_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_to_matrix(transform):
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quaternion_to_matrix(transform.transform.rotation)
    t = transform.transform.translation
    mat[:3, 3] = [t.x, t.y, t.z]
    return mat


def transform_points(points, mat):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return points
    return points @ mat[:3, :3].T + mat[:3, 3]


def point_in_polygon_mask(points_xy, polygon_xy):
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    polygon_xy = np.asarray(polygon_xy, dtype=np.float64).reshape(-1, 2)
    if len(polygon_xy) < 3 or points_xy.size == 0:
        return np.zeros(points_xy.shape[0], dtype=bool)

    x = points_xy[:, 0]
    y = points_xy[:, 1]
    inside = np.zeros(points_xy.shape[0], dtype=bool)
    xj, yj = polygon_xy[-1]
    for xi, yi in polygon_xy:
        crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= crosses
        xj, yj = xi, yi
    return inside


def polygon_area(points_xy):
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(points_xy) < 3:
        return 0.0
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def convex_hull(points_xy):
    if len(points_xy) <= 1:
        return np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    points = sorted(set(map(tuple, np.round(points_xy, 4))))
    if len(points) <= 1:
        return np.asarray(points, dtype=np.float32)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)


def bresenham_cells(x0, y0, x1, y1):
    cells = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return cells


class ElevatorSpaceMonitor:
    def __init__(self):
        rospy.init_node("elevator_space_monitor", anonymous=False)

        self.target_frame = rospy.get_param("~target_frame", "base_link")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/scan_pcd")
        self.lidar_type = rospy.get_param("~lidar_type", "scan").lower()
        self.person_topic = rospy.get_param("~person_topic", "/tracked_persons_situ")
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.elevator_topic = rospy.get_param("~elevator_area_topic", "/elevator_area")
        self.resolution = float(rospy.get_param("~grid_resolution", 0.05))
        self.z_min = float(rospy.get_param("~z_min", -0.55))
        self.z_max = float(rospy.get_param("~z_max", 1.20))
        self.robot_width = float(rospy.get_param("~robot_width", 0.60))
        self.robot_length = float(rospy.get_param("~robot_length", 0.75))
        self.safety_margin = float(rospy.get_param("~safety_margin", 0.08))
        self.area_threshold = float(rospy.get_param("~area_threshold", 1.20))
        self.person_radius = float(rospy.get_param("~person_radius", 0.25))
        self.person_stale_sec = float(rospy.get_param("~person_stale_sec", 1.0))
        self.unknown_as_occupied = bool(rospy.get_param("~unknown_as_occupied", True))
        self.stable_enter_frames = int(rospy.get_param("~stable_enter_frames", 3))
        self.stable_block_frames = int(rospy.get_param("~stable_block_frames", 2))
        self.debug = bool(rospy.get_param("~debug", True))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.map_msg = None
        self.elevator_msg = None
        self.persons = {}
        self.enter_history = deque(maxlen=max(self.stable_enter_frames, self.stable_block_frames, 1))

        self.occupancy_pub = rospy.Publisher("/elevator/occupancy_grid", OccupancyGrid, queue_size=1)
        self.legacy_map_pub = rospy.Publisher("/map_occ", OccupancyGrid, queue_size=1)
        self.area_polygon_pub = rospy.Publisher("/elevator/area_polygon", PolygonStamped, queue_size=1)
        self.free_polygon_pub = rospy.Publisher("/elevator/free_space_polygon", PolygonStamped, queue_size=1)
        self.marker_pub = rospy.Publisher("/elevator/free_space_markers", MarkerArray, queue_size=1)
        self.enterable_pub = rospy.Publisher("/elevator/enterable", Bool, queue_size=1)
        self.free_area_pub = rospy.Publisher("/elevator/free_area", Float32, queue_size=1)
        self.free_ratio_pub = rospy.Publisher("/elevator/free_ratio", Float32, queue_size=1)
        self.legacy_free_pub = rospy.Publisher("/elevator_occupancy", Float32, queue_size=1)
        self.map_ratio_pub = rospy.Publisher("/elevator/static_map_free_ratio", Float32, queue_size=1)
        self.person_count_pub = rospy.Publisher("/elevator/person_count", Int32, queue_size=1)
        self.legacy_count_pub = rospy.Publisher("/elevator_person_count", Int32, queue_size=1)
        self.legacy_flag_pub = rospy.Publisher("/elevator_flag", Int32, queue_size=1)
        self.metrics_pub = rospy.Publisher("/elevator/metrics", String, queue_size=1)

        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback, queue_size=1)
        rospy.Subscriber(self.elevator_topic, PolygonStamped, self.elevator_callback, queue_size=1)
        if self.lidar_type in ("pointcloud2", "cloud", "pcd"):
            rospy.Subscriber(self.lidar_topic, PointCloud2, self.cloud_callback, queue_size=1)
        else:
            rospy.Subscriber(self.lidar_topic, LaserScan, self.scan_callback, queue_size=1)
        if TrackedPersons is not None:
            rospy.Subscriber(self.person_topic, TrackedPersons, self.person_callback, queue_size=1)
        else:
            rospy.logwarn("spencer_tracking_msgs is unavailable; person input disabled")

        rospy.loginfo("elevator_space_monitor started: lidar=%s (%s), person=%s, map=%s, area=%s",
                      self.lidar_topic, self.lidar_type, self.person_topic, self.map_topic, self.elevator_topic)

    def map_callback(self, msg):
        self.map_msg = msg

    def elevator_callback(self, msg):
        self.elevator_msg = msg

    def person_callback(self, msg):
        target_polygon = self.get_polygon_in_frame(self.target_frame)
        if target_polygon is None:
            return
        now = rospy.Time.now()
        current_ids = set()
        for idx, track in enumerate(msg.tracks):
            track_id = int(getattr(track, "track_id", idx))
            pos = track.pose.pose.position
            point = np.array([[pos.x, pos.y, pos.z]], dtype=np.float64)
            source_frame = normalize_frame_id(msg.header.frame_id or self.target_frame)
            point = self.transform_points_between(point, source_frame, self.target_frame)
            if point is None:
                continue
            inside = bool(point_in_polygon_mask(point[:, :2], target_polygon)[0])
            self.persons[track_id] = {"xy": point[0, :2], "stamp": now, "inside": inside}
            current_ids.add(track_id)
        for old_id in list(self.persons.keys()):
            if old_id not in current_ids and (now - self.persons[old_id]["stamp"]).to_sec() > self.person_stale_sec:
                del self.persons[old_id]

    def scan_callback(self, scan):
        points = []
        angle = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                points.append([r * math.cos(angle), r * math.sin(angle), 0.0])
            angle += scan.angle_increment
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        source_frame = normalize_frame_id(scan.header.frame_id or self.target_frame)
        points = self.transform_points_between(points, source_frame, self.target_frame)
        if points is None:
            return
        self.process(points, scan.header.stamp if scan.header.stamp else rospy.Time.now(), "scan")

    def cloud_callback(self, cloud):
        if pc2 is None:
            rospy.logwarn_throttle(2.0, "sensor_msgs.point_cloud2 unavailable; cannot read PointCloud2")
            return
        points = np.asarray(
            [[x, y, z] for x, y, z in pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)],
            dtype=np.float64,
        ).reshape(-1, 3)
        points = points[(points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)]
        source_frame = normalize_frame_id(cloud.header.frame_id or self.target_frame)
        points = self.transform_points_between(points, source_frame, self.target_frame)
        if points is None:
            return
        self.process(points, cloud.header.stamp if cloud.header.stamp else rospy.Time.now(), "cloud")

    def transform_points_between(self, points, source_frame, target_frame):
        source_frame = normalize_frame_id(source_frame)
        target_frame = normalize_frame_id(target_frame)
        if points.size == 0 or source_frame == target_frame:
            return points
        try:
            tf_msg = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.1))
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "TF %s -> %s unavailable: %s", source_frame, target_frame, exc)
            return None
        return transform_points(points, transform_to_matrix(tf_msg))

    def get_polygon_in_frame(self, target_frame):
        if self.elevator_msg is None:
            return None
        polygon = np.asarray([[p.x, p.y, p.z] for p in self.elevator_msg.polygon.points], dtype=np.float64)
        if len(polygon) < 3:
            return None
        source_frame = normalize_frame_id(self.elevator_msg.header.frame_id or self.map_frame)
        points = self.transform_points_between(polygon, source_frame, target_frame)
        if points is None:
            return None
        return points[:, :2]

    def process(self, lidar_points, stamp, input_kind):
        polygon = self.get_polygon_in_frame(self.target_frame)
        if polygon is None:
            rospy.logwarn_throttle(1.0, "Waiting for valid /elevator_area and TF")
            self.publish_enterable(False)
            return

        person_points, person_count = self.current_person_obstacles(polygon)
        lidar_points = lidar_points[(lidar_points[:, 2] >= self.z_min) & (lidar_points[:, 2] <= self.z_max)]
        obstacle_points = np.vstack([lidar_points[:, :3], person_points]) if len(person_points) else lidar_points[:, :3]
        cropped = obstacle_points[point_in_polygon_mask(obstacle_points[:, :2], polygon)]

        free_grid, occupancy, grid_info = self.build_grids(cropped, lidar_points, polygon)
        free_hull, free_area = self.free_hull_from_grid(free_grid, grid_info)
        free_ratio = self.free_ratio(free_grid, polygon, grid_info)
        map_free_ratio = self.static_map_free_ratio()
        enterable_raw = self.decide_enterable(free_hull, free_area)
        enterable = self.apply_stability(enterable_raw)

        occ_msg = self.make_occupancy_grid_msg(occupancy, grid_info, stamp)
        area_msg = self.make_polygon_msg(polygon, stamp)
        free_msg = self.make_polygon_msg(free_hull, stamp)
        markers = self.make_markers(polygon, free_hull, enterable, free_area, person_count, stamp)

        self.occupancy_pub.publish(occ_msg)
        self.legacy_map_pub.publish(occ_msg)
        self.area_polygon_pub.publish(area_msg)
        self.free_polygon_pub.publish(free_msg)
        self.marker_pub.publish(markers)
        self.enterable_pub.publish(Bool(data=enterable))
        self.free_area_pub.publish(Float32(free_area))
        self.free_ratio_pub.publish(Float32(free_ratio))
        self.legacy_free_pub.publish(Float32(free_ratio * 100.0))
        self.map_ratio_pub.publish(Float32(map_free_ratio))
        self.person_count_pub.publish(Int32(person_count))
        self.legacy_count_pub.publish(Int32(person_count))
        self.legacy_flag_pub.publish(Int32(1 if enterable else 0))

        metrics = {
            "stamp": stamp.to_sec(),
            "input": input_kind,
            "lidar_points": int(len(lidar_points)),
            "cropped_obstacles": int(len(cropped)),
            "person_count": int(person_count),
            "free_area": round(float(free_area), 3),
            "free_ratio": round(float(free_ratio), 3),
            "static_map_free_ratio": round(float(map_free_ratio), 3),
            "enterable": bool(enterable),
        }
        self.metrics_pub.publish(String(data=json.dumps(metrics, sort_keys=True)))
        if self.debug:
            rospy.loginfo_throttle(1.0, "elevator metrics: %s", metrics)

    def current_person_obstacles(self, polygon):
        now = rospy.Time.now()
        points = []
        person_count = 0
        for person_id in list(self.persons.keys()):
            person = self.persons[person_id]
            if (now - person["stamp"]).to_sec() > self.person_stale_sec:
                del self.persons[person_id]
                continue
            if not bool(point_in_polygon_mask(np.asarray([person["xy"]]), polygon)[0]):
                continue
            person_count += 1
            cx, cy = person["xy"]
            for i in range(12):
                angle = 2.0 * math.pi * i / 12.0
                points.append([cx + self.person_radius * math.cos(angle), cy + self.person_radius * math.sin(angle), 0.5])
            points.append([cx, cy, 0.5])
        return np.asarray(points, dtype=np.float64).reshape(-1, 3), person_count

    def build_grids(self, occupied_points, observed_points, polygon):
        inflate_radius = self.robot_width * 0.5 + self.safety_margin
        min_xy = polygon.min(axis=0) - inflate_radius
        max_xy = polygon.max(axis=0) + inflate_radius
        width = max(1, int(math.ceil((max_xy[0] - min_xy[0]) / self.resolution)) + 1)
        height = max(1, int(math.ceil((max_xy[1] - min_xy[1]) / self.resolution)) + 1)
        xs = min_xy[0] + np.arange(width) * self.resolution
        ys = min_xy[1] + np.arange(height) * self.resolution
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        cells = np.column_stack([gx.ravel(), gy.ravel()])
        inside = point_in_polygon_mask(cells, polygon).reshape(height, width)

        occupied = np.zeros((height, width), dtype=bool)
        if occupied_points.size > 0:
            ij = np.floor((occupied_points[:, :2] - min_xy) / self.resolution).astype(np.int32)
            valid = (ij[:, 0] >= 0) & (ij[:, 0] < width) & (ij[:, 1] >= 0) & (ij[:, 1] < height)
            ij = ij[valid]
            inflate_cells = int(math.ceil(inflate_radius / self.resolution))
            offsets = [
                (dx, dy)
                for dx in range(-inflate_cells, inflate_cells + 1)
                for dy in range(-inflate_cells, inflate_cells + 1)
                if dx * dx + dy * dy <= inflate_cells * inflate_cells
            ]
            for dx, dy in offsets:
                xs_idx = ij[:, 0] + dx
                ys_idx = ij[:, 1] + dy
                valid = (xs_idx >= 0) & (xs_idx < width) & (ys_idx >= 0) & (ys_idx < height)
                occupied[ys_idx[valid], xs_idx[valid]] = True

        observed = self.compute_observed_grid(observed_points, min_xy, width, height)
        free_grid = inside & observed & ~occupied
        component = self.largest_free_component(free_grid)

        occupancy = np.full((height, width), -1, dtype=np.int8)
        if self.unknown_as_occupied:
            occupancy[inside] = 100
        occupancy[inside & observed] = 0
        occupancy[inside & occupied] = 100
        return component, occupancy, {"origin": min_xy, "width": width, "height": height, "inside": inside}

    def compute_observed_grid(self, points, min_xy, width, height):
        observed = np.zeros((height, width), dtype=bool)
        if points.size == 0:
            return observed
        origin = np.floor((np.asarray([0.0, 0.0]) - min_xy) / self.resolution).astype(np.int32)
        point_cells = np.floor((points[:, :2] - min_xy) / self.resolution).astype(np.int32)
        valid = (
            (point_cells[:, 0] >= 0) & (point_cells[:, 0] < width)
            & (point_cells[:, 1] >= 0) & (point_cells[:, 1] < height)
        )
        for x1, y1 in point_cells[valid]:
            for x, y in bresenham_cells(int(origin[0]), int(origin[1]), int(x1), int(y1))[:-1]:
                if 0 <= x < width and 0 <= y < height:
                    observed[y, x] = True
        return observed

    def largest_free_component(self, free_grid):
        visited = np.zeros_like(free_grid, dtype=bool)
        best = []
        height, width = free_grid.shape
        for sy, sx in np.argwhere(free_grid):
            if visited[sy, sx]:
                continue
            queue = deque([(int(sy), int(sx))])
            visited[sy, sx] = True
            component = []
            while queue:
                y, x = queue.popleft()
                component.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and free_grid[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(component) > len(best):
                best = component
        result = np.zeros_like(free_grid, dtype=bool)
        if best:
            ys, xs = zip(*best)
            result[np.asarray(ys), np.asarray(xs)] = True
        return result

    def free_hull_from_grid(self, free_grid, grid_info):
        indices = np.argwhere(free_grid)
        if len(indices) < 3:
            return np.empty((0, 2), dtype=np.float32), 0.0
        origin = grid_info["origin"]
        points = np.column_stack([origin[0] + indices[:, 1] * self.resolution, origin[1] + indices[:, 0] * self.resolution])
        hull = convex_hull(points)
        return hull, polygon_area(hull)

    def free_ratio(self, free_grid, polygon, grid_info):
        inside_count = int(np.count_nonzero(grid_info["inside"]))
        if inside_count == 0:
            return 0.0
        return float(np.count_nonzero(free_grid) / float(inside_count))

    def static_map_free_ratio(self):
        if self.map_msg is None or self.elevator_msg is None:
            return -1.0
        polygon = self.get_elevator_polygon_in_map()
        if polygon is None:
            return -1.0
        info = self.map_msg.info
        data = np.asarray(self.map_msg.data, dtype=np.int16).reshape(info.height, info.width)
        xs = info.origin.position.x + np.arange(info.width) * info.resolution
        ys = info.origin.position.y + np.arange(info.height) * info.resolution
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        inside = point_in_polygon_mask(np.column_stack([gx.ravel(), gy.ravel()]), polygon).reshape(info.height, info.width)
        total = int(np.count_nonzero(inside))
        if total == 0:
            return -1.0
        free = int(np.count_nonzero(inside & (data == 0)))
        return float(free / float(total))

    def get_elevator_polygon_in_map(self):
        if self.elevator_msg is None:
            return None
        polygon = np.asarray([[p.x, p.y, p.z] for p in self.elevator_msg.polygon.points], dtype=np.float64)
        source_frame = normalize_frame_id(self.elevator_msg.header.frame_id or self.map_frame)
        points = self.transform_points_between(polygon, source_frame, self.map_frame)
        if points is None:
            return None
        return points[:, :2]

    def decide_enterable(self, free_hull, free_area):
        if len(free_hull) < 3:
            return False
        span = free_hull.max(axis=0) - free_hull.min(axis=0)
        width_ok = min(span) >= self.robot_width + 2.0 * self.safety_margin
        depth_ok = max(span) >= self.robot_length + self.safety_margin
        area_ok = free_area >= self.area_threshold
        return bool(area_ok and width_ok and depth_ok)

    def apply_stability(self, raw):
        self.enter_history.append(bool(raw))
        hist = list(self.enter_history)
        if len(hist) >= self.stable_enter_frames and all(hist[-self.stable_enter_frames:]):
            return True
        if len(hist) >= self.stable_block_frames and not any(hist[-self.stable_block_frames:]):
            return False
        return False

    def publish_enterable(self, value):
        self.enter_history.append(False)
        self.enterable_pub.publish(Bool(data=bool(value)))
        self.legacy_flag_pub.publish(Int32(1 if value else 0))

    def make_polygon_msg(self, points_xy, stamp):
        msg = PolygonStamped()
        msg.header.frame_id = self.target_frame
        msg.header.stamp = stamp
        msg.polygon.points = [Point32(x=float(x), y=float(y), z=0.0) for x, y in points_xy]
        return msg

    def make_occupancy_grid_msg(self, occupancy, grid_info, stamp):
        msg = OccupancyGrid()
        msg.header.frame_id = self.target_frame
        msg.header.stamp = stamp
        msg.info = MapMetaData()
        msg.info.resolution = self.resolution
        msg.info.width = int(grid_info["width"])
        msg.info.height = int(grid_info["height"])
        msg.info.origin.position.x = float(grid_info["origin"][0])
        msg.info.origin.position.y = float(grid_info["origin"][1])
        msg.info.origin.orientation.w = 1.0
        msg.data = occupancy.astype(np.int8).ravel(order="C").tolist()
        return msg

    def make_markers(self, polygon, free_hull, enterable, free_area, person_count, stamp):
        markers = MarkerArray()
        markers.markers.append(self.make_line_marker(0, "elevator_area", polygon, stamp, (1.0, 0.8, 0.0, 1.0)))
        markers.markers.append(self.make_line_marker(1, "free_space_hull", free_hull, stamp, (0.0, 0.8, 0.2, 1.0)))
        text = Marker()
        text.header.frame_id = self.target_frame
        text.header.stamp = stamp
        text.ns = "elevator_metrics"
        text.id = 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.orientation.w = 1.0
        center = np.mean(polygon, axis=0)
        text.pose.position.x = float(center[0])
        text.pose.position.y = float(center[1])
        text.pose.position.z = 1.2
        text.scale.z = 0.18
        text.color.r = 0.0 if enterable else 1.0
        text.color.g = 1.0 if enterable else 0.0
        text.color.a = 1.0
        text.text = "ENTERABLE area={:.2f} persons={}".format(free_area, person_count) if enterable else \
            "BLOCKED area={:.2f} persons={}".format(free_area, person_count)
        markers.markers.append(text)
        return markers

    def make_line_marker(self, marker_id, ns, points_xy, stamp, rgba):
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.03
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba
        if len(points_xy) > 0:
            closed = np.vstack([points_xy, points_xy[0]])
            marker.points = [Point(x=float(x), y=float(y), z=0.02) for x, y in closed]
        return marker


if __name__ == "__main__":
    node = ElevatorSpaceMonitor()
    rospy.spin()
