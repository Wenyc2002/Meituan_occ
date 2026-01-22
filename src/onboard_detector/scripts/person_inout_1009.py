#!/usr/bin/env python3
import rospy
import numpy as np
import math
import tf2_ros
import tf2_geometry_msgs
from shapely.geometry import Point, Polygon
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32
from spencer_tracking_msgs.msg import TrackedPersons, TrackedPerson

class ElevatorSpaceMonitor:
    def __init__(self):
        rospy.init_node("elevator_space_monitor", anonymous=True)

        # 电梯区域（map坐标）lab_elevator
        # self.elevator_points_map = [
        #     (13.9, 29.8), (15, 32),
        #     (16.2, 31.4), (15, 29.2)
        # ]
        # 电梯区域（map坐标）toilet_elevator1
        self.elevator_points_map = [
            (-8, -13.9), (-8.18, -12.8),
            (-6.26, -12.7), (-6.15, -13.6)
        ]
        # self.elevator_points_map = [
        # (16.5, 27.7), (13.1, 30),
        #     (11.8, 27.7),(15.2, 25.6)
        # ]


        # 地图参数
        self.resolution = rospy.get_param("~resolution", 0.1)
        self.width = rospy.get_param("~width", 80)
        self.height = rospy.get_param("~height", 80)
        self.origin_x = - (self.width * self.resolution) / 2.0
        self.origin_y = - (self.height * self.resolution) / 2.0

        # TF 监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 发布者
        self.map_pub = rospy.Publisher("/map_occ", OccupancyGrid, queue_size=1)
        self.elevator_free_pub = rospy.Publisher("/elevator_occupancy", Float32, queue_size=1)
        self.elevator_count_pub = rospy.Publisher("/elevator_person_count", Int32, queue_size=1)
        self.elevator_flag_pub = rospy.Publisher("/elevator_flag", Int32, queue_size=1)

        # 订阅者
        rospy.Subscriber("/scan_pcd", LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber("/tracked_persons_situ", TrackedPersons, self.tracked_persons_callback, queue_size=1)

        # 状态与计数
        self.person_state = {}        # {merged_id: inside_bool}
        self.entered_count = 0
        self.exited_count = 0
        self.last_transition_time = {}  # {merged_id: rospy.Time}
        self.min_transition_time = rospy.Duration(1.0)

        # 新增：轨迹融合（跨帧）
        self.last_seen = {}   # {merged_id: (x, y, time)}
        self.id_alias = {}    # {new_id: old_id}
        self.merge_distance = 0.1  # m，跨帧融合距离（用于 resolve_track_id）
        self.merge_time = rospy.Duration(2.0)
        self.min_fusion_dist = rospy.get_param("~min_fusion_dist", 0.1)  # 当前帧内融合最小距离

        # 记录每个 merged id 的上一次位置（用于判断是否穿过门线）
        self.prev_pos = {}  # {merged_id: (x, y)}

        rospy.loginfo("Elevator space monitor started")
        rospy.spin()

    # ======================
    # 激光雷达栅格计算（不变）
    # ======================
    def scan_callback(self, scan):
        grid = -1 * np.ones((self.height, self.width), dtype=np.int8)
        angle = scan.angle_min
        for r in scan.ranges:
            if scan.range_min < r < scan.range_max:
                steps = int(r / self.resolution)
                for s in range(steps):
                    x_free = s * self.resolution * math.cos(angle)
                    y_free = s * self.resolution * math.sin(angle)
                    gx = int((x_free - self.origin_x) / self.resolution)
                    gy = int((y_free - self.origin_y) / self.resolution)
                    if 0 <= gx < self.width and 0 <= gy < self.height:
                        grid[gy, gx] = 0
                x_occ = r * math.cos(angle)
                y_occ = r * math.sin(angle)
                gx = int((x_occ - self.origin_x) / self.resolution)
                gy = int((y_occ - self.origin_y) / self.resolution)
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    grid[gy, gx] = 100
            angle += scan.angle_increment

        occ = OccupancyGrid()
        occ.header.frame_id = "base_link"
        occ.header.stamp = rospy.Time.now()
        occ.info = MapMetaData()
        occ.info.resolution = self.resolution
        occ.info.width = self.width
        occ.info.height = self.height
        occ.info.origin.position.x = self.origin_x
        occ.info.origin.position.y = self.origin_y
        occ.info.origin.orientation.w = 1.0
        occ.data = grid.flatten().tolist()
        self.map_pub.publish(occ)

        # 电梯占用率发布
        elevator_polygon = self.transform_elevator_polygon()
        if elevator_polygon:
            free_ratio = self.compute_elevator_space(grid, elevator_polygon)

            # 发布电梯剩余空间百分比
            self.elevator_free_pub.publish(Float32(free_ratio * 100.0))

            # ===== 新增：根据 free space 发布 elevator flag =====
            # free_ratio ∈ [0,1]
            if free_ratio > 0.5:
                flag = 1
            else:
                flag = 0

            self.elevator_flag_pub.publish(Int32(flag))


    # ======================
    # TF 转换：map -> base_link
    # ======================
    def transform_elevator_polygon(self):
        try:
            transform = self.tf_buffer.lookup_transform("base_link", "map", rospy.Time(0), rospy.Duration(0.2))
        except:
            rospy.logwarn("TF transform map→base_link unavailable")
            return None

        transformed = []
        for x, y in self.elevator_points_map:
            pt = PointStamped()
            pt.header.frame_id = "map"
            pt.point.x, pt.point.y, pt.point.z = x, y, 0.0
            tp = tf2_geometry_msgs.do_transform_point(pt, transform)
            transformed.append((tp.point.x, tp.point.y))
        return transformed

    def compute_elevator_space(self, grid, polygon):
        poly = Polygon(polygon)
        total, occupied = 0, 0
        for gx in range(self.width):
            for gy in range(self.height):
                wx = self.origin_x + gx * self.resolution
                wy = self.origin_y + gy * self.resolution
                if poly.contains(Point(wx, wy)):
                    total += 1
                    if grid[gy, gx] != 0:
                        occupied += 1
        return 1 - (occupied / total) if total > 0 else 0.0

    def transform_elevator_polygon_to_frame(self, target_frame):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, "map", rospy.Time(0), rospy.Duration(0.2))
        except:
            rospy.logwarn(f"TF transform map→{target_frame} unavailable")
            return None

        transformed = []
        for x, y in self.elevator_points_map:
            pt = PointStamped()
            pt.header.frame_id = "map"
            pt.point.x, pt.point.y, pt.point.z = x, y, 0.0
            tp = tf2_geometry_msgs.do_transform_point(pt, transform)
            transformed.append((tp.point.x, tp.point.y))
        return transformed

    # ======================
    # 当前帧轨迹融合（把当前 frame 内靠得很近的多个 tid 合并）
    # 返回 dict: {fused_id: (x, y)}
    # ======================
    def fuse_tracks(self, tracks, min_dist):
        if not tracks:
            return {}
        # tracks: list of (tid, x, y)
        tracks = sorted(tracks, key=lambda t: t[0])  # 按tid排序处理（任意稳定顺序）
        fused = {}  # {fused_id: {'pos': (x, y), 'count': int}}
        for tid, x, y in tracks:
            assigned = None
            min_d = float('inf')
            for fid in fused:
                fx, fy = fused[fid]['pos']
                d = math.hypot(x - fx, y - fy)
                if d < min_dist and d < min_d:
                    min_d = d
                    assigned = fid
            if assigned is not None:
                old = fused[assigned]
                new_x = (old['pos'][0] * old['count'] + x) / (old['count'] + 1)
                new_y = (old['pos'][1] * old['count'] + y) / (old['count'] + 1)
                fused[assigned] = {'pos': (new_x, new_y), 'count': old['count'] + 1}
                rospy.logdebug(f"Fuse current {tid} into {assigned} (dist={min_d:.2f})")
            else:
                fused[tid] = {'pos': (x, y), 'count': 1}
        return {fid: data['pos'] for fid, data in fused.items()}

    # ======================
    # 主逻辑：轨迹融合 + 门线跨越判定（仅跨门线统计）
    # ======================
    def tracked_persons_callback(self, msg):
        elevator_polygon = self.transform_elevator_polygon_to_frame("livox_frame")
        if not elevator_polygon:
            return
        poly = Polygon(elevator_polygon)
        now = rospy.Time.now()

        door_p1 = elevator_polygon[0]
        door_p2 = elevator_polygon[3]

        # 计算电梯中心（用来判定哪一侧是内侧）
        elevator_center = np.mean(np.array(elevator_polygon), axis=0)

        # Step1: 收集当前帧所有 raw tracks (tid, x, y)
        raw_tracks = []
        for track in msg.tracks:
            px = track.pose.pose.position.x
            py = track.pose.pose.position.y
            tid = track.track_id
            raw_tracks.append((tid, px, py))

        # Step2: 当前帧内融合（把太靠近的同帧 id 合并）
        fused_current = self.fuse_tracks(raw_tracks, self.min_fusion_dist)

        # Step3: 对于每个融合后的ID，进行跨帧融合和门线跨越判断
        for fused_tid, (px, py) in fused_current.items():
            # 跨帧 ID 解析（resolve -> 返回 merged_id）
            pid = self.resolve_track_id(fused_tid, px, py, now)

            # 先拿到之前的位置（如果有）
            prev = self.prev_pos.get(pid, None)

            # 记录一下 current pos（后面会更新 prev_pos）
            # 但先做跨线判断需要 prev 存在
            if prev is not None:
                prev_x, prev_y = prev
                # 计算两侧的叉积值（side test）
                prev_side = self.side_of_line(prev_x, prev_y, door_p1[0], door_p1[1], door_p2[0], door_p2[1])
                curr_side = self.side_of_line(px, py, door_p1[0], door_p1[1], door_p2[0], door_p2[1])

                # 若符号改变，说明穿过了门线（忽略刚好==0 的精确落在线上的情况）
                if prev_side * curr_side < 0:
                    # 判断哪一侧是电梯内侧（用电梯中心点判断）
                    center_side = self.side_of_line(elevator_center[0], elevator_center[1],
                                                    door_p1[0], door_p1[1], door_p2[0], door_p2[1])
                    # 如果 center_side 与 curr_side 同号 => curr 在内侧
                    curr_is_inside = (center_side * curr_side) > 0
                    prev_is_inside = (center_side * prev_side) > 0

                    # 防抖：要求两次实际触发间隔大于 min_transition_time
                    last_time = self.last_transition_time.get(pid, rospy.Time(0))
                    if (now - last_time) > self.min_transition_time:
                        if (not prev_is_inside) and curr_is_inside:
                            # 外 -> 内 跨门线：进入
                            self.entered_count += 1
                            rospy.loginfo(f"Person {pid} ENTERED via door (crossed). Total entered: {self.entered_count}")
                            self.last_transition_time[pid] = now
                        elif prev_is_inside and (not curr_is_inside):
                            # 内 -> 外 跨门线：离开
                            self.exited_count += 1
                            rospy.loginfo(f"Person {pid} EXITED via door (crossed). Total exited: {self.exited_count}")
                            self.last_transition_time[pid] = now
                        # else: 符号变了但并未从内<->外（理论上不可能），忽略

            # 更新当前状态与历史位置（用于下次比较）
            self.person_state[pid] = poly.contains(Point(px, py))  # 可选：保持 polygon inside 状态
            self.prev_pos[pid] = (px, py)
            # last_seen 用于跨帧 id 合并
            self.last_seen[pid] = (px, py, now)

        rospy.loginfo(f"Entered={self.entered_count}, Exited={self.exited_count}")

    # ======================
    # 叉积侧别测试函数（点相对于有向线段的侧别）
    # 返回值：>0 左侧，<0 右侧，=0 在直线上
    # ======================
    def side_of_line(self, px, py, x1, y1, x2, y2):
        return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

    # ======================
    # ID 跨帧合并逻辑（与之前一致）
    # ======================
    def resolve_track_id(self, tid, x, y, now):
        # 如果之前已经合并过，直接返回旧ID
        if tid in self.id_alias:
            return self.id_alias[tid]

        # 查找最近的旧轨迹（在 last_seen 中）
        for old_id, (ox, oy, t_last) in self.last_seen.items():
            if (now - t_last) < self.merge_time:
                dist = math.hypot(x - ox, y - oy)
                if dist < self.merge_distance:
                    self.id_alias[tid] = old_id
                    rospy.logdebug(f"Merge track {tid} → {old_id} (dist={dist:.2f})")
                    return old_id

        # 没合并到已有 id，就沿用原 tid 作为新的 merged id
        return tid


if __name__ == "__main__":
    ElevatorSpaceMonitor()
