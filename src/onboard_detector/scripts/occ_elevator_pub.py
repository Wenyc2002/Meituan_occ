#!/usr/bin/env python3
import rospy
import numpy as np
import math
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData
from shapely.geometry import Point, Polygon

class ElevatorSpaceMonitor:
    def __init__(self):
        rospy.init_node("elevator_space_monitor", anonymous=True)

    
        self.elevator_points_map = [
        (13.9, 29.8),(15, 32),
                (16.2, 31.4),(15, 29.2)
    ]



        # 地图参数 (以 base_link 为中心)
        self.resolution = rospy.get_param("~resolution", 0.1)
        self.width = rospy.get_param("~width", 80)
        self.height = rospy.get_param("~height", 80)
        self.origin_x = - (self.width * self.resolution) / 2.0
        self.origin_y = - (self.height * self.resolution) / 2.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.map_pub = rospy.Publisher("/map_occ", OccupancyGrid, queue_size=1)
        rospy.Subscriber("/scan_pcd", LaserScan, self.scan_callback, queue_size=1)

        rospy.loginfo("Elevator space monitor started")
        rospy.spin()

    def transform_elevator_polygon(self):
        """将电梯多边形从 map → base_link 坐标系转换"""
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                "map",
                rospy.Time(0),
                rospy.Duration(0.2)
            )
        except:
            rospy.logwarn("TF transform map→base_link unavailable")
            return None

        transformed = []
        for x, y in self.elevator_points_map:
            pt = PointStamped()
            pt.header.frame_id = "map"
            pt.point.x = x
            pt.point.y = y
            pt.point.z = 0.0
            tp = tf2_geometry_msgs.do_transform_point(pt, transform)
            transformed.append((tp.point.x, tp.point.y))
        return transformed

    def scan_callback(self, scan):
        # 初始化栅格 (-1 表示未知)
        grid = -1 * np.ones((self.height, self.width), dtype=np.int8)

        angle = scan.angle_min
        for r in scan.ranges:
            if scan.range_min < r < scan.range_max:
                # 1. 标记激光路径为空闲
                steps = int(r / self.resolution)
                for s in range(steps):
                    x_free = s * self.resolution * math.cos(angle)
                    y_free = s * self.resolution * math.sin(angle)
                    gx = int((x_free - self.origin_x) / self.resolution)
                    gy = int((y_free - self.origin_y) / self.resolution)
                    if 0 <= gx < self.width and 0 <= gy < self.height:
                        grid[gy, gx] = 0  # 空闲区域

                # 2. 最后一段为障碍物
                x_occ = r * math.cos(angle)
                y_occ = r * math.sin(angle)
                gx = int((x_occ - self.origin_x) / self.resolution)
                gy = int((y_occ - self.origin_y) / self.resolution)
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    grid[gy, gx] = 100  # 占据
            angle += scan.angle_increment

        # 发布占据地图
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

        # 计算电梯区域占用情况
        elevator_polygon = self.transform_elevator_polygon()
        if elevator_polygon:
            free_ratio = self.compute_elevator_space(grid, elevator_polygon)
            rospy.loginfo(f"Elevator free space: {free_ratio*100:.2f}%")

    def compute_elevator_space(self, grid, polygon):
        """计算电梯内剩余空间比例 (未知区域视为占据)"""
        poly = Polygon(polygon)
        total = 0
        occupied = 0

        for gx in range(self.width):
            for gy in range(self.height):
                wx = self.origin_x + gx * self.resolution
                wy = self.origin_y + gy * self.resolution
                if poly.contains(Point(wx, wy)):
                    total += 1
                    # 只认为 grid=0 是空闲，其余 (-1 和 100) 都当作占据
                    if grid[gy, gx] != 0:
                        occupied += 1

        free_ratio = 1 - (occupied / total) if total > 0 else 0.0
        return free_ratio

if __name__ == "__main__":
    ElevatorSpaceMonitor()
