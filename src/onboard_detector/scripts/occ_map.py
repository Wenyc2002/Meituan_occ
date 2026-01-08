#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData
import math

class LaserToOccMap:
    def __init__(self):
        rospy.init_node("laserscan_to_occmap", anonymous=True)

        # 地图参数
        self.resolution = rospy.get_param("~resolution", 0.1)  # 每栅格大小（米）
        self.width = rospy.get_param("~width", 80)  # 栅格宽度（单位：格）
        self.height = rospy.get_param("~height", 80)  # 栅格高度（单位：格）

        # 以机器人为中心，origin 设为地图中心负半宽
        self.origin_x = - (self.width * self.resolution) / 2.0
        self.origin_y = - (self.height * self.resolution) / 2.0

        self.map_pub = rospy.Publisher("/map_occ", OccupancyGrid, queue_size=1)
        rospy.Subscriber("/scan_pcd", LaserScan, self.callback, queue_size=1)

        rospy.loginfo("LaserScan → OccupancyGrid (base_link-centered) started")
        rospy.spin()

    def callback(self, scan):
        # 初始化地图 (-1: 未知, 0: 空闲, 100: 占据)
        grid = -1 * np.ones((self.height, self.width), dtype=np.int8)

        angle = scan.angle_min
        for r in scan.ranges:
            if scan.range_min < r < scan.range_max:
                x = r * math.cos(angle)  # 雷达坐标系下
                y = r * math.sin(angle)
                gx = int((x - self.origin_x) / self.resolution)
                gy = int((y - self.origin_y) / self.resolution)
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    grid[gy, gx] = 100  # 占据
            angle += scan.angle_increment

        occ = OccupancyGrid()
        occ.header.frame_id = "base_link"  # 地图始终绑定机器人
        occ.header.stamp = rospy.Time.now()
        occ.info = MapMetaData()
        occ.info.resolution = self.resolution
        occ.info.width = self.width
        occ.info.height = self.height
        occ.info.origin.position.x = self.origin_x
        occ.info.origin.position.y = self.origin_y
        occ.data = grid.flatten().tolist()

        self.map_pub.publish(occ)

if __name__ == "__main__":
    LaserToOccMap()
