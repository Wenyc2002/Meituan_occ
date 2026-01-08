#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
from cv_bridge import CvBridge
import cv2

class SensorProcessing:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node("sensor_processing", anonymous=True)
        self.bridge = CvBridge()

        # 鱼眼相机内参
        self.K_fisheye = np.array([
            [354.12958248, 0.0, 626.70832345],
            [0.0, 345.88396395, 335.49813176],
            [0.0, 0.0, 1.0]
        ])
        self.D_fisheye = np.array([0.00296807, 0.16182434, -0.16563839, 0.05418764])

        # 图像分辨率
        self.width = 1280
        self.height = 720

        # 去畸变映射
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K_fisheye, self.D_fisheye, np.eye(3), self.K_fisheye, (self.width, self.height), cv2.CV_16SC2
        )

        # 发布和订阅
        self.image_pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=10)
        self.image_sub = rospy.Subscriber("/ahdcam_module/front/image/compressed", CompressedImage, self.image_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber("/localization_module/pose_global_high_frequency", Odometry, self.odom_callback, queue_size=10)
        self.pose_pub = rospy.Publisher("/mavros/local_position/pose", PoseStamped, queue_size=10)

        rospy.loginfo("Sensor processing node started.")
        rospy.loginfo("Subscribing to: /ahdcam_module/front/image/compressed, /localization_module/pose_global_high_frequency")
        rospy.loginfo("Publishing to: /camera/color/image_raw, /mavros/local_position/pose")

    def image_callback(self, msg):
        try:
            # 解压图像
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # 去畸变
            undistorted_image = cv2.remap(cv_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
            # 发布去畸变图像
            image_msg = self.bridge.cv2_to_imgmsg(undistorted_image, encoding="bgr8")
            image_msg.header = msg.header
            image_msg.header.frame_id = "camera_color_frame"
            self.image_pub.publish(image_msg)
        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))

    def odom_callback(self, msg):
        # 创建 PoseStamped 消息
        pose_stamped_msg = PoseStamped()
        # 复制 header 和 pose 部分
        pose_stamped_msg.header = msg.header
        pose_stamped_msg.pose = msg.pose.pose
        # 发布转换后的消息
        self.pose_pub.publish(pose_stamped_msg)
        rospy.logdebug("Converted and published PoseStamped message with stamp: %s", msg.header.stamp)

if __name__ == "__main__":
    try:
        node = SensorProcessing()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Sensor processing node shut down.")