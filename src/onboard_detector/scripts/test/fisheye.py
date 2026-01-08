import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class FisheyeUndistort:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node("fisheye_undistort", anonymous=True)
        self.bridge = CvBridge()

        # 鱼眼相机内参
        self.K = np.array([
            [354.12958248, 0.0, 626.70832345],
            [0.0, 345.88396395, 335.49813176],
            [0.0, 0.0, 1.0]
        ])
        self.D = np.array([0.00296807, 0.16182434, -0.16563839, 0.05418764])

        # 去畸变映射
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.K, (1280, 720), cv2.CV_16SC2
        )  # 假设分辨率为 1280x720，需根据实际调整

        # 发布和订阅
        self.image_pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=10)
        self.image_sub = rospy.Subscriber("/ahdcam_module/front/image/compressed", CompressedImage, self.image_callback)

    def image_callback(self, msg):
        try:
            # 解压图像
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # 去畸变
            undistorted_image = cv2.remap(cv_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
            # 转换为 ROS 消息
            image_msg = self.bridge.cv2_to_imgmsg(undistorted_image, encoding="bgr8")
            image_msg.header = msg.header
            image_msg.header.frame_id = "camera_color_frame"
            self.image_pub.publish(image_msg)
        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))

if __name__ == "__main__":
    try:
        node = FisheyeUndistort()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass