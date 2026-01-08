#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs  # 必须有，否则transform不了PoseStamped
from geometry_msgs.msg import PoseStamped
from spencer_tracking_msgs.msg import TrackedPersons, TrackedPerson

from visualization_msgs.msg import Marker, MarkerArray


class TrackVisualizer:
    def __init__(self):
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscriber
        rospy.Subscriber("/tracked_persons", TrackedPersons, self.callback)

        # Publisher for markers
        self.marker_pub = rospy.Publisher("/tracked_persons_markers", MarkerArray, queue_size=10)

    def callback(self, msg):
        marker_array = MarkerArray()

        for idx, t in enumerate(msg.tracks):
            pose_livox = PoseStamped()
            pose_livox.header = msg.header  # frame_id=livox_frame
            pose_livox.pose = t.pose.pose   # PoseWithCovariance → Pose

            try:
                pose_map = self.tf_buffer.transform(pose_livox, "map", rospy.Duration(0.5))

                m = Marker()
                m.header.frame_id = "map"   # Marker坐标系
                m.header.stamp = rospy.Time.now()
                m.ns = "tracked_persons"
                m.id = t.track_id           # 每个人一个唯一id
                m.type = Marker.SPHERE      # 球体(在2D看就是圆)
                m.action = Marker.ADD
                m.pose.position.x = pose_map.pose.position.x
                m.pose.position.y = pose_map.pose.position.y
                m.pose.position.z = 0.1     # 稍微抬高一点
                m.pose.orientation.w = 1.0
                m.scale.x = 0.4  # 直径
                m.scale.y = 0.4
                m.scale.z = 0.1
                m.color.r = 1.0  # 红色
                m.color.g = 0.0
                m.color.b = 0.0
                m.color.a = 0.8  # 透明度

                marker_array.markers.append(m)

            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn("TF transform failed for track_id {}: {}".format(t.track_id, e))

        # 清除旧marker：把没有出现的id也清掉
        # 可以定期发布空MarkerArray，或用DELETEALL
        self.marker_pub.publish(marker_array)

if __name__ == "__main__":
    rospy.init_node("tracked_persons_visualizer")
    TrackVisualizer()
    rospy.spin()
