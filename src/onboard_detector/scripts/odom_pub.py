#!/usr/bin/env python3
import rospy
import tf
from nav_msgs.msg import Odometry

def odom_callback(msg):
    br = tf.TransformBroadcaster()
    pos = msg.pose.pose.position
    ori = msg.pose.pose.orientation

    # 广播 odom -> base_link
    br.sendTransform(
        (pos.x, pos.y, pos.z),
        (ori.x, ori.y, ori.z, ori.w),
        msg.header.stamp,
        msg.child_frame_id,      # "base_link"
        msg.header.frame_id      # "odom"
    )

if __name__ == '__main__':
    rospy.init_node('odom_tf_broadcaster')
    rospy.Subscriber('/odom', Odometry, odom_callback)
    rospy.spin()
