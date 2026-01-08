#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PolygonStamped, Point32

def publish_polygon():
    rospy.init_node('polygon_marker_node')
    pub = rospy.Publisher('/elevator_area', PolygonStamped, queue_size=1, latch=True)

    poly = PolygonStamped()
    poly.header.frame_id = "map"  
    poly.header.stamp = rospy.Time.now()

    points = [
        (13.9, 29.8),(15, 32),
                (16.2, 31.4),(15, 29.2)
    ]
    
    # points = [
    #     (16.5, 27.7), (13.1, 30),
    #         (11.8, 27.7),(15.2, 25.6)
    #     ]


    for x, y in points:
        p = Point32()
        p.x = x
        p.y = y
        p.z = 0.0
        poly.polygon.points.append(p)

    pub.publish(poly)
    rospy.loginfo("Polygon published.")

    rospy.spin()

if __name__ == "__main__":
    publish_polygon()
