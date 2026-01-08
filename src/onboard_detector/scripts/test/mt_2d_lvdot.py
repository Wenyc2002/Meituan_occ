import numpy as np
import cv2
import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
import tf2_ros
import tf.transformations as tft
import sensor_msgs.point_cloud2 as pcl2
import subprocess

# 配置参数
cell_size = 0.1
map_range_x = (-70, 70)   # x: [-7, 7] 米，中心为0
map_range_y = (-70, 70)   # y: [-7, 7] 米，中心为0

def world_to_pixel(points_xyz, x_range, y_range, cell_size, h, w):
    points_xy = points_xyz[:, :2]
    # 逆时针旋转90度：(x, y) -> (-y, x)
    rotated_xy = np.zeros_like(points_xy)
    rotated_xy[:, 0] = -points_xy[:, 1]  # x' = -y
    rotated_xy[:, 1] = points_xy[:, 0]   # y' = x
    # 映射到图像中心
    px = (w / 2 + rotated_xy[:, 0] / cell_size).astype(np.int32)  # x' -> px
    py = (h / 2 - rotated_xy[:, 1] / cell_size).astype(np.int32)  # y' -> py (反向)
    
    # 确保像素坐标在图像范围内
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    return np.column_stack([px, py])

class MapGenerator:
    def __init__(self):
        self.cell_size = cell_size
        self.map_range_x = map_range_x
        self.map_range_y = map_range_y
        
        self.frame_count = 0
        
        # 增加TF缓冲区时间
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))  # 缓存10秒
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.w = int((self.map_range_y[1] - self.map_range_y[0]) / self.cell_size)  # 140
        self.h = int((self.map_range_x[1] - self.map_range_x[0]) / self.cell_size)  # 140
        
        # 发布器
        self.map_pub = rospy.Publisher('/occupancy_map', OccupancyGrid, queue_size=10)
        
        # 订阅器
        self.pc_sub = rospy.Subscriber('/onboard_detector/raw_lidar_point_cloud', PointCloud2, self.pc_callback)
        
        # 启动静态TF发布
        self.start_static_tf_publisher()

    def start_static_tf_publisher(self):
        """启动静态TF变换发布：map到base_link"""
        try:
            cmd = "rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map base_link 10"
            subprocess.Popen(cmd, shell=True)
            rospy.loginfo("Started static TF publisher: map -> base_link")
        except Exception as e:
            rospy.logerr("Failed to start static TF publisher: {}".format(str(e)))

    def pc_callback(self, pc_msg):
        timestamp = pc_msg.header.stamp
        
        try:
            # 获取最新的map到base_link的变换
            try:
                trans = self.tf_buffer.lookup_transform('map', 'base_link', timestamp, rospy.Duration(0.5))
            except tf2_ros.ExtrapolationException as e:
                latest_time = self.tf_buffer.get_latest_common_time('map', 'base_link')
                trans = self.tf_buffer.lookup_transform('map', 'base_link', latest_time, rospy.Duration(0.5))
                rospy.logwarn("Frame {}: Using latest TF at time {} due to extrapolation error: {}".format(self.frame_count, latest_time, str(e)))
            
            t = trans.transform.translation
            q = trans.transform.rotation
            trans_vec = (t.x, t.y, t.z)
            rot_quat = (q.x, q.y, q.z, q.w)
            T_map_to_base_link = tft.concatenate_matrices(
                tft.translation_matrix(trans_vec),
                tft.quaternion_matrix(rot_quat)
            )
            
            # 转换点云到map坐标系
            T_base_link_to_map = np.linalg.inv(T_map_to_base_link)
            
            # 获取点云数据
            points_list = list(pcl2.read_points(pc_msg, field_names=('x', 'y', 'z'), skip_nans=True))
            if not points_list:
                rospy.logwarn("Frame {}: Empty point cloud".format(self.frame_count))
                return
            points = np.array(points_list)
            
            # 转换到map坐标系
            points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
            points_map_homo = (T_base_link_to_map @ points_homo.T).T
            points_map = points_map_homo[:, :3]
            
            # 创建占据地图
            mask = (points_map[:, 0] > self.map_range_x[0]) & (points_map[:, 0] < self.map_range_x[1]) & \
                   (points_map[:, 1] > self.map_range_y[0]) & (points_map[:, 1] < self.map_range_y[1]) & \
                   (points_map[:, 2] > -0.1) & (points_map[:, 2] < 0.65)
            points_map = points_map[mask]
            
            map2d = np.ones([self.h, self.w], dtype=np.uint8) * 255
            points_2d = points_map[:, :2]
            points_rasterized = world_to_pixel(points_2d, self.map_range_x, self.map_range_y, self.cell_size, self.h, self.w)
            
            valid_mask = (points_rasterized[:, 0] >= 0) & (points_rasterized[:, 0] < self.w) & \
                         (points_rasterized[:, 1] >= 0) & (points_rasterized[:, 1] < self.h)
            points_rasterized = points_rasterized[valid_mask]
            points_rasterized = np.unique(points_rasterized, axis=0)
            rospy.loginfo("Frame {}: Rasterized points shape: {}".format(self.frame_count, points_rasterized.shape))
            
            map2d[points_rasterized[:, 1], points_rasterized[:, 0]] = 0
            
            # 转换为OccupancyGrid
            grid_msg = OccupancyGrid()
            grid_msg.header.stamp = timestamp
            grid_msg.header.frame_id = 'map'
            grid_msg.info.resolution = self.cell_size
            grid_msg.info.width = self.w
            grid_msg.info.height = self.h
            
            # 设置地图原点，使几何中心与map的(0,0,0)重合
            grid_msg.info.origin = Pose()
            grid_msg.info.origin.position.x = self.map_range_x[0]  # -7.0
            grid_msg.info.origin.position.y = self.map_range_y[0]  # -7.0
            grid_msg.info.origin.position.z = 0.0
            # 设置逆时针90度旋转（绕z轴）
            grid_msg.info.origin.orientation.x = 0.0
            grid_msg.info.origin.orientation.y = 0.0
            grid_msg.info.origin.orientation.z = 0.7071067811865475  # sin(π/4)
            grid_msg.info.origin.orientation.w = 0.7071067811865475  # cos(π/4)
            
            # 转换map2d为OccupancyGrid数据
            grid_data = np.ones((self.h * self.w), dtype=np.int8) * -1  # 默认未知
            grid_data[map2d.flatten() == 255] = 0  # 未占据
            grid_data[map2d.flatten() == 0] = 100  # 占据
            grid_msg.data = grid_data.tolist()
            
            self.map_pub.publish(grid_msg)
            rospy.loginfo("Frame {}: Published occupancy map".format(self.frame_count))
            
            self.frame_count += 1
        
        except tf2_ros.LookupException as e:
            rospy.logerr("Frame {}: TF Lookup error: {}".format(self.frame_count, str(e)))
        except tf2_ros.ConnectivityException as e:
            rospy.logerr("Frame {}: TF Connectivity error: {}".format(self.frame_count, str(e)))
        except tf2_ros.ExtrapolationException as e:
            rospy.logerr("Frame {}: TF Extrapolation error: {}".format(self.frame_count, str(e)))
        except Exception as e:
            rospy.logerr("Frame {}: Error processing frame: {}".format(self.frame_count, str(e)))

if __name__ == '__main__':
    rospy.init_node('map_generator', anonymous=True)
    mg = MapGenerator()
    rospy.spin()