import os
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PolygonStamped
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
import tf2_ros
import tf.transformations as tft
import message_filters
import sensor_msgs.point_cloud2 as pcl2

# 配置参数
cell_size = 0.1
map_range_x = (-5, 12)  # x: [-5, 12] 米，扩展范围
map_range_y = (-7, 7)   # y: [-7, 7] 米，扩展范围
freespace_threshold = 0.25  # 机器人进入电梯的最小未占据面积（平方米）

# 电梯多边形数据（map坐标系）
elevator_points_map_default = np.array([
    [31.43711472, 19.42847443, 0.0],
    [32.37048721, 21.19874763, 0.0],
    [30.90929413, 21.95504951, 0.0],
    [30.0013237, 20.11806488, 0.0]
])

# 订阅的话题列表
subscribed_topics = [
    '/lidar_module/livox/lidar',
    '/tof_module/forward/pointcloud',
    '/tof_module/downward/pointcloud'
]

# 静态变换矩阵
T_base_footprint_to_base_link = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.1],
    [0, 0, 0, 1]
])
T_base_link_to_sensor_kit = np.array([
    [1, 0, 0, 0.345],
    [0, 1, 0, 0],
    [0, 0, 1, 0.6274],
    [0, 0, 0, 1]
])
T_sensor_kit_to_livox = np.eye(4)
T_livox_to_livox_frame = np.eye(4)

def world_to_pixel(points_xyz, x_range, y_range, cell_size, h, w):
    points_xy = points_xyz[:, :2]
    # 将LiDAR坐标系的(0,0)映射到图像中心
    # x轴（LiDAR）映射到图像y轴（垂直方向），x=0映射到h/2
    py = (h / 2 - points_xy[:, 0] / cell_size).astype(np.int32)
    # y轴（LiDAR）映射到图像x轴（水平方向），y=0映射到w/2
    px = (w / 2 + points_xy[:, 1] / cell_size).astype(np.int32)
    
    # 确保像素坐标在图像范围内
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    return np.column_stack([px, py])

# 计算电梯内的未占据面积
def calculate_freespace_area(map2d, elevator_pixel, cell_size):
    """计算多边形区域内的未占据面积（平方米）"""
    freespace_pixels = 0
    for y in range(map2d.shape[0]):
        for x in range(map2d.shape[1]):
            if map2d[y, x] == 255:  # 未占据像素
                if cv2.pointPolygonTest(elevator_pixel, (x, y), False) >= 0:  # 点在多边形内或边界上
                    freespace_pixels += 1
    freespace_area = freespace_pixels * (cell_size ** 2)  # 像素数转换为面积
    return freespace_area

class MapGenerator:
    def __init__(self):
        self.cell_size = cell_size
        self.map_range_x = map_range_x
        self.map_range_y = map_range_y
        self.freespace_threshold = freespace_threshold
        
        self.elevator_points_map = elevator_points_map_default
        
        self.frame_count = 0
        
        # 加载extrinsics
        npz_file = np.load('/home/yanci/depth/mt_code/extrinsics.npz', allow_pickle=True)
        self.extrinsics = {key: npz_file[key] for key in npz_file.files}
        rospy.loginfo("Extrinsics keys: {}".format(list(self.extrinsics.keys())))
        
        # 验证extrinsics键
        for topic in subscribed_topics:
            if topic not in self.extrinsics:
                rospy.logwarn("Extrinsic for {} not found in npz file".format(topic))
        
        # 增加TF缓冲区时间
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))  # 缓存10秒
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.T_base_footprint_to_base_link = T_base_footprint_to_base_link
        self.T_base_link_to_sensor_kit = T_base_link_to_sensor_kit
        self.T_sensor_kit_to_livox = T_sensor_kit_to_livox
        self.T_livox_to_livox_frame = T_livox_to_livox_frame
        
        self.w = int((self.map_range_y[1] - self.map_range_y[0]) / self.cell_size)
        self.h = int((self.map_range_x[1] - self.map_range_x[0]) / self.cell_size)
        
        # 发布器
        self.map_pub = rospy.Publisher('/occupancy_map', OccupancyGrid, queue_size=10)
        
        # 订阅器
        sub1 = message_filters.Subscriber(subscribed_topics[0], PointCloud2)
        sub2 = message_filters.Subscriber(subscribed_topics[1], PointCloud2)
        sub3 = message_filters.Subscriber(subscribed_topics[2], PointCloud2)
        
        self.sync = message_filters.ApproximateTimeSynchronizer([sub1, sub2, sub3], queue_size=10, slop=0.2)
        self.sync.registerCallback(self.pc_callback)
        
        self.elev_sub = rospy.Subscriber('/elevator_area', PolygonStamped, self.elev_callback)
    
    def elev_callback(self, msg):
        if msg.header.frame_id != 'map':
            rospy.logwarn("Elevator area frame_id is not 'map', got {}".format(msg.header.frame_id))
            return
        points = np.array([[p.x, p.y, p.z] for p in msg.polygon.points])
        self.elevator_points_map = points
        rospy.loginfo("Updated elevator_points_map")
    
    def pc_callback(self, pc1, pc2, pc3):
        timestamp = pc1.header.stamp  # 使用第一个点云的时间戳
        
        try:
            # 获取最新的map到base_footprint的变换
            try:
                trans = self.tf_buffer.lookup_transform('map', 'base_footprint', timestamp, rospy.Duration(0.5))
            except tf2_ros.ExtrapolationException as e:
                latest_time = self.tf_buffer.get_latest_common_time('map', 'base_footprint')
                trans = self.tf_buffer.lookup_transform('map', 'base_footprint', latest_time, rospy.Duration(0.5))
                rospy.logwarn("Frame {}: Using latest TF at time {} due to extrapolation error: {}".format(self.frame_count, latest_time, str(e)))
            
            t = trans.transform.translation
            q = trans.transform.rotation
            trans_vec = (t.x, t.y, t.z)
            rot_quat = (q.x, q.y, q.z, q.w)
            T_map_to_base_footprint = tft.concatenate_matrices(
                tft.translation_matrix(trans_vec),
                tft.quaternion_matrix(rot_quat)
            )
            
            # 计算map到livox的完整变换
            T_map_to_livox = T_map_to_base_footprint @ self.T_base_footprint_to_base_link @ \
                             self.T_base_link_to_sensor_kit @ self.T_sensor_kit_to_livox @ \
                             self.T_livox_to_livox_frame
            T_livox_to_map = np.linalg.inv(T_map_to_livox)
            rospy.loginfo("Frame {}: T_livox_to_map:\n{}".format(self.frame_count, T_livox_to_map))
            
            # 转换多边形到livox坐标系
            elevator_points_map_homo = np.hstack([self.elevator_points_map, np.ones((self.elevator_points_map.shape[0], 1))])
            elevator_points_lidar_homo = (T_livox_to_map @ elevator_points_map_homo.T).T
            elevator_points_lidar = elevator_points_lidar_homo[:, :3]
            rospy.loginfo("Frame {}: elevator_points_lidar:\n{}".format(self.frame_count, elevator_points_lidar))
            
            # 像素映射
            elevator_pixel = world_to_pixel(elevator_points_lidar, self.map_range_x, self.map_range_y, self.cell_size, self.h, self.w)
            rospy.loginfo("Frame {}: elevator_pixel:\n{}".format(self.frame_count, elevator_pixel))
            
            # 检查像素坐标有效性
            if np.any(elevator_pixel < 0) or np.any(elevator_pixel[:, 0] >= self.w) or np.any(elevator_pixel[:, 1] >= self.h):
                rospy.logwarn("Frame {}: elevator_pixel out of bounds".format(self.frame_count))
            
            # 收集点云数据（直接在livox_frame中处理）
            list_points = []
            pcs = [pc1, pc2, pc3]
            for i, pc in enumerate(pcs):
                topic = subscribed_topics[i]
                # 获取xyz
                points_list = list(pcl2.read_points(pc, field_names=('x', 'y', 'z'), skip_nans=True))
                if not points_list:
                    rospy.logwarn("Frame {}: Empty point cloud for {}".format(self.frame_count, topic))
                    continue
                points = np.array(points_list)
                
                if topic in self.extrinsics:
                    T2lidar = np.array(self.extrinsics[topic])
                    points_at_lidar = np.dot(T2lidar[:3, :3], points.T).T + T2lidar[:3, 3]
                    list_points.append(points_at_lidar)
                else:
                    rospy.logwarn("Frame {}: No extrinsic for {}, using raw points".format(self.frame_count, topic))
                    list_points.append(points)
            
            if len(list_points) < 2:
                rospy.logwarn("Frame {}: Insufficient point cloud data, got {} topics".format(self.frame_count, len(list_points)))
                return
            
            # 创建占据地图
            points = np.concatenate(list_points, axis=0)
            mask = (points[:, 0] > self.map_range_x[0]) & (points[:, 0] < self.map_range_x[1]) & \
                   (points[:, 1] > self.map_range_y[0]) & (points[:, 1] < self.map_range_y[1]) & \
                   (points[:, 2] > -0.1) & (points[:, 2] < 0.65)
            points = points[mask]
            
            map2d = np.ones([self.h, self.w], dtype=np.uint8) * 255
            points_2d = points[:, :2]
            points_rasterized = world_to_pixel(points_2d, self.map_range_x, self.map_range_y, self.cell_size, self.h, self.w)
            
            valid_mask = (points_rasterized[:, 0] >= 0) & (points_rasterized[:, 0] < self.w) & \
                         (points_rasterized[:, 1] >= 0) & (points_rasterized[:, 1] < self.h)
            points_rasterized = points_rasterized[valid_mask]
            points_rasterized = np.unique(points_rasterized, axis=0)
            rospy.loginfo("Frame {}: Rasterized points shape: {}".format(self.frame_count, points_rasterized.shape))
            
            map2d[points_rasterized[:, 1], points_rasterized[:, 0]] = 0
            
            # 绘制电梯多边形
            if elevator_pixel.shape[0] >= 3:  # 确保多边形有效
                elevator_pixel = elevator_pixel.astype(np.int32)
                # 使用OpenCV在map2d上绘制电梯多边形（值为50，区分于0=occupied, 255=free）
                cv2.fillPoly(map2d, [elevator_pixel], 50)
                rospy.loginfo("Frame {}: Elevator polygon drawn on map2d".format(self.frame_count))
            else:
                rospy.logwarn("Frame {}: Invalid elevator polygon, cannot draw".format(self.frame_count))
            
            # 转换为OccupancyGrid
            grid_msg = OccupancyGrid()
            grid_msg.header.stamp = timestamp
            grid_msg.header.frame_id = 'livox_frame'
            grid_msg.info.resolution = self.cell_size
            grid_msg.info.width = self.w
            grid_msg.info.height = self.h
            
            # 设置地图原点为LiDAR坐标系的(0,0,0)
            grid_msg.info.origin = Pose()
            grid_msg.info.origin.position.x = 0.0
            grid_msg.info.origin.position.y = 0.0
            grid_msg.info.origin.position.z = 0.0
            grid_msg.info.origin.orientation.w = 1.0
            
            # 转换map2d为OccupancyGrid数据（0=free, 100=occupied, 50=elevator, -1=unknown）
            grid_data = np.ones((self.h * self.w), dtype=np.int8) * -1  # 默认未知
            grid_data[map2d.flatten() == 255] = 0  # 未占据
            grid_data[map2d.flatten() == 0] = 100  # 占据
            grid_data[map2d.flatten() == 50] = 50  # 电梯区域
            grid_msg.data = grid_data.tolist()
            
            self.map_pub.publish(grid_msg)
            rospy.loginfo("Frame {}: Published occupancy map with elevator polygon".format(self.frame_count))
            
            # 计算未占据面积（用于日志）
            can_enter = False
            if elevator_pixel.shape[0] >= 3:
                freespace_area = calculate_freespace_area(map2d, elevator_pixel, self.cell_size)
                can_enter = freespace_area >= self.freespace_threshold
                status_text = "Frame {}: Freespace: {:.2f} m², {}".format(self.frame_count, freespace_area, 'Can Enter' if can_enter else 'Cannot Enter')
                rospy.loginfo(status_text)
            else:
                status_text = "Frame {}: Invalid polygon, Cannot Enter".format(self.frame_count)
                rospy.logwarn(status_text)
            
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