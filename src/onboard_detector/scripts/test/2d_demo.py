import os
import numpy as np
import cv2
from utils_ros2 import read_ros2db3, deserialize_msg

# 配置参数
path_db3 = '/media/yanci/Yc/DATA/20250516_111434'
path_img = '/media/yanci/Yc/DATA/524'
cell_size = 0.1
map_range_x = (-5, 12)  # x: [-5, 12] 米，扩展范围
map_range_y = (-7, 7)   # y: [-7, 7] 米，扩展范围
interval = 1e8  # 100ms per frame
freespace_threshold = 0.25  # 机器人进入电梯的最小未占据面积（平方米）

# 电梯多边形数据（map坐标系）
elevator_points_map_default = np.array([
    [31.43711472, 19.42847443, 0.0],
    [32.37048721, 21.19874763, 0.0],
    [30.90929413, 21.95504951, 0.0],
    [30.0013237, 20.11806488, 0.0]
])

# 导出的话题列表
exported_topics = [
    '/lidar_module/livox/lidar',
    '/tof_module/forward/pointcloud',
    '/tof_module/downward/pointcloud',
    '/tf',
    '/elevator_area'  # 添加电梯区域话题
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

# 改进的PointCloud2反序列化函数
def deserialize_pointcloud2(msg):
    point_step = msg.point_step
    point_count = msg.width * msg.height
    dtype_list = [(field.name, np.float32) for field in msg.fields if field.name in ['x', 'y', 'z']]
    if len(dtype_list) < 3:
        raise ValueError("PointCloud2消息中未找到x,y,z字段")
    dtype = np.dtype(dtype_list)
    points = np.frombuffer(msg.data, dtype=dtype, count=point_count)
    xyz = np.column_stack([points['x'], points['y'], points['z']])
    return {'xyz': xyz}

# 解析PolygonStamped消息
def deserialize_polygon_stamped(msg):
    points = [[p.x, p.y, p.z] for p in msg.polygon.points]
    return np.array(points)

def world_to_pixel(points_xyz, x_range, y_range, cell_size, h, w):
    points_xy = points_xyz[:, :2]
    # x轴
    py = ((x_range[1] - points_xy[:, 0]) / cell_size).astype(np.int32)
    # y轴
    px = ((y_range[1] - points_xy[:, 1]) / cell_size).astype(np.int32)  
    
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    return np.column_stack([px, py])
# 四元数到旋转矩阵
def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

# 动态TF获取
def get_tf_transform(tf_data, timestamp, max_time_diff=0.1):
    """获取map→odom和odom→base_footprint的变换矩阵"""
    if not tf_data:
        print("Error: Empty TF data")
        return None, None

    tf_timestamps = np.array(sorted(tf_data.keys()))
    time_diffs = np.abs(tf_timestamps - timestamp) / 1e9
    
    if np.min(time_diffs) > max_time_diff:
        print(f"Error: No TF data within {max_time_diff}s (min_diff: {np.min(time_diffs):.3f}s)")
        return None, None
    
    closest_idx = np.argmin(time_diffs)
    closest_tf = tf_timestamps[closest_idx]
    tf_msg = tf_data[closest_tf]
    
    T_map_to_odom = None
    T_odom_to_base_footprint = None
    
    for transform in tf_msg.transforms:
        try:
            q = np.array([transform.transform.rotation.x,
                         transform.transform.rotation.y,
                         transform.transform.rotation.z,
                         transform.transform.rotation.w])
            if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-3):
                print(f"Warning: Unnormalized quaternion in {transform.header.frame_id}→{transform.child_frame_id}")
                q = q / np.linalg.norm(q)
            
            R = quaternion_to_rotation_matrix(*q)
            t = np.array([transform.transform.translation.x,
                         transform.transform.translation.y,
                         transform.transform.translation.z])
            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            
            if transform.header.frame_id == '/map' and transform.child_frame_id == '/odom':
                T_map_to_odom = T
            elif transform.header.frame_id == '/odom' and transform.child_frame_id == '/base_footprint':
                T_odom_to_base_footprint = T
                
        except Exception as e:
            print(f"Error processing {transform.header.frame_id}→{transform.child_frame_id}: {str(e)}")
            continue
    
    if T_map_to_odom is None or T_odom_to_base_footprint is None:
        print("Error: Missing required TF transform")
        return None, None
    
    return T_map_to_odom, T_odom_to_base_footprint

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

# 读取数据
topic_type_data, _ = read_ros2db3(path_db3, 'ROS2_HUMBLE', exported_topics)

# 同步函数
def synchronize(topic_type_data, topic_names, interval):
    dict_timestamps = {}
    all_timestamps = []
    for name in topic_names:
        if name in topic_type_data:
            timestamps = np.array(sorted(list(topic_type_data[name]['data'].keys())))
            dict_timestamps[name] = timestamps
            all_timestamps.append(timestamps)
    keys = np.concatenate([t // interval for t in all_timestamps]).astype(np.int64)
    list_frame_id = np.unique(keys)
    print('frame number: ', len(list_frame_id))
    print("Frame IDs range: [{}, {}]".format(list_frame_id[0], list_frame_id[-1]))
    for topic in topic_names:
        if topic in dict_timestamps:
            print(f"Timestamps for {topic}: [{dict_timestamps[topic][0]}, {dict_timestamps[topic][-1]}]")
    return list_frame_id, dict_timestamps

list_frame_id, dict_timestamps = synchronize(topic_type_data, exported_topics, interval)

# 加载extrinsics
npz_file = np.load('/home/yanci/depth/mt_code/extrinsics.npz', allow_pickle=True)
extrinsics = {key: npz_file[key] for key in npz_file.files}
print("Extrinsics keys:", list(extrinsics.keys()))

# 验证extrinsics键
expected_topics = exported_topics[:3]
for topic in expected_topics:
    if topic not in extrinsics:
        print(f"Warning: Extrinsic for {topic} not found in npz file")

# 处理 /elevator_area 数据
elevator_area_topic = '/elevator_area'
elevator_points_map = elevator_points_map_default  # 初始多边形
elevator_timestamps = []
if elevator_area_topic in topic_type_data:
    elevator_timestamps = sorted(list(topic_type_data[elevator_area_topic]['data'].keys()))
    print(f"Found {len(elevator_timestamps)} /elevator_area messages at timestamps: {elevator_timestamps}")
else:
    print("Warning: No /elevator_area data found, using default elevator points")

dir_save = os.path.join(path_db3, '0811')#######################################
os.makedirs(dir_save, exist_ok=True)

# TF数据
tf_topic = '/tf'
tf_data = topic_type_data[tf_topic]['data'] if tf_topic in topic_type_data else {}

# 处理每一帧
w = int((map_range_y[1] - map_range_y[0]) / cell_size)  # w=140，对应y范围
h = int((map_range_x[1] - map_range_x[0]) / cell_size)  # h=170，对应x范围
for frame_id in list_frame_id[:50]: ### 处理前多少帧
    try:
        # 查找当前帧的点云时间戳
        timestamp = None
        for topic in exported_topics[:3]:
            if topic in dict_timestamps:
                np_topic_timestamp = dict_timestamps[topic]
                idx = np.argwhere(np.abs(np_topic_timestamp - (frame_id * interval)) < interval // 2)
                if len(idx) == 1:
                    timestamp = np_topic_timestamp[idx].item()
                    break
        if timestamp is None:
            print(f"Frame {frame_id}: No valid point cloud timestamp found")
            continue

        # 更新多边形（如果有新的 /elevator_area 数据）
        if elevator_timestamps:
            # 查找最近的 /elevator_area 时间戳
            time_diffs = np.abs(np.array(elevator_timestamps) - timestamp)
            closest_idx = np.argmin(time_diffs)
            if time_diffs[closest_idx] / 1e9 < 10.0:  # 10秒内有效
                msg = topic_type_data[elevator_area_topic]['data'][elevator_timestamps[closest_idx]]
                elevator_points_map = deserialize_polygon_stamped(msg)
                print(f"Frame {frame_id}: Updated elevator_points_map from /elevator_area at timestamp {elevator_timestamps[closest_idx]}")
            else:
                print(f"Frame {frame_id}: Using previous elevator_points_map (no recent /elevator_area data)")

        # 获取动态TF
        T_map_to_odom, T_odom_to_base_footprint = get_tf_transform(tf_data, timestamp)
        if T_map_to_odom is None or T_odom_to_base_footprint is None:
            print(f"Frame {frame_id}: Skipping due to missing TF transforms")
            continue

        # 计算从map到livox_frame的完整变换
        T_map_to_base_footprint = T_map_to_odom @ T_odom_to_base_footprint
        T_map_to_livox = (T_map_to_base_footprint @ T_base_footprint_to_base_link @ 
                         T_base_link_to_sensor_kit @ T_sensor_kit_to_livox @ T_livox_to_livox_frame)
        T_livox_to_map = np.linalg.inv(T_map_to_livox)
        print(f"Frame {frame_id}, T_livox_to_map:\n{T_livox_to_map}")

        # 转换多边形到livox坐标系
        elevator_points_map_homo = np.hstack([elevator_points_map, np.ones((elevator_points_map.shape[0], 1))])
        elevator_points_lidar_homo = (T_livox_to_map @ elevator_points_map_homo.T).T
        elevator_points_lidar = elevator_points_lidar_homo[:, :3]
        print(f"Frame {frame_id}, elevator_points_lidar:\n{elevator_points_lidar}")

        # 像素映射
        elevator_pixel = world_to_pixel(elevator_points_lidar, map_range_x, map_range_y, cell_size, h, w)
        print(f"Frame {frame_id}, elevator_pixel:\n{elevator_pixel}")

        # 检查像素坐标有效性
        if np.any(elevator_pixel < 0) or np.any(elevator_pixel[:, 0] >= w) or np.any(elevator_pixel[:, 1] >= h):
            print(f"Frame {frame_id}: Warning: elevator_pixel out of bounds")

        # 收集点云数据
        list_points = []
        for topic in exported_topics[:3]:
            if topic not in dict_timestamps:
                continue
            np_topic_timestamp = dict_timestamps[topic]
            idx = np.argwhere(np.abs(np_topic_timestamp - timestamp) < interval // 2)
            if len(idx) != 1:
                print(f"Frame {frame_id}: No matching timestamp for {topic}")
                continue
            ts = np_topic_timestamp[idx].item()
            msg_type, data = topic_type_data[topic]['msg_type'], topic_type_data[topic]['data'][ts]
            # if msg_type == 'sensor_msgs/msg/PointCloud2':
            #     points_data = deserialize_pointcloud2(data)
            # else:
            #     points_data = deserialize_msg(msg_type, data)
            points_data = deserialize_msg(msg_type, data)
            points = points_data['xyz']
            if topic in extrinsics:
                T2lidar = np.array(extrinsics[topic])  # 确保extrinsics是numpy数组
                points_at_lidar = np.dot(T2lidar[:3, :3], points.T).T + T2lidar[:3, 3]
                list_points.append(points_at_lidar)
            else:
                print(f"Frame {frame_id}: No extrinsic for {topic}, skipping")
        
        if len(list_points) < 2:
            print(f"Frame {frame_id}: Missing point cloud data, got {len(list_points)} topics")
            continue

        # 创建占据地图
        points = np.concatenate(list_points, axis=0)

        
        

        mask = (points[:, 0] > map_range_x[0]) & (points[:, 0] < map_range_x[1]) & \
               (points[:, 1] > map_range_y[0]) & (points[:, 1] < map_range_y[1]) & \
               (points[:, 2] > -0.1) & (points[:, 2] < 0.65)  # -0.65
        points = points[mask]
        # 可视化点云
        # import open3d as o3d
        # import numpy as np

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)

        # o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud for Frame {frame_id}")
        
        map2d = np.ones([h, w], dtype=np.uint8) * 255
        points_2d = points[:, :2]
        points_rasterized = world_to_pixel(points_2d, map_range_x, map_range_y, cell_size, h, w)
        
        valid_mask = (points_rasterized[:, 0] >= 0) & (points_rasterized[:, 0] < w) & \
                     (points_rasterized[:, 1] >= 0) & (points_rasterized[:, 1] < h)
        points_rasterized = points_rasterized[valid_mask]
        points_rasterized = np.unique(points_rasterized, axis=0)
        print(f"Frame {frame_id}, Rasterized points shape: {points_rasterized.shape}")
        
        map2d[points_rasterized[:, 1], points_rasterized[:, 0]] = 0  # x, y 顺序适配旋转
        map2d_color = cv2.cvtColor(map2d, cv2.COLOR_GRAY2BGR)
        
        # 绘制多边形和顶点
        elevator_pixel = elevator_pixel.astype(np.int32)
        if elevator_pixel.shape[0] >= 3:
            cv2.polylines(map2d_color, [elevator_pixel], isClosed=True, color=(0, 0, 255), thickness=1)
            for pt in elevator_pixel:
                cv2.circle(map2d_color, tuple(pt), 2, (255, 0, 0), -1)
        else:
            print(f"Frame {frame_id}: Invalid elevator_pixel for drawing")
        
        # 计算未占据面积
        if elevator_pixel.shape[0] >= 3:
            freespace_area = calculate_freespace_area(map2d, elevator_pixel, cell_size)
            can_enter = freespace_area >= freespace_threshold
            status_text = f"Freespace: {freespace_area:.2f} m², {'Can Enter' if can_enter else 'Cannot Enter'}"
            print(f"Frame {frame_id}: {status_text}")
        else:
            freespace_area = 0
            status_text = "Invalid polygon, Cannot Enter"
            print(f"Frame {frame_id}: {status_text}")

        # 绘制LiDAR坐标系原点
        origin_pixel = world_to_pixel(np.array([[0, 0, 0]]), map_range_x, map_range_y, cell_size, h, w)
        cv2.circle(map2d_color, tuple(origin_pixel[0].astype(np.int32)), 3, (0, 255, 0), -1)

        # 调整大小
        map2d_rsz = cv2.resize(map2d_color, (w*7, h*7), interpolation=cv2.INTER_NEAREST)

        # 添加文本
        cv2.putText(map2d_rsz, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if can_enter else (0, 0, 255), 1)

        # 相机图像
        front_image = os.path.join(path_img, 'output', f'{frame_id}.png')
        if os.path.exists(front_image):
            front = cv2.imread(front_image)
            h_, w_ = front.shape[:2]
            scale = h*7 / h_
            front = cv2.resize(front, (int(w_*scale), h*7), interpolation=cv2.INTER_NEAREST)
            map2d_rsz = np.concatenate([map2d_rsz, front], axis=1)
        else:
            print(f"Frame {frame_id}: Camera image {front_image} not found")

        # 保存结果
        path_save = os.path.join(dir_save, f'{frame_id}.png')
        cv2.imwrite(path_save, map2d_rsz)
        print(f"已处理帧 {frame_id}，结果保存到: {path_save}")

    except Exception as e:
        print(f"处理帧 {frame_id} 时出错: {str(e)}")
        continue

print(f"\n处理完成，结果保存在: {dir_save}")