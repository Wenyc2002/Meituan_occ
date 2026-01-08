
# "1. 启动底盘..."
./bringup_scout.sh &


# "2. 启动雷达..."
roslaunch livox_ros_driver2 msg_MID360.launch &


# "3. 启动相机..."
roslaunch zed_wrapper zed2.launch &


# "4. 启动 FAST-LIO..."
roslaunch fast_lio mapping_mid360.launch &


#  "5. 转格式节点..."
cd ~/yanci_ws
source devel/setup.bash
rosrun livox_to_pointcloud2 livox_to_pointcloud2_node &

# run occupancy
cd ~/yanci_ws
source devel/setup.bash
roslaunch map_lab.launch
### The car needs to be remotely controlled to move first.
python3 /home/goodboy/yanci_ws/src/onboard_detector/scripts/person_inout_1009.py

# Next run tracking part
cd /meituan_ws
conda activate meituan
cd /meituan_ws/src/my_yolox_ros/launch/pedestrian_tracking22_local.launch


