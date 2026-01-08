#!/bin/bash
set -e                 # 遇错即停，方便排错

echo "1. 启动底盘..."
./bringup_scout.sh &

sleep 4
echo "2. 启动雷达..."
roslaunch livox_ros_driver2 msg_MID360.launch &

sleep 8
echo "3. 启动相机..."
roslaunch zed_wrapper zed2.launch &

sleep 10
echo "4. 启动 FAST-LIO..."
roslaunch fast_lio mapping_mid360.launch &

sleep 10
echo "5. 转格式节点..."
cd ~/yanci_ws
source devel/setup.bash
rosrun livox_to_pointcloud2 livox_to_pointcloud2_node &


