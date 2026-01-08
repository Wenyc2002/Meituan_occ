sudo ip link set can0 up type can bitrate 500000
source /home/goodboy/scout_ws/devel/setup.bash
roslaunch scout_bringup scout_mini_robot_base.launch
