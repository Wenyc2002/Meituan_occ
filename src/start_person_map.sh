#!/bin/bash
set -e  # 遇错即停，方便排错

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "/opt/ros/noetic/setup.bash"
source "$WORKSPACE_ROOT/devel/setup.bash"


echo "1. 启动 map_lab.launch..."
roslaunch onboard_detector map_lab.launch &

sleep 5
echo "2. 启动 person_inout_1009.py..."
python3 "$WORKSPACE_ROOT/src/onboard_detector/scripts/person_inout_1009.py" &

echo "两个节点已启动。"
wait
