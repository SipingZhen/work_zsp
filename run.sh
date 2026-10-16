#!/bin/bash
sleep 1
sudo  ifconfig eth0 up
#gnome-terminal -- bash -c "./setrgmii.sh"
./setrgmii.sh
sudo -s <<EOF
echo "##################"
cd /home/noetix/work_zkt/noetix_sdk_release
./noetix_joint_controller
EOF
wait
