#!/bin/bash
mkdir build
cd build
cmake ..
make
cd ..
rm -rf build
chmod +x noetix_joint_controller

