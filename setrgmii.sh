#!/bin/bash
sleep 1
sudo su -c "echo 0x10 0x0f > /sys/devices/platform/fe1c0000.ethernet/rgmii_delayline"
exit
