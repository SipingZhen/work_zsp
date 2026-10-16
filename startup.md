gnome-terminal -- bash -c " /home/oem/work/startup.sh;exec bash"
wget http://fishros.com/install -O fishros && . fishros

sudp cp sudoers /etc/sudoers
cd backport-iwlwifi
cd iwlwifi-stack-dev    //M9
sudo make defconfig-iwlwifi-public
sudo make 
sudo make install


sudo cp linux-firmware-xxx/iwlwifi-*.ucode /lib/firmware
sudo apt install linux-firmware


sudo cp usbtty.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && udevadm trigger

