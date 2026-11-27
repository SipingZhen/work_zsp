# N2 深度步态部署与 sim2sim

以下命令均在仓库根目录执行。

## Jetson 实机部署

1. 将 RealSense 接到 Jetson。默认 DDS 配置要求 Jetson 的机器人网口使用 `192.168.55.101`，N2 地址为 `192.168.55.102`。地址不同就修改 `config/dds.xml`。
2. 在 Ubuntu 22.04 上安装编译依赖。激活与 JetPack 匹配的 Python 环境，用同一个解释器安装依赖、编译和启动。[RealSense Jetson 安装指南](https://github.com/realsenseai/librealsense/blob/master/doc/installation_jetson.md)

   ```bash
   sudo apt update
   sudo apt install -y cmake make gcc-11 g++-11 libeigen3-dev libboost-all-dev pybind11-dev python3-dev
   python -m pip install numpy onnxruntime pyrealsense2
   CC=gcc-11 CXX=g++-11 cmake -S . -B build -DBUILD_CPP_EXAMPLES=OFF \
     -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE="$(command -v python)"
   cmake --build build --target lowcontrol_py -j
   python examples/py/test_policy.py
   ```

3. 遥控器按 SDK 逻辑键 `9` 进入 `LIE`，按 `2 + 10` 站立，站稳后按 `2 + 5` 进入 `USERWALK`。步态中按 `11` 回到站立，按 `9` 停止控制，终端按 `Ctrl+C` 退出。

## MuJoCo sim2sim

在 Linux 上连接手柄后运行。以下命令使用 `n2_depth.onnx`，并将 ZhiXu/Xbox 布局的 **B** 键映射为返回站立。

```bash
python -m pip install -r requirements-sim.txt
python examples/py/sim2sim.py --joystick --joystick-device /dev/input/js0 \
  --joystick-button-map 11=1 --terrain flat --no-vis-attention
```

按 **Start** 启动，按 **LB + Back/Select** 站立，站稳后按 **LB + X** 进入步态。左摇杆上下控制前进速度，右摇杆左右控制转向。按 **B** 回到站立，按 **Start** 停止控制，窗口内按 **R** 重置。可用 `--joystick-debug` 确认手柄按键编号，或者用 `--terrain stairs --step-height 0.10` 切换到台阶场景。

无窗口、无手柄时可运行 6 秒平地检查，终端应依次显示 `LIE`、`STAND`、`USERWALK`。

```bash
python examples/py/sim2sim.py --headless --no-joystick --terrain flat --duration 6 --vx 0.25
```

两个入口均支持 `--inference-frames`。默认 `1` 使用同步推理。设为 `2`、`3`、`4`、`5` 后，推理在独立进程执行，采样后分别约 4、6、8、10 ms 开始接收结果。关节目标仍以 2 ms 为周期下发，等待期间沿用上一目标。在下一次策略采样时仍无结果，控制器返回站立。
