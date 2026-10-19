# N2 ARM 部署项目

本项目包含在基于ARM架构的设备上使用瑞芯微RKNN框架运行神经网络模型的部署脚本和工具，主要用于Noetix机器人控制系统。

## 与当前 N2 训练策略对齐

`noetix_sdk_release/policy/policy.onnx` 是 4+2 选择的默认策略，输入是 5 帧 × 62 维（310 维），输出是 18 维动作，控制频率为 50 Hz。部署观测依次为速度指令 3、机体角速度 3、roll/pitch 2、相对默认关节角 18、关节速度 18、上一帧动作 18；关节按 `config/ning_user.yaml` 中的 `policy_joint_order` 排列，动作按每个关节的 `action_scale` 执行。`policy_walk.onnx` 由 4+3 选择，输入为 5 帧 × 39 维，使用机体系投影重力和左右腿各 5 个关节，输出 10 个腿关节动作。`policy_run.onnx` 在仿真中由 4+0 选择，实机暂用逻辑键 2+7，输入为 5 帧 × 63 维（315 维），使用机体系投影重力和全部 18 个关节，输出 18 个动作。walk 和 run 的观测缩放、默认角度、PD 参数及动作缩放分别来自 `config/ning_ac.yaml` 的对应配置，在 ARM 控制器上通过 ONNX Runtime 运行。切换策略时，5 帧历史均填入当前观测。

run 模型没有关节顺序元数据；当前实现按来源分支 SDK 的电机顺序排列 run 的关节观测和动作：左臂 4、左腿 5、右臂 4、右腿 5。仿真会在此顺序与 `ning_user.yaml` 的关节顺序之间转换。

更新策略时，将训练项目 `logs/rsl_rl/n2_walkrun/<run>/exported/policy.onnx` 复制到 `noetix_sdk_release/policy/policy.onnx`，再运行：

```bash
python onnx2rknn/onnx2rknn.py --validate-only
python onnx2rknn/onnx2rknn.py
```

第二条命令需要 RKNN Toolkit 2，生成 `noetix_sdk_release/policy/policy.rknn`。ARM 控制器使用该 RKNN 文件，必须先完成转换再上机。转换脚本会检查新 ONNX 的输入输出形状、关节顺序、默认角度、PD 增益和动作缩放是否与部署配置相符。修改 C++ 后也需要重新编译控制器。首次上机前应先确认 IMU 角速度坐标系和电机编号与当前硬件一致。

## N2 MuJoCo 仿真

完整的 Conda 环境安装、运行参数和排错步骤见 [N2 仿真安装与运行文档](docs/simulation-conda.md)。

`noetix_sdk_release/src/sim2sim.py` 使用 N2 的 MJCF 和 ONNX 策略。每次启动 sim 都会用 `g++` 从 `usercontroller.cpp` 临时编译默认、walk 和 run 的观测函数，并直接读取它们生成的观测缓冲区；修改这些函数后，无需同步 Python 观测代码。仿真适配类型放在 `include/sim_usercontroller_adapter.h`。观测维度、历史帧数和动作维度从配置及 ONNX 形状读取；`--config` 和 `--profile` 可选择其他策略配置。不需要动作 NPZ 文件或 RKNN。

```bash
python -m pip install -r noetix_sdk_release/requirements-sim.txt
python noetix_sdk_release/src/sim2sim.py --no-joystick --vx 0.5
# 没有图形桌面时
python noetix_sdk_release/src/sim2sim.py --no-joystick --headless --duration 3 --vx 0.5
```

仿真默认进入手柄模式，设备未连接时等待连接；使用 `--no-joystick` 可直接运行策略。仿真步长为 0.002 秒，策略更新间隔从配置的 `control.decimation` 读取；默认 N2 配置每 10 步更新一次（50 Hz）。无手柄模式下，`--vx`、`--vy`、`--yaw` 分别设置前进、横移和转向指令。MuJoCo 传感器代替实体 IMU 和关节反馈，ONNX Runtime 代替 RKNN 推理；硬件接口和这两部分的差异仍需单独核对。

更换策略时需让配置中的 `size` 与 ONNX 输入形状一致，并在 `computeObservation()` 中生成对应观测；sim 会直接取该函数写入的缓冲区。额外的 ONNX 输入可以通过 `--aux-input 输入名=数据.npy` 提供，支持单个张量或按策略步排列的帧序列，也可以在配置项 `sim_inputs` 中写入输入名到文件路径的映射。多输入模型可以用 `sim_observation_input` 指定观测输入名，用 `sim_action_output` 指定动作输出名。当前仓库没有 GMT、AMP、感知策略的模型和数据定义，视觉帧等数据仍需由使用者提供。

## 项目结构

- `noetix_sdk_release/`: Noetix机器人SDK发布版本（核心部分）
- `onnx2rknn/`: ONNX模型转RKNN格式工具
- `rknpu2v2.1.0/`: 瑞芯微NPU SDK及示例

## Noetix机器人SDK详解

### 目录结构
```
noetix_sdk_release/
├── noetix_joint_controller    # 关节控制器主程序
├── build_release.sh          # 编译脚本
├── CMakeLists.txt            # CMake配置文件
├── config/                   # 配置文件目录
├── include/                  # 头文件目录
├── lib/                      # 库文件目录
├── policy/                   # 策略模型目录
├── robot/                    # 机器人配置目录
└── src/                      # 源代码目录
```

### 核心组件

#### 1. 关节控制器 (`noetix_joint_controller`)
主控制程序，负责：
- 机器人关节运动控制
- 策略模型推理
- 传感器数据处理
- 通信管理

#### 2. 策略模型 (`policy/`)
- `policy.onnx`: 当前 N2 训练策略
- `policy.rknn`: 从该 ONNX 转换得到的 RKNN 模型（用于 NPU 加速）

#### 3. 机器人配置 (`robot/`)
机器人硬件配置文件：
- 运动学参数
- 关节限制
- 传感器校准参数

#### 4. 源代码 (`src/`)
核心算法实现：
- 运动控制算法
- 传感器融合
- 通信协议实现

### 快速部署

#### 1. 系统初始化
```bash
# 复制配置文件
sudo cp sudoers /etc/sudoers
sudo cp usbtty.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# 网络配置
sudo ifconfig eth0 up
./setrgmii.sh
```

#### 2. 启动机器人控制器
```bash
# 方法1：使用运行脚本
./run.sh

# 方法2：手动启动
cd noetix_sdk_release
./noetix_joint_controller
```

#### 3. 编译（如需修改源码）
```bash
cd noetix_sdk_release
chmod +x build_release.sh
./build_release.sh
```

### 配置说明

#### 修改机器人参数
编辑 `robot/` 目录下的配置文件：
- 关节角度限制
- 运动速度参数
- 传感器校准值

#### 更新策略模型
1. 将新的 ONNX 模型保存为 `noetix_sdk_release/policy/policy.onnx`
2. 运行 `python onnx2rknn/onnx2rknn.py --validate-only` 检查模型与部署配置
3. 运行 `python onnx2rknn/onnx2rknn.py` 生成 `policy.rknn`

#### 调试和监控
```bash
# 查看控制器状态
ps aux | grep noetix

# 查看系统日志
journalctl -f

# USB调试设备连接后
ls -la /dev/remote
```

## 其他工具简要说明

### RKNN模型转换 (`onnx2rknn/`)
```bash
cd onnx2rknn
python3 onnx2rknn.py  # 转换ONNX模型到RKNN格式
```

### RKNN SDK示例 (`rknpu2v2.1.0/`)
包含各种AI模型示例：
- YOLOv5目标检测
- MobileNet图像分类
- 基础API使用示例

### 系统管理脚本
- `run.sh`: 主运行脚本
- `closedisplay.sh`: 关闭图形界面
- `opendisplay.sh`: 开启图形界面
- `setrgmii.sh`: 网络优化配置

## 环境要求

- 基于Rockchip RK3588S的ARM设备
- Linux系统（推荐Ubuntu 20.04）
- Python 3.6+
- Rockchip NPU运行时环境

## 故障排除

### 常见问题
1. **控制器启动失败**
   ```bash
   # 检查权限
   sudo -l
   # 检查依赖库
   ldd noetix_sdk_release/noetix_joint_controller
   ```

2. **模型推理失败**
   ```bash
   # 检查RKNN环境
   echo $LD_LIBRARY_PATH
   # 检查模型文件
   ls -la noetix_sdk_release/policy/
   ```

3. **网络连接问题**
   ```bash
   ifconfig eth0
   ./setrgmii.sh
   ```

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 致谢

- 感谢瑞芯微提供的RKNN工具链和NPU SDK
- 开源社区提供的各种工具和库
# work_zsp
