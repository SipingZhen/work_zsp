# N2 ARM 部署项目

本项目包含在基于ARM架构的设备上使用瑞芯微RKNN框架运行神经网络模型的部署脚本和工具，主要用于Noetix机器人控制系统。

## 与当前 N2 训练策略对齐

`noetix_sdk_release/policy/policy.onnx` 是唯一的 ONNX 策略文件。策略输入是 5 帧 × 62 维（310 维），输出是 18 维动作，控制频率为 50 Hz。部署观测依次为速度指令 3、机体角速度 3、roll/pitch 2、相对默认关节角 18、关节速度 18、上一帧动作 18；关节按 `config/ning_user.yaml` 中的 `policy_joint_order` 排列，动作按每个关节的 `action_scale` 执行。首次进入行走模式时，5 帧历史均填入当前观测。

更新策略时，将训练项目 `logs/rsl_rl/n2_walkrun/<run>/exported/policy.onnx` 复制到 `noetix_sdk_release/policy/policy.onnx`，再运行：

```bash
python onnx2rknn/onnx2rknn.py --validate-only
python onnx2rknn/onnx2rknn.py
```

第二条命令需要 RKNN Toolkit 2，生成 `noetix_sdk_release/policy/policy.rknn`。ARM 控制器使用该 RKNN 文件，必须先完成转换再上机。转换脚本会检查新 ONNX 的输入输出形状、关节顺序、默认角度、PD 增益和动作缩放是否与部署配置相符。修改 C++ 后也需要重新编译控制器。首次上机前应先确认 IMU 角速度坐标系和电机编号与当前硬件一致。

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
