# N2 ARM 部署项目

本项目包含在基于ARM架构的设备上使用瑞芯微RKNN框架运行神经网络模型的部署脚本和工具，主要用于Noetix机器人控制系统。

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
- `policy_1.onnx`: 原始ONNX格式模型
- `policy_1.rknn`: 转换后的RKNN格式模型（用于NPU加速）

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
1. 将新的ONNX模型放入 `onnx2rknn/` 目录
2. 转换为RKNN格式
3. 替换 `policy/` 目录下的模型文件

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
