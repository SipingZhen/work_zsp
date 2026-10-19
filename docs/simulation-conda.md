# N2 MuJoCo 仿真：Conda 安装与运行

从仓库根目录运行 `noetix_sdk_release/src/sim2sim.py`。仿真读取仓库内的 N2 MuJoCo 模型、`policy.onnx`、`policy_walk.onnx` 和 `ning_user.yaml`，用 ONNX Runtime 在 CPU 上推理；无需 RKNN、机器人本体或动作 NPZ。以下以 Linux 为例。

## 1. 准备系统工具

```bash
cd /path/to/work_zsp
test -f noetix_sdk_release/robot/mjcf/N2.xml
test -f noetix_sdk_release/policy/policy.onnx
test -f noetix_sdk_release/config/ning_user.yaml
g++ --version
```

启动时脚本会调用系统 `g++` 临时编译 `usercontroller.cpp` 中的观测函数。Ubuntu/Debian 缺少它时执行：

```bash
sudo apt update
sudo apt install -y g++
```

若尚未安装 Conda，按 [Miniforge 官方说明](https://github.com/conda-forge/miniforge/blob/main/README.md#unix-like-platforms-macos-linux--wsl)安装对应架构的 Linux 版本（x86_64 或 aarch64）。安装后重新打开终端，用 `conda --version` 确认。已有 Miniconda 或 Anaconda 时可直接使用。

## 2. 建立 Conda 环境

在仓库根目录执行：

```bash
conda create -n n2_deploy -c conda-forge python=3.11 pip -y
conda activate n2_deploy
python -m pip install -r noetix_sdk_release/requirements-sim.txt
```

依赖清单包含 `mujoco`、`numpy`、`onnxruntime` 和 `PyYAML`。确认当前 Python 和依赖：

```bash
python -c "import sys, mujoco, numpy, onnxruntime, yaml; print(sys.executable); print('mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'onnxruntime', onnxruntime.__version__, 'PyYAML', yaml.__version__)"
```

以后每次打开新终端，先进入仓库并执行 `conda activate n2_deploy`。若提示无法激活环境，按 [Conda 环境管理说明](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)初始化当前 shell，再打开新终端。

## 3. 运行仿真

先做无界面检查，不需要显示器：

```bash
python noetix_sdk_release/src/sim2sim.py --no-joystick --headless --duration 3 --vx 0.5
```

成功结束时会打印仿真时间、墙钟时间和速度倍数。无界面且未连接遥控器时会尽快运行，因此可能快于 1 倍速。在有桌面显示的终端查看机器人运动：

```bash
python noetix_sdk_release/src/sim2sim.py --no-joystick --duration 10 --vx 0.5
```

关闭 MuJoCo 窗口可提前结束。

图形模式按累计仿真时间与墙钟同步，并以约 60 Hz 刷新画面；末尾的速度倍数接近 `1.00x` 表示接近原速。如果机器算力不足，仍可能低于 1 倍速。

要用与实体控制器相同的遥控器按键流程，连接 Linux 摇杆设备后运行：

```bash
ls -l /dev/input/js0
python noetix_sdk_release/src/sim2sim.py --joystick
```

默认进入手柄模式；如果设备还未连接，程序会提示一次并等待。设备路径来自 `config/hwconfig.yaml` 的 `remote_port`，可用 `--joystick-device /dev/input/js1` 覆盖；用 `--no-joystick` 可直接运行策略。实体控制器和仿真都调用 `usercontroller.cpp` 中的 `updateRemoteMode()` 处理按键，并调用 `readRemoteCommand()` 把轴 1、轴 0 分别映射为前进和转向指令（横向指令为 0）。仿真默认使用你提供的手柄编号：物理按键 `7` 使能，按住 `4` 再按 `6` 站立，按住 `4` 再按 `2` 选择原策略，按住 `4` 再按 `3` 选择 walk，按住 `4` 再按 `0` 选择 run。它们分别映射到逻辑按键 `9`、`2+10`、`2+5`、`2+6`、`2+8`；逻辑按键 `11` 可从 USERWALK 返回 STAND。实机的 run 组合键为 `2+8`。仿真在 DEFAULT 模式持续运行物理仿真，关节为零力矩，机器人会自然倒下。遥控模式会使用摇杆指令，忽略 `--vx`、`--vy` 和 `--yaw`；默认持续运行，关闭窗口或按 `Ctrl+C` 结束，也可用 `--duration 10` 指定仿真时长。

如果手柄按键或轴编号不同，先运行 `python noetix_sdk_release/src/sim2sim.py --joystick --joystick-debug`，按动手柄查看终端打印的物理编号，再用 `--joystick-button-map 9=9` 这类参数覆盖默认映射；可重复指定。默认前进使用左摇杆竖直轴 1 的反方向，转向使用右摇杆水平轴 3 的反方向；用 `--joystick-forward-axis`、`--joystick-yaw-axis` 和相应的 `--joystick-forward-sign`、`--joystick-yaw-sign` 调整。窗口获得焦点时按住 `Ctrl+Shift+R`，仿真会把机器人扶正、提起并固定在空中；保持按键时可以依次用遥控器使能、站立、进入策略，松开任意一个键后机器人下落。此快捷键只在 MuJoCo 中有效。实体控制器仍通过 SDK 的 `get_jsdata()` 获取手柄数据。

默认文件如下：

| 文件 | 路径 |
| --- | --- |
| MuJoCo 模型 | `noetix_sdk_release/robot/mjcf/N2.xml` |
| ONNX 策略 | `noetix_sdk_release/policy/policy.onnx` |
| Walk ONNX 策略 | `noetix_sdk_release/policy/policy_walk.onnx` |
| Run ONNX 策略 | `noetix_sdk_release/policy/policy_run.onnx` |
| 控制配置 | 默认策略用 `ning_user.yaml` 的 `user`；walk 和 run 用 `ning_ac.yaml` 的对应配置 |

仿真步长是 0.002 秒；三种策略每 10 个物理步更新一次，即 50 Hz。默认策略使用 5×62 维观测和 18 个关节动作；walk 使用 5×39 维观测和 10 个腿关节动作；run 使用 5×63 维观测和 18 个关节动作。walk 与 run 的观测均调用部署端 C++ 函数。

## 4. 常用参数

运行 `python noetix_sdk_release/src/sim2sim.py --help` 可查看全部选项。

| 参数 | 用途 | 默认值 |
| --- | --- | --- |
| `--headless` | 不打开窗口 | 关闭 |
| `--duration` | 仿真时长，单位秒 | 普通模式 `10`；遥控模式不限时 |
| `--joystick`、`--no-joystick` | 启用或关闭 Linux 遥控器模式 | 默认启用；设备未连接时等待 |
| `--joystick-device` | 指定摇杆设备 | `hwconfig.yaml` 中的 `remote_port` |
| `--joystick-debug` | 打印物理轴、按键和映射后的 `x/yaw` 指令 | 关闭 |
| `--joystick-button-map LOGICAL=PHYSICAL` | 覆盖按键 2、5、6、8、9、10、11 的物理编号 | `2→4`、`5→2`、`6→3`、`8→0`、`9→7`、`10→6`、`11→11` |
| `--joystick-forward-axis`、`--joystick-yaw-axis` | 前进、转向使用的物理轴编号 | `1`、`3` |
| `--joystick-forward-sign`、`--joystick-yaw-sign` | 对轴值乘以 `-1` 或 `1` | `-1`、`-1` |
| `--vx` | 前进输入；默认、walk、run 的速度指令下限均为 -0.6，walk 上限 1，run 上限 2 | `0` |
| `--vy` | 横向速度指令，单位 m/s | `0` |
| `--yaw` | 转向角速度指令，单位 rad/s | `0` |
| `--base-height` | 初始机身高度，单位 m | `0.7` |
| `--config`、`--profile` | 配置文件及其顶层配置段 | `ning_user.yaml`、`user` |
| `--xml`、`--policy`、`--walk-policy` | 指定其他 MJCF 和 ONNX 文件 | 上表路径 |
| `--aux-input NAME=FILE.npy` | 给 ONNX 的额外输入提供数据，可重复指定 | 无 |

例如：

```bash
python noetix_sdk_release/src/sim2sim.py --no-joystick --headless --duration 5 --vx 0.5 --yaw 0.2
```

当前配置中 `axis_mappings.scaley` 是 `0.0`，因此默认配置下修改 `--vy` 不会改变送入策略的横向速度指令。前进指令还会经过观测函数的死区和逐帧限幅，第一帧不会立即达到 `--vx` 的值。
默认、walk、run 的前进指令下限均为 -0.6；walk 的上限为 1，run 的上限为 2。walk 和 run 的 x 输入均限制在手柄轴范围 `[-1, 1]`，因此 `--vx` 超过 1 也不会继续增大前进指令。

## 5. 更换策略或配置

可用 `--policy`、`--walk-policy`、`--run-policy` 指定各自的 ONNX 文件；需要不同参数时，再用 `--config /path/to/config.yaml --profile user` 指定默认策略配置。观测输入形状须与配置中的 `observations_size × stack_size` 对应，动作输出维度须与 `actions_size` 一致。关节顺序也须与策略元数据及 N2 模型一致。修改默认、walk 或 run 的 C++ 观测函数后，下次启动会重新编译相应函数。

多输入模型可以重复使用 `--aux-input 输入名=文件.npy` 提供额外输入，也可以在配置段的 `sim_inputs` 中将输入名映射到文件路径（路径相对于配置文件所在目录）。文件可以是单个张量，也可以是在最前面增加策略步维度的 `.npy` 帧序列；脚本还接受 `.npz` 文件。若观测输入或动作输出不是模型的第一个输入/输出，分别设置 `sim_observation_input` 和 `sim_action_output`。额外数据仍须由使用者按策略的实际输入定义准备；本仓库没有为视觉等输入提供通用数据源。

仿真使用 MuJoCo 传感器代替实体传感器，并通过 CPU 上的 ONNX Runtime 推理；它不验证 RKNN 转换结果、NPU 推理或实体硬件接口。

## 6. 常见问题

| 现象 | 检查方法 |
| --- | --- |
| `ModuleNotFoundError` | 执行 `conda activate n2-sim`，用 `python -c "import sys; print(sys.executable)"` 确认解释器，再安装依赖。 |
| `g++: No such file or directory` | 安装系统 `g++`；它不是 Python 包。 |
| 找不到 STL、XML 或 ONNX | 检查文件是否齐全；MJCF 使用相对网格路径，须保留 `robot/mjcf` 与 `robot/meshes` 的目录关系。 |
| `GLFW`、`DISPLAY` 或 OpenGL 报错 | 无桌面会话时加 `--headless`。需要窗口时检查桌面显示和图形驱动。 |
| 找不到 `/dev/input/js0` | 检查遥控器连接和实际设备路径；必要时指定 `--joystick-device`。当前用户还需有设备读取权限。 |
| ONNX 形状或 `joint_names` 不匹配 | 核对 `ning_user.yaml` 的 `size`、`policy_joint_order` 与 ONNX 元数据；不能只替换策略文件。 |

相关官方文档：[Conda 环境](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)、[MuJoCo Python](https://mujoco.readthedocs.io/en/latest/python.html)、[ONNX Runtime Python](https://onnxruntime.ai/docs/get-started/with-python)。
