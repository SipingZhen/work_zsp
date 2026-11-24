# Jetson Orin Nano Super 深度步态 GPU 部署

N2 的 `test_policy.py` 默认在 Jetson 上用 ONNX Runtime 双线程 CPU 推理 `policy/n2_depth.onnx`。可以保留模型，在 ONNX Runtime 中启用 CUDA 或 TensorRT。策略目标频率为 50 Hz，关节控制目标频率为 500 Hz。当前推理在关节控制循环里同步执行，所以选后端时重点看推理和下发的尾部延迟。 [推理实现](../examples/py/depth_policy.py) · [控制循环](../examples/py/user_controller.py) · [实机部署](deployment.md)

## 1. 确认 JetPack 环境

在 Jetson 上，从仓库根目录执行以下命令。激活用于实机部署的 Python 环境，让 `lowcontrol_py`、`pyrealsense2` 和 ONNX Runtime 使用同一个解释器。 [项目构建步骤](../README.md#n2-depth-policy)

```bash
uname -m
python --version
dpkg-query -W nvidia-l4t-core 'nvidia-jetpack*' 'cuda-cudart-*' 'libcudnn*' 'libnvinfer*'
```

Jetson 应显示 `aarch64`。按实际安装的库选择版本，例如 JetPack 6.2 对应 CUDA 12.6、cuDNN 9.3 和 TensorRT 10.3。选 ONNX Runtime wheel 时同时核对 Jetson 架构、Python ABI、CUDA 和 cuDNN，启用 TensorRT 后端还要核对 TensorRT 版本。 [JetPack 6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62) · [ONNX Runtime CUDA 兼容表](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) · [ONNX Runtime TensorRT 兼容表](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements)

## 2. 用 TensorRT 单测模型

如果已安装 TensorRT 10.x 的 `trtexec`，可直接检查模型能否构建并运行。如果缺少 `trtexec`，在已配置 JetPack 软件源的板上，用 `apt-cache policy libnvinfer-bin` 确认候选版本与板上的 TensorRT 一致，再运行 `sudo apt install libnvinfer-bin`。随后在仓库根目录执行下方命令。 [NVIDIA Jetson 软件包](https://repo.download.nvidia.com/jetson/) · [TensorRT 测速说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/benchmarking.html)

```bash
TRTEXEC=$(command -v trtexec || printf '%s\n' /usr/src/tensorrt/bin/trtexec)
"$TRTEXEC" --onnx=policy/n2_depth.onnx --fp16 --warmUp=1000 --duration=30
```

查看 `Latency` 的 P99 和最大值，同时看 `Enqueue Time` 和 `GPU Compute Time`，判断时间花在主机提交还是 GPU 计算。该命令只运行模型，可以在机器人未连接时执行。实机程序的 Python 预处理、控制循环和 DDS 下发时间需要另测。 [TensorRT 性能指标](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/benchmarking.html)

## 3. 在部署程序中选择 GPU 后端

1. 在部署用的 Python 环境安装与板上版本匹配的 Jetson GPU 版 ONNX Runtime。预编译 wheel 从官方安装页列出的 Jetson 渠道选取，找不到匹配的 wheel 时按官方 Jetson 构建说明编译。CPU 和 GPU 版都提供 `onnxruntime` 模块，因此同一环境只安装其中一个。有 CPU 版时，先用 `python -m pip uninstall onnxruntime` 卸载，再用 `python -m pip install /实际路径/匹配的GPU版.whl` 安装。GPU 版也带 CPU 后端，可用同一个环境做对照。 [ONNX Runtime 安装](https://onnxruntime.ai/docs/install/) · [Jetson 源码构建](https://onnxruntime.ai/docs/build/eps.html#nvidia-jetson-tx1tx2nanoxavierorinthor)
2. 检查 Python 实际导入的版本和可用后端。CUDA 路径需看到 `CUDAExecutionProvider`，TensorRT 路径还需看到 `TensorrtExecutionProvider`。

   ```bash
   python -c 'import onnxruntime as ort; print(ort.__version__, ort.__file__); print(ort.get_available_providers())'
   ```

3. 在 `examples/py/depth_policy.py` 顶部加上 `import os`，将 `DepthPolicy.__init__()` 中创建 `SessionOptions` 和 `InferenceSession` 的部分替换为下方代码。后面的元数据读取和输入输出处理照常执行。默认 `cpu`，首轮 CUDA 对照关闭 TF32，避免同时引入 TF32 的精度变化。`use_tf32` 选项要求 ONNX Runtime 1.18 或更新版本。TensorRT 模式启用 FP16，并允许未在 TensorRT 执行的算子走 CUDA 或 CPU。 [CUDA 后端选项](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32) · [TensorRT 后端选项](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)

   ```python
   options = ort.SessionOptions()
   options.intra_op_num_threads = 2
   cpu = "CPUExecutionProvider"
   cuda = ("CUDAExecutionProvider", {"use_tf32": 0})
   tensorrt = ("TensorrtExecutionProvider", {"trt_fp16_enable": True})

   backend = os.environ.get("N2_ORT_BACKEND", "cpu")
   backends = {"cpu": [cpu], "cuda": [cuda, cpu], "tensorrt": [tensorrt, cuda, cpu]}
   if backend not in backends:
       raise ValueError(f"Unknown N2_ORT_BACKEND: {backend}")

   providers = backends[backend]
   required = [p if isinstance(p, str) else p[0] for p in providers]
   available = ort.get_available_providers()
   missing = [p for p in required if p not in available]
   if missing:
       raise RuntimeError(f"Unavailable ONNX Runtime providers: {missing}")

   self.session = ort.InferenceSession(str(path), options, providers=providers)
   active = self.session.get_providers()
   if active != required:
       raise RuntimeError(f"Requested providers {required}, got {active}")
   self.session.disable_fallback()
   ```

   `get_providers()` 确认会话加载了所选后端。需要确认哪些算子实际在 GPU 上运行时，可临时设置 `options.enable_profiling = True`，推理后调用 `self.session.end_profiling()`，检查生成的 JSON 中算子事件的 `provider`。`disable_fallback()` 关闭运行失败后自动切换执行后端的重试，建图时分配给 CPU 的算子仍可执行。 [后端分配规则](https://onnxruntime.ai/docs/execution-providers/) · [ONNX Runtime 分析工具](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html) · [Python 会话 API](https://onnxruntime.ai/docs/api/python/api_summary.html)

## 4. 对比延迟和输出后再上机

应用上面的代码后，在仓库根目录执行下方片段，用同样的输入跑 CPU 和 CUDA。要测 TensorRT，在 `python - cpu cuda` 后再加 `tensorrt`。脚本只调用策略推理，统计 50 Hz 下 `DepthPolicy.step()` 的耗时，并报告相对 CPU 的关节目标偏差。 [策略输入和输出](../examples/py/depth_policy.py)

```bash
PYTHONPATH=examples/py python - cpu cuda <<'PY'
import os
import sys
import time

import numpy as np

from depth_policy import DepthPolicy

reference = None
for backend in sys.argv[1:]:
    os.environ["N2_ORT_BACKEND"] = backend
    policy = DepthPolicy("policy/n2_depth.onnx")
    args = (
        policy.stand_pos.copy(), np.zeros(18, dtype=np.float32),
        np.array([0, 0, 0, 1], dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.array([0.25, 0, 0], dtype=np.float32), np.ones((16, 24), dtype=np.float32),
    )
    for _ in range(50):
        policy.step(*args)
    policy.reset()

    times = []
    targets = []
    deadline = time.perf_counter()
    for _ in range(500):
        start = time.perf_counter()
        targets.append(policy.step(*args))
        times.append((time.perf_counter() - start) * 1000)
        deadline += policy.config["control_dt"]
        time.sleep(max(0.0, deadline - time.perf_counter()))

    p50, p99, maximum = np.percentile(times, [50, 99, 100])
    print(f"{backend}: p50={p50:.3f} ms, p99={p99:.3f} ms, max={maximum:.3f} ms")
    if backend == "cpu":
        reference = np.asarray(targets)
    elif reference is not None:
        error = np.max(np.abs(np.asarray(targets) - reference))
        print(f"{backend}: max joint target error vs CPU = {error:.6f} rad")
PY
```

合成输入用于初查。上实机前，用代表性的关节、IMU 和深度序列回放，对比 `actions`、`velocity` 和最终关节目标，尤其要检查 FP16。通过对照后，在吊架上用选定的后端启动实机入口，同开 RealSense 和 DDS，记录相邻 `set_joint()` 调用的 P99、最大间隔和超过 2 ms 的比例，确认状态和深度没有过期。切回 CPU 时使用 `N2_ORT_BACKEND=cpu`。 [实机入口](../examples/py/test_policy.py) · [关节控制与超时处理](../examples/py/user_controller.py) · [深度帧过期判断](../examples/py/depth_camera.py)

```bash
N2_ORT_BACKEND=cuda python examples/py/test_policy.py
```

## GPU 加速的代价

| 项目 | 对当前程序的影响 |
| --- | --- |
| 延迟 | 这份模型是 batch 1 的小模型。GPU 提交、输入输出搬运和部分算子回到 CPU 都可能抵消计算收益。2 ms 是关节控制目标周期，20 ms 是策略目标周期，200 ms 是状态或深度数据的过期阈值。GPU 推理均值下降后，仍要看控制循环的尾部延迟。 [ONNX Runtime 性能排查](https://onnxruntime.ai/docs/performance/tune-performance/troubleshooting.html) · [实机控制循环](../examples/py/test_policy.py) |
| 功耗与内存 | CUDA 上下文和 TensorRT 工作区会占用 Jetson 的共享内存。GPU 负载会改变整板功耗和温度，持续运行时可能受功耗或散热限制。比较各后端时保持同一功耗模式、相机设置和散热条件，用 `tegrastats` 观察 RAM、`GR3D_FREQ`、温度和电源轨读数。 [Jetson 资源监控](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html) |
| 数值精度 | CUDA 与 CPU 的算子实现可能产生浮点差异。TensorRT FP16 混合精度还可能放大误差或产生非有限值。策略把上一次动作和速度估计送回模型，所以要比较连续序列的输出。 [TensorRT 精度说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/accuracy-considerations.html) · [策略历史输入](../examples/py/depth_policy.py) |
| 部署与启动 | GPU 版运行库需要与 JetPack 匹配。TensorRT 首次构建引擎会增加启动时间，可配置 `trt_engine_cache_enable` 和 `trt_engine_cache_path` 重用引擎。更换模型、TensorRT 版本或目标硬件后，使用与新环境对应的缓存目录重新构建。 [TensorRT 引擎缓存](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#trt_engine_cache_enable) |
