# N2 ONNX 转 RKNN

在已安装 `onnx`、`PyYAML` 和 RKNN Toolkit 2 的 Python 环境中，从仓库根目录运行：

```bash
python onnx2rknn/onnx2rknn.py --validate-only
python onnx2rknn/onnx2rknn.py
```

默认读取 `noetix_sdk_release/policy/policy.onnx`，输出到同目录的 `policy.rknn`。可用 `--onnx` 和 `--output` 指定其他路径。验证步骤会检查 N2 策略的 310 维输入、18 维输出以及与 `config/ning_user.yaml` 一致的关节顺序、默认角度、PD 增益和动作缩放。只有转换成功后才部署生成的 RKNN 文件。
