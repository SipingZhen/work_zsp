# noetix_sdk_n2
noetix robot N2 sdk.

### Prebuild environment
* OS  (Ubuntu 22.04 LTS)
* CPU  (aarch64 and x86_64)
* Compiler  (gcc version 11.4.0)

### Environment Setup

Before building or running the SDK, ensure the following dependencies are installed:

- CMake (version 3.3 or higher)
- GCC (version 11.4.0)
- Make

You can install the required packages on Ubuntu 22.04 with:

```bash
apt-get update
apt-get install -y cmake g++ build-essential libyaml-cpp-dev libeigen3-dev libboost-all-dev libfmt-dev pybind11-dev
```

### Build examples

To build the examples inside this repository:

```bash
./build.sh
```

### N2 depth policy

Use the same Python environment to install the depth policy dependencies and build the Python binding with GCC 11.

```bash
python -m pip install numpy onnxruntime pyrealsense2
CC=gcc-11 CXX=g++-11 cmake -S . -B build -DBUILD_CPP_EXAMPLES=OFF \
  -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE="$(command -v python)"
cmake --build build --target lowcontrol_py -j
python examples/py/test_policy.py
```

On Jetson, use the Python version supported by `pyrealsense2` for your JetPack release. [RealSense package instructions](https://pypi.org/project/pyrealsense2/)

The example loads `policy/n2_depth.onnx` and uses the network interface and peer in `config/dds.xml`. Set `CYCLONEDDS_URI` to use a different DDS configuration. The default RealSense stream is `640×480 Z16` at 90 fps. Use `--camera-fps` if your camera requires another rate.

The policy expects the physical camera to match the mounting pose in the ONNX metadata. The example samples the latest frame at each 50 Hz policy step, using the RealSense intrinsics to map it to the trained `16×24` view. Depth stays in metres through the ONNX input. [RealSense projection and depth units](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/)

The remote uses the SDK example's button indices. Button `9` enables or stops control. Press `10 + 2` to stand, then press `5 + 2` after standing to start the policy. Button `11` returns to standing. If the depth stream times out during walking, the controller returns to standing.

### Notice
For more reference information, please go to [Document Center](https://web.noetixrobotics.com/docs/).
