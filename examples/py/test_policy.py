import argparse
import os
from pathlib import Path
import sys
import time

from depth_policy import DepthPolicy
from realsense_depth import RealSenseDepth
from user_controller import MOTOR_PERIOD, UserController


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Run the N2 depth policy through the low-level SDK.")
    parser.add_argument("--camera-fps", type=int, default=90, help="RealSense depth stream frame rate")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "build"))
    from lowcontrol_py import LowController, MotorCmd

    os.environ.setdefault("CYCLONEDDS_URI", (ROOT / "config/dds.xml").as_uri())
    policy = DepthPolicy(ROOT / "policy/n2_depth.onnx")
    with RealSenseDepth(policy.config["camera"], args.camera_fps) as camera:
        ctrl = LowController.instance()
        print("Waiting for the N2 DDS peer", flush=True)
        if not ctrl.init():
            raise RuntimeError("LowController initialization failed")
        controller = UserController(ctrl, MotorCmd, policy, camera)
        try:
            deadline = time.monotonic()
            while True:
                controller.process()
                now = time.monotonic()
                deadline = max(deadline + MOTOR_PERIOD, now)
                time.sleep(deadline - now)
        except KeyboardInterrupt:
            pass
        finally:
            if controller.status is not None:
                controller.handle_default_mode()


if __name__ == "__main__":
    main()
