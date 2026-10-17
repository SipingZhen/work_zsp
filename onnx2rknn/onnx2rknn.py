"""Validate the exported N2 policy and convert it for RK3588S deployment."""

import argparse
from pathlib import Path

import onnx
import yaml


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "noetix_sdk_release" / "policy"
CONFIG = ROOT / "noetix_sdk_release" / "config" / "ning_user.yaml"
JOINT_SUFFIXES = {
    "arm_shoulder_pitch_joint": "arm_{}1_joint",
    "arm_shoulder_roll_joint": "arm_{}2_joint",
    "arm_shoulder_yaw_joint": "arm_{}3_joint",
    "arm_elbow_joint": "arm_{}4_joint",
    "leg_hip_yaw_joint": "leg_{}1_joint",
    "leg_hip_roll_joint": "leg_{}2_joint",
    "leg_hip_pitch_joint": "leg_{}3_joint",
    "leg_knee_joint": "leg_{}4_joint",
    "leg_ankle_joint": "leg_{}5_joint",
}


def validate_policy(path):
    model = onnx.load(str(path), load_external_data=False)
    inputs = model.graph.input
    outputs = model.graph.output
    shape = lambda value: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
    if len(inputs) != 1 or shape(inputs[0]) != [1, 310]:
        raise ValueError("N2 policy input must be [1, 310]")
    if len(outputs) != 1 or shape(outputs[0]) != [1, 18]:
        raise ValueError("N2 policy output must be [1, 18]")

    metadata = {item.key: item.value for item in model.metadata_props}
    config = yaml.safe_load(CONFIG.read_text())["user"]
    names = metadata["joint_names"].split(",")
    mapped_names = []
    for name in names:
        side, suffix = name.split("_", 1)
        if side not in ("L", "R") or suffix not in JOINT_SUFFIXES:
            raise ValueError(f"Unknown N2 policy joint: {name}")
        mapped_names.append(JOINT_SUFFIXES[suffix].format(side.lower()))
    if mapped_names != config["policy_joint_order"]:
        raise ValueError("ONNX joint_names differ from config/ning_user.yaml policy_joint_order")

    for key, field in (
        ("action_scale", "action_scale"),
        ("joint_stiffness", "stiffness"),
        ("joint_damping", "damping"),
    ):
        values = [float(value) for value in metadata[key].split(",")]
        if len(values) != 18 or any(
            abs(value - config["control"][field][name]) > 0.001
            for name, value in zip(mapped_names, values)
        ):
            raise ValueError(f"ONNX {key} differs from config/ning_user.yaml")
    defaults = [float(value) for value in metadata["default_joint_pos"].split(",")]
    if len(defaults) != 18 or any(
        abs(value - config["init_state"]["default_joint_angle"][name]) > 0.001
        for name, value in zip(mapped_names, defaults)
    ):
        raise ValueError("ONNX default_joint_pos differs from config/ning_user.yaml")
    print(f"Validated N2 policy: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=POLICY_DIR / "policy.onnx")
    parser.add_argument("--output", type=Path, default=POLICY_DIR / "policy.rknn")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    validate_policy(args.onnx)
    if args.validate_only:
        return

    from rknn.api import RKNN

    rknn = RKNN()
    try:
        rknn.config(target_platform="rk3588")
        if rknn.load_onnx(str(args.onnx)) != 0:
            raise RuntimeError("Failed to load ONNX policy")
        if rknn.build(do_quantization=False) != 0:
            raise RuntimeError("Failed to build RKNN policy")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if rknn.export_rknn(str(args.output)) != 0:
            raise RuntimeError("Failed to export RKNN policy")
        print(f"Exported RKNN policy: {args.output}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
