import json

import numpy as np
import onnxruntime as ort


# SDK motor order: left arm, left leg, right arm, right leg.
SDK_JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("L", "R")
    for joint in (
        "arm_shoulder_pitch", "arm_shoulder_roll", "arm_shoulder_yaw", "arm_elbow",
        "leg_hip_yaw", "leg_hip_roll", "leg_hip_pitch", "leg_knee", "leg_ankle",
    )
]


def projected_gravity(quat_xyzw):
    norm = np.linalg.norm(quat_xyzw)
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("Invalid IMU quaternion")
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float32) / norm
    return np.array([
        2 * (w * y - x * z),
        -2 * (y * z + w * x),
        2 * (x * x + y * y) - 1,
    ], dtype=np.float32)


class DepthPolicy:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"],
        )
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.config = json.loads(metadata["ame_n2_depth"])
        cfg = self.config
        self.output_names = ["actions", "velocity"]
        if "attention_token_shape" in cfg:
            self.output_names.append("attention")

        self.to_model = [SDK_JOINT_NAMES.index(name) for name in cfg["joint_names"]]
        self.to_sdk = np.argsort(self.to_model)
        self.default_pos = np.asarray(cfg["default_joint_pos"], dtype=np.float32)
        self.default_vel = np.asarray(cfg["default_joint_vel"], dtype=np.float32)
        self.action_scale = np.asarray(cfg["action_scale"], dtype=np.float32)
        self.stand_pos = self.default_pos[self.to_sdk]
        self.kp = np.asarray(cfg["actuators"]["stiffness"])[self.to_sdk]
        self.kd = np.asarray(cfg["actuators"]["damping"])[self.to_sdk]
        self.command_ranges = np.asarray([
            cfg["command_ranges"][name]
            for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        ], dtype=np.float32)

        inputs = cfg["inputs"]
        self.proprio = np.empty(inputs["proprio"], dtype=np.float32)
        self.velocity = np.zeros(inputs["velocity_history"], dtype=np.float32)
        self.depth_stride = cfg["depth_history"]["frame_stride"]
        depth_steps = (inputs["depth_m"][1] - 1) * self.depth_stride + 1
        self.depth = np.empty((1, depth_steps, *inputs["depth_m"][-2:]), dtype=np.float32)
        self.last_action = np.zeros(len(self.to_model), dtype=np.float32)
        self.reset()

    def reset(self):
        self.first_step = True
        self.attention = None
        self.last_action.fill(0)
        self.velocity.fill(0)

    def step(self, joint_pos, joint_vel, quat_xyzw, gyro, command, depth_m):
        scales = self.config["observation_scales"]
        proprio = np.concatenate((
            gyro * scales["base_ang_vel"],
            projected_gravity(quat_xyzw),
            np.clip(command, self.command_ranges[:, 0], self.command_ranges[:, 1]),
            joint_pos[self.to_model] - self.default_pos,
            (joint_vel[self.to_model] - self.default_vel) * scales["joint_vel"],
            self.last_action,
        ))
        if not np.isfinite(proprio).all():
            raise ValueError("Invalid proprioception")

        # Repeat the first observation. Subsequent entries are oldest to newest.
        if self.first_step:
            self.proprio[:] = proprio
            self.depth[:] = depth_m
            self.first_step = False
        else:
            self.proprio[:, :-1] = self.proprio[:, 1:]
            self.proprio[:, -1] = proprio
            self.depth[:, :-1] = self.depth[:, 1:]
            self.depth[:, -1] = depth_m

        actions, velocity, *attention = self.session.run(self.output_names, {
            "proprio": self.proprio,
            "depth_m": self.depth[:, ::self.depth_stride],
            "velocity_history": self.velocity,
        })
        if not np.isfinite(actions).all() or not np.isfinite(velocity).all():
            raise ValueError("Nonfinite policy output")

        self.attention = attention[0] if attention else None
        self.last_action[:] = actions[0]
        self.velocity[:, :-1] = self.velocity[:, 1:]
        self.velocity[:, -1] = velocity
        return (self.default_pos + self.action_scale * self.last_action)[self.to_sdk]
