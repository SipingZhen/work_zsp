from enum import Enum
import time

import numpy as np


MOTOR_PERIOD = 0.002
STAND_DURATION = 2.0
STATUS_TIMEOUT = 0.2


class WorkMode(Enum):
    DEFAULT = 0
    LIE = 1
    STAND = 2
    USERWALK = 3


class UserController:
    def __init__(self, ctrl, motor_cmd, policy, camera, *, clock=time.monotonic):
        self.ctrl = ctrl
        self.policy = policy
        self.camera = camera
        self.clock = clock
        self.status = None
        self.buttons = np.zeros(14, dtype=bool)
        self.mode = WorkMode.DEFAULT
        self.commands = [motor_cmd() for _ in policy.stand_pos]
        for i, cmd in enumerate(self.commands):
            cmd.motor_id = i
            cmd.vel = cmd.tau = 0.0

        # The SDK is commanded at 500 Hz and holds each 50 Hz policy target.
        self.decimation = round(policy.config["control_dt"] / MOTOR_PERIOD)
        self.pose_kp = np.tile([20.0] * 4 + [50.0] * 5, 2)
        self.pose_kd = np.tile([0.1] * 4 + [1.0] * 5, 2)
        self.lie_pos = np.zeros_like(policy.stand_pos)
        self.ctrl.subscribe_robot_hardware_status(self.on_status)

    def on_status(self, status):
        self.status = self.clock(), status

    def set_mode(self, mode, joint_pos, now):
        self.mode = mode
        self.mode_start = now
        self.start_pos = joint_pos
        if mode == WorkMode.USERWALK:
            self.policy.reset()
            self.policy_tick = 0
        print(mode.name, flush=True)

    def send_targets(self, target, kp, kd):
        for cmd, pos, stiffness, damping in zip(self.commands, target, kp, kd):
            cmd.pos = float(pos)
            cmd.kp = float(stiffness)
            cmd.kd = float(damping)
        self.ctrl.set_joint(self.commands)

    def handle_default_mode(self):
        for cmd in self.commands:
            cmd.pos = cmd.kp = 0.0
            cmd.kd = 0.1
        self.ctrl.set_joint(self.commands)

    def handle_pose_mode(self, now):
        target = self.policy.stand_pos if self.mode == WorkMode.STAND else self.lie_pos
        progress = min((now - self.mode_start) / STAND_DURATION, 1.0)
        self.send_targets(self.start_pos + progress * (target - self.start_pos),
                          self.pose_kp, self.pose_kd)

    def process(self):
        snapshot = self.status
        if snapshot is None:
            return
        now = self.clock()
        stamp, status = snapshot
        if now - stamp > STATUS_TIMEOUT:
            if self.mode != WorkMode.DEFAULT:
                print("Robot status timed out", flush=True)
                self.set_mode(WorkMode.DEFAULT, None, now)
            self.handle_default_mode()
            return

        motors = status.motor_data
        joint_pos = np.array([motor.pos for motor in motors], dtype=np.float32)
        if not np.isfinite(joint_pos).all():
            if self.mode != WorkMode.DEFAULT:
                self.set_mode(WorkMode.DEFAULT, None, now)
            self.handle_default_mode()
            return

        remote = status.remote_data
        buttons = remote.button == 1
        pressed = buttons & ~self.buttons
        self.buttons = buttons
        stand_pressed = buttons[2] and buttons[10] and (pressed[2] or pressed[10])
        walk_pressed = buttons[2] and buttons[5] and (pressed[2] or pressed[5])
        if pressed[9]:
            mode = WorkMode.LIE if self.mode == WorkMode.DEFAULT else WorkMode.DEFAULT
            self.set_mode(mode, joint_pos, now)
        elif pressed[11] and self.mode == WorkMode.DEFAULT:
            self.set_mode(WorkMode.LIE, joint_pos, now)
        elif (
            (stand_pressed and self.mode in (WorkMode.LIE, WorkMode.USERWALK))
            or (pressed[11] and self.mode == WorkMode.USERWALK)
        ):
            self.set_mode(WorkMode.STAND, joint_pos, now)
        elif self.mode == WorkMode.STAND and walk_pressed:
            if now - self.mode_start < STAND_DURATION:
                print("Wait for the standing transition to finish", flush=True)
            elif self.camera.latest_frame() is None:
                print("Waiting for RealSense depth", flush=True)
            else:
                self.set_mode(WorkMode.USERWALK, joint_pos, now)

        if self.mode == WorkMode.USERWALK:
            depth = self.camera.latest_frame()
            if depth is None:
                print("RealSense depth timed out", flush=True)
                self.set_mode(WorkMode.STAND, joint_pos, now)
            else:
                try:
                    self.handle_walk_mode(status, motors, joint_pos, remote.axes, depth)
                    return
                except ValueError as error:
                    print(error, flush=True)
                    self.set_mode(WorkMode.STAND, joint_pos, now)

        if self.mode == WorkMode.DEFAULT:
            self.handle_default_mode()
        else:
            self.handle_pose_mode(now)

    def handle_walk_mode(self, status, motors, joint_pos, axes, depth):
        if self.policy_tick == 0:
            joint_vel = np.array([motor.vel for motor in motors], dtype=np.float32)
            imu = status.imu_data
            command = np.asarray([axes[1], 0.0, axes[0]], dtype=np.float32)
            command *= self.policy.command_ranges[:, 1]
            self.walk_target = self.policy.step(
                joint_pos, joint_vel, imu.ori, imu.angular_vel,
                command, depth,
            )
        self.send_targets(self.walk_target, self.policy.kp, self.policy.kd)
        self.policy_tick = (self.policy_tick + 1) % self.decimation
