import argparse
from contextlib import contextmanager, nullcontext
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from depth_camera import DepthSource, optical_rays
from depth_policy import SDK_JOINT_NAMES
from policy_process import add_inference_argument, open_depth_policy
from sim_remote import LinuxJoystick, button_binding
from user_controller import MOTOR_PERIOD, STAND_DURATION, UserController, WorkMode


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = ROOT / "simulation/mjcf/N2.xml"
POLICY_PATH = ROOT / "policy/n2_depth.onnx"
ATTENTION_POLICY_PATH = ROOT / "policy/n2_depth_attention.onnx"
DEFAULT_BASE_HEIGHT = 0.67


def load_model(xml, terrain, step_height):
    spec = mujoco.MjSpec.from_file(str(xml))
    if terrain == "stairs":
        steps, start, tread, half_width = 10, 1.4, 0.3, 0.8
        obstacle = spec.find_default("obstacle")
        for i in range(steps):
            top = (i + 1) * step_height
            spec.worldbody.add_geom(
                default=obstacle, name=f"stair_{i + 1}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(start + (i + 0.5) * tread, 0, top / 2),
                size=(tread / 2, half_width, top / 2), rgba=(0.55, 0.60, 0.65, 1),
            )
        height = steps * step_height
        stair_end = start + steps * tread
        landing_length = 0.8
        spec.worldbody.add_geom(
            default=obstacle, name="stair_landing", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(stair_end + landing_length / 2, 0, height / 2),
            size=(landing_length / 2, half_width, height / 2), rgba=(0.55, 0.60, 0.65, 1),
        )
        angle = math.radians(20)
        ramp_start = stair_end + landing_length
        half_thickness = 0.05
        spec.worldbody.add_geom(
            default=obstacle, name="down_ramp", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(ramp_start + height / (2 * math.tan(angle))
                 - half_thickness * math.sin(angle), 0, height / 2 - half_thickness * math.cos(angle)),
            quat=(math.cos(angle / 2), 0, math.sin(angle / 2), 0),
            size=(height / (2 * math.sin(angle)), half_width, half_thickness),
            rgba=(0.65, 0.55, 0.40, 1),
        )
    model = spec.compile()
    model.opt.timestep = MOTOR_PERIOD
    return model


class MujocoDepth(DepthSource):
    def __init__(self, model, data, base_id, config):
        super().__init__(clock=lambda: data.time)
        self.model, self.data, self.base_id = model, data, base_id
        rays = optical_rays(config)
        self.shape = rays.shape[:2]
        rays = rays.reshape(-1, 3)
        self.rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        self.offset = np.asarray(config["position_base"], dtype=np.float64)
        quat = np.asarray(config["quaternion_wxyz"], dtype=np.float64)
        self.mount_rotation = np.empty(9)
        mujoco.mju_quat2Mat(self.mount_rotation, quat / np.linalg.norm(quat))
        self.mount_rotation = self.mount_rotation.reshape(3, 3)
        self.period = config["update_period"]
        self.max_depth = config["max_distance"]
        # A separate geometry group limits depth to world geometry, including the ground and stairs.
        model.geom_group[model.geom_bodyid == 0] = 5
        model.geom_group[model.geom_bodyid != 0] %= 5
        self.terrain_group = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
        self.distances = np.empty(len(rays))
        self.geom_ids = np.empty(len(rays), dtype=np.int32)
        self.reset()

    def reset(self):
        self.latest = None
        self.hits_world = None
        self.next_frame = self.data.time

    def update(self):
        if self.data.time < self.next_frame:
            return
        data, base_id = self.data, self.base_id
        base_rotation = data.xmat[base_id].reshape(3, 3)
        origin = data.xpos[base_id] + base_rotation @ self.offset
        camera_rotation = base_rotation @ self.mount_rotation
        directions = self.rays @ camera_rotation.T
        # A finite geometry cutoff can cull the infinite ground plane by its origin.
        # Apply the camera range to each hit's optical depth instead.
        mujoco.mj_multiRay(
            self.model, data, origin, directions.ravel(), self.terrain_group, True, -1,
            self.geom_ids, self.distances, None, len(self.rays), np.inf,
        )
        depth = self.distances * self.rays[:, 2]
        valid = (depth > 0) & (depth <= self.max_depth)
        hits = origin + self.distances[:, None] * directions
        hits[~valid] = np.nan
        self.hits_world = hits.reshape(*self.shape, 3)
        self.publish(np.where(valid, depth, 0).reshape(self.shape).astype(np.float32))
        self.next_frame += self.period


def draw_depth_points(scene, hits_world, depth, attention, token_shape, depth_range):
    scene.ngeom = 0
    if depth is None:
        return

    near, far = depth_range
    valid = np.isfinite(hits_world).all(axis=-1) & (depth >= near) & (depth <= far) & (depth > 0)
    points = hits_world[valid]
    if not len(points):
        return

    weights = np.zeros(len(points))
    if attention is not None and np.isfinite(attention).all():
        grid = attention.reshape(token_shape)
        rows = np.arange(depth.shape[0]) * grid.shape[0] // depth.shape[0]
        cols = np.arange(depth.shape[1]) * grid.shape[1] // depth.shape[1]
        weights = grid[np.ix_(rows, cols)][valid]
        weights = (weights - weights.min()) / max(weights.max() - weights.min(), 1e-8)

    points[:, 2] += 0.05  # Lift the markers above the surface for visibility.
    radii = 0.01 * (0.5 + 2.5 * weights)
    red = np.floor(weights * 7) / 7
    colors = np.column_stack((red, np.zeros_like(red), 1 - red, np.ones_like(red))).astype(np.float32)
    identity = np.eye(3).ravel()
    count = min(len(points), scene.maxgeom)
    for i in range(count):
        mujoco.mjv_initGeom(scene.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radii[i]),
                            points[i], identity, colors[i])
    scene.ngeom = count


class MujocoRobot:
    def __init__(self, model, policy, base_height):
        self.model, self.data, self.policy = model, mujoco.MjData(model), policy
        self.base_height = base_height
        self.base_id = model.body("base_link").id
        joints = [model.joint(name) for name in SDK_JOINT_NAMES]
        self.qpos = np.array([joint.qposadr.item() for joint in joints])
        self.qvel = np.array([joint.dofadr.item() for joint in joints])
        self.actuators = np.array([model.actuator(name.removesuffix("_joint")).id for name in SDK_JOINT_NAMES])
        self.status = SimpleNamespace(
            motor_data=[SimpleNamespace(pos=0., vel=0.) for _ in joints],
            imu_data=SimpleNamespace(), remote_data=None,
        )
        self.camera = MujocoDepth(model, self.data, self.base_id, policy.config["camera"])
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = self.base_height
        self.data.qpos[self.qpos] = self.policy.stand_pos
        self.support_pose = self.data.qpos[:7].copy()
        self.supported = True
        self.camera.reset()
        self.depth_sample = None
        self.policy_depth_sample = None
        self.controller = UserController(self, SimpleNamespace, self.policy, self.camera,
                                         clock=lambda: self.data.time)
        mujoco.mj_forward(self.model, self.data)

    def subscribe_robot_hardware_status(self, callback):
        self.on_status = callback

    def set_joint(self, commands):
        indices = [cmd.motor_id for cmd in commands]
        position = self.data.qpos[self.qpos[indices]]
        velocity = self.data.qvel[self.qvel[indices]]
        torque = [cmd.kp * (cmd.pos - q) + cmd.kd * (cmd.vel - dq) + cmd.tau
                  for cmd, q, dq in zip(commands, position, velocity)]
        actuators = self.actuators[indices]
        limits = self.model.actuator_ctrlrange[actuators]
        self.data.ctrl[actuators] = np.clip(torque, limits[:, 0], limits[:, 1])

    def step(self, remote):
        data = self.data
        mujoco.mj_step1(self.model, data)
        self.camera.update()
        for motor, pos, vel in zip(self.status.motor_data, data.qpos[self.qpos], data.qvel[self.qvel]):
            motor.pos, motor.vel = pos, vel
        self.status.imu_data.ori = data.sensor("orientation").data[[1, 2, 3, 0]]
        self.status.imu_data.angular_vel = data.sensor("angular-velocity").data
        self.status.remote_data = remote
        self.on_status(self.status)
        previous_attention = self.policy.attention
        self.controller.process()
        previous_sample = self.policy_depth_sample
        if self.controller.policy_sampled:
            self.policy_depth_sample = self.camera.hits_world, self.camera.latest_frame()
        attention = self.policy.attention
        if self.controller.mode != WorkMode.USERWALK:
            self.depth_sample = None
        elif attention is not None and attention is not previous_attention:
            sample = self.policy_depth_sample if self.policy.inference_frames == 1 else previous_sample
            self.depth_sample = (*sample, attention)

        mujoco.mj_step2(self.model, data)
        if self.controller.mode == WorkMode.USERWALK:
            self.supported = False
        if self.supported:
            data.qpos[:7] = self.support_pose
            data.qvel[:6] = 0


def scripted_remote(sim_time, command, policy):
    """Exercise the same button transitions when nobody is driving the simulation."""
    buttons = np.zeros(14, dtype=bool)
    if sim_time < 0.1:
        buttons[9] = True
    elif 0.2 <= sim_time < 0.3:
        buttons[[2, 10]] = True
    elif STAND_DURATION + 0.3 <= sim_time < STAND_DURATION + 0.4:
        buttons[[2, 5]] = True
    axes = (command / policy.command_ranges[[2, 0], 1]).clip(-1, 1)
    return SimpleNamespace(button=buttons, axes=axes)


@contextmanager
def passive_viewer(model, data, key_callback):
    from mujoco import viewer

    running = set(threading.enumerate())
    handle = viewer.launch_passive(model, data, key_callback=key_callback)
    render_threads = set(threading.enumerate()) - running
    try:
        yield handle
    finally:
        handle.close()
        # Let GLFW finish tearing down its context before Python exits.
        for thread in render_threads:
            thread.join()


def main():
    parser = argparse.ArgumentParser(description="Run the N2 depth deployment controller in MuJoCo.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--terrain", choices=("flat", "stairs"), default="stairs",
                        help="Ascending stairs with a landing and downhill ramp (default), or flat ground")
    parser.add_argument("--step-height", type=float, default=0.10)
    parser.add_argument("--base-height", type=float, default=DEFAULT_BASE_HEIGHT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-vis-attention", action="store_true", help="Hide the ground depth points")
    parser.add_argument("--duration", type=float, help="Simulation time in seconds")
    input_mode = parser.add_mutually_exclusive_group()
    input_mode.add_argument("--joystick", action="store_true", default=True, help="Use a Linux joystick (default)")
    input_mode.add_argument("--no-joystick", action="store_false", dest="joystick",
                            help="Run a scripted controller test")
    parser.add_argument("--vx", type=float, default=0.0, help="Scripted forward command in m/s")
    parser.add_argument("--yaw", type=float, default=0.0, help="Scripted yaw command in rad/s")
    parser.add_argument("--joystick-device", default="/dev/input/js0")
    parser.add_argument("--joystick-button-map", type=button_binding, action="append", default=[])
    parser.add_argument("--joystick-forward-axis", type=int, choices=range(256), default=1, metavar="AXIS")
    parser.add_argument("--joystick-yaw-axis", type=int, choices=range(256), default=3, metavar="AXIS")
    parser.add_argument("--joystick-forward-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--joystick-yaw-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--joystick-debug", action="store_true")
    add_inference_argument(parser)
    args = parser.parse_args()
    positive = (args.step_height, args.base_height) + (() if args.duration is None else (args.duration,))
    if any(not math.isfinite(x) or x <= 0 for x in positive):
        parser.error("step height, base height and duration must be positive and finite")
    if not math.isfinite(args.vx) or not math.isfinite(args.yaw):
        parser.error("scripted commands must be finite")
    duration = args.duration if args.duration is not None else (math.inf if args.joystick else 8.0)
    reset = threading.Event()

    def on_key(key):
        if key == ord("R"):
            reset.set()

    try:
        source = LinuxJoystick(
            args.joystick_device, args.joystick_forward_axis, args.joystick_yaw_axis,
            args.joystick_forward_sign, args.joystick_yaw_sign,
            args.joystick_button_map, args.joystick_debug,
        ) if args.joystick else nullcontext(None)
        show_points = not args.headless and not args.no_vis_attention
        policy_path = ATTENTION_POLICY_PATH if show_points else POLICY_PATH
        with source as joystick, open_depth_policy(policy_path, args.inference_frames) as policy:
            model = load_model(args.xml, args.terrain, args.step_height)
            robot = MujocoRobot(model, policy, args.base_height)
            view = nullcontext(None) if args.headless else passive_viewer(model, robot.data, on_key)
            run(robot, joystick, view, reset, duration, np.array([args.yaw, args.vx]))
    except OSError as error:
        parser.exit(1, f"{error}\nUse --joystick-device to select a joystick or --no-joystick for a scripted run.\n")


def run(robot, joystick, view, reset, duration, command):
    start = time.monotonic()
    ticks = 0
    token_shape = robot.policy.config.get("attention_token_shape")
    depth_range = robot.policy.config["depth_range"]
    with view as viewer:
        if viewer is not None:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = robot.base_id
                if token_shape is not None:
                    viewer.cam.distance = 3.0
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -20
                viewer.opt.geomgroup[5] = 1
        try:
            while ticks * MOTOR_PERIOD < duration and (viewer is None or viewer.is_running()):
                with viewer.lock() if viewer is not None else nullcontext():
                    if reset.is_set():
                        reset.clear()
                        robot.reset()
                    remote = (joystick.poll() if joystick is not None else
                              scripted_remote(robot.data.time, command, robot.policy))
                    robot.step(remote)
                    if viewer is not None and token_shape is not None and (ticks + 1) % 8 == 0:
                        sample = robot.depth_sample or (robot.camera.hits_world, robot.camera.latest_frame(), None)
                        draw_depth_points(viewer.user_scn, *sample, token_shape, depth_range)
                ticks += 1
                if viewer is not None and ticks % 8 == 0:
                    viewer.sync()
                if viewer is not None or joystick is not None or robot.policy.inference_frames > 1:
                    time.sleep(max(0.0, start + ticks * MOTOR_PERIOD - time.monotonic()))
        except KeyboardInterrupt:
            pass
    position = robot.data.qpos[:3]
    print(f"Simulated {ticks * MOTOR_PERIOD:.3f} s in {time.monotonic() - start:.3f} s. "
          f"Base position {position.round(3).tolist()}", flush=True)


if __name__ == "__main__":
    main()
