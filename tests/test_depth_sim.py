import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/py"))
from depth_policy import DepthPolicy
from realsense_depth import RealSenseDepth
from sim2sim import (
    ATTENTION_POLICY_PATH, DEFAULT_BASE_HEIGHT, DEFAULT_XML, POLICY_PATH,
    MujocoDepth, MujocoRobot, draw_depth_points, load_model,
)
from sim_remote import EVENT, LinuxJoystick, button_binding
from user_controller import MOTOR_PERIOD, STAND_DURATION, WorkMode


class DepthSimulationTests(unittest.TestCase):
    def test_attention_points_use_the_inference_frame(self):
        policy = DepthPolicy(ATTENTION_POLICY_PATH)
        baseline = DepthPolicy(POLICY_PATH)
        robot = MujocoRobot(load_model(DEFAULT_XML, "flat", 0.1), policy, DEFAULT_BASE_HEIGHT)
        remote = SimpleNamespace(button=np.zeros(14, dtype=bool), axes=np.zeros(2))
        robot.controller.set_mode(WorkMode.USERWALK, policy.stand_pos, robot.data.time)
        joint_pos = robot.data.qpos[robot.qpos]
        joint_vel = robot.data.qvel[robot.qvel]
        quat = robot.data.sensor("orientation").data[[1, 2, 3, 0]]
        gyro = robot.data.sensor("angular-velocity").data.copy()

        robot.step(remote)
        sample = robot.depth_sample
        hits, depth, attention = sample
        self.assertIs(hits, robot.camera.hits_world)
        np.testing.assert_array_equal(depth, policy.depth[0, -1])
        self.assertEqual(attention.shape, (1, 1, 96))
        self.assertTrue(np.isfinite(attention).all())
        target = baseline.step(joint_pos, joint_vel, quat, gyro, np.zeros(3, dtype=np.float32), depth)
        np.testing.assert_array_equal(robot.controller.walk_target, target)
        np.testing.assert_array_equal(policy.velocity, baseline.velocity)

        robot.camera.next_frame = robot.data.time
        robot.step(remote)
        self.assertIsNot(robot.camera.hits_world, hits)
        self.assertIs(robot.depth_sample, sample)
        robot.reset()
        self.assertIsNone(robot.depth_sample)

    def test_depth_points_use_attention_colors_and_valid_range(self):
        model = mujoco.MjModel.from_xml_string("<mujoco/>")
        scene = mujoco.MjvScene(model, maxgeom=4)
        depth = np.array([[0.5, 1.0], [0.0, 3.0]])
        hits = np.array([[[0., 0., 0.], [1., 0., 0.]], [[0., 1., 0.], [1., 1., 0.]]])
        draw_depth_points(scene, hits, depth, np.array([0.1, 0.9, 0.4, 0.3]), (2, 2), (0.3, 2.0))
        self.assertEqual(scene.ngeom, 2)
        np.testing.assert_array_equal(scene.geoms[0].rgba, [0., 0., 1., 1.])
        np.testing.assert_array_equal(scene.geoms[1].rgba, [1., 0., 0., 1.])
        np.testing.assert_allclose(scene.geoms[1].pos, [1., 0., 0.05])
        self.assertGreater(scene.geoms[1].size[0], scene.geoms[0].size[0])
        np.testing.assert_array_equal(hits[..., 2], 0)

    def test_stairs_join_the_landing_and_downhill_ramp(self):
        model = load_model(DEFAULT_XML, "stairs", 0.10)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        slope = math.tan(math.radians(20))
        ramp_end = 5.2 + 1.0 / slope
        for x, expected in (
            (1.55, 0.10), (4.399, 1.0), (4.401, 1.0), (4.8, 1.0),
            (5.199, 1.0), (5.201, 1.0 - 0.001 * slope),
            (6.2, 1.0 - slope), (ramp_end - 0.001, 0.001 * slope), (ramp_end + 0.001, 0.0),
        ):
            with self.subTest(x=x):
                distance = mujoco.mj_ray(model, data, np.array([x, 0., 3.]), np.array([0., 0., -1.]),
                                         None, True, -1, np.empty(1, dtype=np.int32), None)
                self.assertAlmostEqual(3.0 - distance, expected, places=6)

        for x, expected in ((1.55, 0.10), (4.8, 1.0), (6.2, 1.0 - slope)):
            for y, height in ((-0.81, 0.0), (-0.79, expected), (0.79, expected), (0.81, 0.0)):
                with self.subTest(x=x, y=y):
                    distance = mujoco.mj_ray(model, data, np.array([x, y, 3.]), np.array([0., 0., -1.]),
                                             None, True, -1, np.empty(1, dtype=np.int32), None)
                    self.assertAlmostEqual(3.0 - distance, height, places=6)

    def test_gamepad_events_drive_the_deployment_controller(self):
        policy = DepthPolicy(POLICY_PATH)
        robot = MujocoRobot(load_model(DEFAULT_XML, "flat", 0.1), policy, DEFAULT_BASE_HEIGHT)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "joystick"
            os.mkfifo(path)
            with LinuxJoystick(path) as joystick, path.open("wb", buffering=0) as device:
                def button(index, pressed):
                    device.write(EVENT.pack(0, pressed, 0x01, index))

                button(9, 1)
                robot.step(joystick.poll())
                self.assertEqual(robot.controller.mode, WorkMode.DEFAULT)
                button(7, 1)
                robot.step(joystick.poll())
                self.assertEqual(robot.controller.mode, WorkMode.LIE)
                button(7, 0)
                button(4, 1)
                button(6, 1)
                robot.step(joystick.poll())
                self.assertEqual(robot.controller.mode, WorkMode.STAND)
                button(6, 0)
                for _ in range(round(STAND_DURATION / MOTOR_PERIOD) + 1):
                    robot.step(joystick.poll())
                device.write(EVENT.pack(0, -8192, 0x82, 1))
                device.write(EVENT.pack(0, -16384, 0x02, 3))
                button(2, 1)
                mapped = joystick.poll()
                np.testing.assert_allclose(mapped.axes, [0.5, 0.25])
                robot.step(mapped)
                self.assertEqual(robot.controller.mode, WorkMode.USERWALK)
                self.assertFalse(policy.first_step)
                self.assertFalse(robot.supported)
                np.testing.assert_allclose(policy.proprio[0, -1, 6:9], [0.25, 0., 0.3])
                target = robot.controller.walk_target.copy()
                device.write(EVENT.pack(0, 0, 0x02, 3))
                for _ in range(robot.controller.decimation - 1):
                    robot.step(joystick.poll())
                np.testing.assert_array_equal(robot.controller.walk_target, target)
                robot.step(joystick.poll())
                self.assertFalse(np.array_equal(robot.controller.walk_target, target))
                for _ in range(round(1.5 / MOTOR_PERIOD)):
                    robot.step(joystick.poll())
                self.assertGreater(robot.data.qpos[0], 0.1)
                self.assertGreater(robot.data.qpos[2], 0.45)
                self.assertTrue(np.isfinite(policy.velocity).all())
                button(11, 1)
                robot.step(joystick.poll())
                self.assertEqual(robot.controller.mode, WorkMode.STAND)

    def test_fall_protection_disables_control_and_requires_a_fresh_press(self):
        policy = DepthPolicy(POLICY_PATH)
        robot = MujocoRobot(load_model(DEFAULT_XML, "flat", 0.1), policy, DEFAULT_BASE_HEIGHT)
        remote = SimpleNamespace(button=np.zeros(14, dtype=bool), axes=np.zeros(2))

        def set_tilt(degrees):
            half_angle = math.radians(degrees) / 2
            horizontal = math.sin(half_angle) / math.sqrt(2)
            robot.data.qpos[3:7] = [math.cos(half_angle), horizontal, horizontal, 0.0]

        robot.controller.set_mode(WorkMode.USERWALK, policy.stand_pos, robot.data.time)
        robot.controller.walk_target = policy.stand_pos
        robot.controller.policy_tick = 1
        robot.supported = False

        set_tilt(44.9)
        robot.step(remote)
        self.assertEqual(robot.controller.mode, WorkMode.USERWALK)
        self.assertEqual(robot.controller.policy_tick, 2)

        set_tilt(45.1)
        robot.data.qvel[robot.qvel] = 0.5
        robot.step(remote)
        self.assertEqual(robot.controller.mode, WorkMode.DEFAULT)
        self.assertEqual(robot.controller.policy_tick, 2)
        np.testing.assert_allclose(robot.data.ctrl[robot.actuators], -0.05)

        remote.button[9] = True
        set_tilt(60)
        robot.step(remote)
        self.assertEqual(robot.controller.mode, WorkMode.DEFAULT)
        set_tilt(0)
        robot.step(remote)
        self.assertEqual(robot.controller.mode, WorkMode.DEFAULT)
        remote.button[9] = False
        robot.step(remote)
        remote.button[9] = True
        robot.step(remote)
        self.assertEqual(robot.controller.mode, WorkMode.LIE)

        robot.status.imu_data.ori = np.zeros(4)
        robot.controller.on_status(robot.status)
        robot.controller.process()
        self.assertEqual(robot.controller.mode, WorkMode.DEFAULT)

    def test_joystick_override_and_disconnect(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "joystick"
            os.mkfifo(path)
            with LinuxJoystick(path, button_map=[button_binding("11=1")]) as joystick:
                with path.open("wb", buffering=0) as device:
                    device.write(EVENT.pack(0, 1, 0x81, 1))
                    self.assertTrue(joystick.poll().button[11])
                with self.assertRaisesRegex(OSError, "disconnected"):
                    joystick.poll()

    def test_mujoco_depth_projects_to_optical_axis_and_expires(self):
        policy = DepthPolicy(POLICY_PATH)
        robot = MujocoRobot(load_model(DEFAULT_XML, "flat", 0.1), policy, DEFAULT_BASE_HEIGHT)
        config = dict(policy.config["camera"], position_base=[0., 0., 0.], quaternion_wxyz=[0., 1., 0., 0.])
        camera = MujocoDepth(robot.model, robot.data, robot.base_id, config)
        camera.update()
        first = camera.latest_frame()
        self.assertEqual(first.shape, (16, 24))
        np.testing.assert_allclose(first, DEFAULT_BASE_HEIGHT)
        robot.data.qpos[2] += 0.1
        mujoco.mj_forward(robot.model, robot.data)
        robot.data.time = 0.01
        camera.update()
        self.assertIs(camera.latest_frame(), first)
        robot.data.time = 0.02
        camera.update()
        np.testing.assert_allclose(camera.latest_frame(), DEFAULT_BASE_HEIGHT + 0.1)
        robot.data.time = 0.23
        self.assertIsNone(camera.latest_frame())

    def test_realsense_still_publishes_calibrated_metric_depth(self):
        raw = np.array([[100, 200], [0, 400]], dtype=np.uint16)
        frame = SimpleNamespace(get_data=lambda: raw, get_units=lambda: 0.002)
        intrinsics = SimpleNamespace(width=2, height=2)
        profile = SimpleNamespace(get_stream=lambda _: SimpleNamespace(
            as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intrinsics),
        ))
        stream = []
        stopped = threading.Event()

        def next_frame(_):
            time.sleep(0.001)
            return True, SimpleNamespace(get_depth_frame=lambda: frame)

        fake_sdk = SimpleNamespace(
            context=lambda: SimpleNamespace(query_devices=lambda: [object()]),
            config=lambda: SimpleNamespace(enable_stream=lambda *args: stream.append(args)),
            pipeline=lambda _: SimpleNamespace(start=lambda _: profile, stop=stopped.set,
                                              try_wait_for_frames=next_frame),
            stream=SimpleNamespace(depth="depth"), format=SimpleNamespace(z16="z16"),
            rs2_project_point_to_pixel=lambda _, ray: ray[:2],
        )
        config = {"height": 2, "width": 3, "intrinsic_matrix": [[1, 0, .5], [0, 1, .5], [0, 0, 1]]}
        published = threading.Event()
        with patch.dict(sys.modules, {"pyrealsense2": fake_sdk}):
            camera = RealSenseDepth(config, 90)
            publish = camera.publish

            def notify(image):
                publish(image)
                published.set()

            with patch.object(camera, "publish", side_effect=notify), camera:
                self.assertTrue(published.wait(timeout=1))
                np.testing.assert_allclose(camera.latest_frame(), [[.2, .4, 0.], [0., .8, 0.]])
        self.assertEqual(stream, [("depth", 640, 480, "z16", 90)])
        self.assertTrue(stopped.is_set())
