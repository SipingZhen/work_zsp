from threading import Event, Thread
import time

import numpy as np


class RealSenseDepth:
    def __init__(self, camera, fps):
        import pyrealsense2 as rs

        self.rs = rs
        self.camera = camera
        context = rs.context()
        if not context.query_devices():
            raise RuntimeError("No RealSense camera connected")
        self.pipeline = rs.pipeline(context)
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        self.stop = Event()
        self.latest = None

    def __enter__(self):
        profile = self.pipeline.start(self.config)
        try:
            intr = profile.get_stream(self.rs.stream.depth).as_video_stream_profile().get_intrinsics()
            matrix = self.camera["intrinsic_matrix"]
            rows, cols = np.indices((self.camera["height"], self.camera["width"]))
            # Training rays are at col/row + 0.5. RealSense uses integer pixel centres.
            x = (cols + 0.5 - matrix[0][2]) / matrix[0][0]
            y = (rows + 0.5 - matrix[1][2]) / matrix[1][1]
            pixels = np.array([
                self.rs.rs2_project_point_to_pixel(intr, [u, v, 1.0])
                for u, v in zip(x.flat, y.flat)
            ])
            self.cols, self.rows = np.floor(pixels + 0.5).astype(int).T
            self.valid = (
                (self.cols >= 0) & (self.cols < intr.width)
                & (self.rows >= 0) & (self.rows < intr.height)
            )
            if not self.valid.any():
                raise ValueError("RealSense intrinsics do not cover the policy camera")
            self.thread = Thread(target=self.capture, name="realsense-depth")
            self.thread.start()
        except BaseException:
            self.pipeline.stop()
            raise
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.pipeline.stop()

    def capture(self):
        while not self.stop.is_set():
            ready, frames = self.pipeline.try_wait_for_frames(100)
            if not ready:
                continue
            frame = frames.get_depth_frame()
            if not frame:
                continue
            raw = np.asanyarray(frame.get_data())
            image = np.zeros((self.camera["height"], self.camera["width"]), dtype=np.float32)
            image.flat[self.valid] = raw[self.rows[self.valid], self.cols[self.valid]] * frame.get_units()
            # Publish one immutable snapshot for the control loop.
            self.latest = time.monotonic(), image

    def latest_frame(self):
        latest = self.latest
        if latest is None or time.monotonic() - latest[0] > 0.2:
            return None
        return latest[1]
