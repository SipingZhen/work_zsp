from threading import Event, Thread

import numpy as np

from depth_camera import DepthSource, optical_rays


class RealSenseDepth(DepthSource):
    def __init__(self, camera, fps):
        import pyrealsense2 as rs

        super().__init__()
        self.rs = rs
        self.camera = camera
        context = rs.context()
        if not context.query_devices():
            raise RuntimeError("No RealSense camera connected")
        self.pipeline = rs.pipeline(context)
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        self.stop = Event()

    def __enter__(self):
        profile = self.pipeline.start(self.config)
        try:
            intr = profile.get_stream(self.rs.stream.depth).as_video_stream_profile().get_intrinsics()
            # Training rays are at col/row + 0.5. RealSense uses integer pixel centres.
            pixels = np.array([
                self.rs.rs2_project_point_to_pixel(intr, ray)
                for ray in optical_rays(self.camera).reshape(-1, 3)
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
            self.publish(image)
