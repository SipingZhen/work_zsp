import time

import numpy as np


def optical_rays(camera):
    """Return optical-frame rays at the centres of the model's depth pixels."""
    matrix = np.asarray(camera["intrinsic_matrix"])
    rows, cols = np.indices((camera["height"], camera["width"]))
    x = (cols + 0.5 - matrix[0, 2]) / matrix[0, 0]
    y = (rows + 0.5 - matrix[1, 2]) / matrix[1, 1]
    return np.stack((x, y, np.ones_like(x)), axis=-1)


class DepthSource:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.latest = None

    def publish(self, image):
        # The producer replaces the whole snapshot while readers may hold the old image.
        self.latest = self.clock(), image

    def latest_frame(self):
        latest = self.latest
        if latest is None or self.clock() - latest[0] > 0.2:
            return None
        return latest[1]
