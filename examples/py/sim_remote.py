import argparse
import os
import struct
from types import SimpleNamespace

import numpy as np


EVENT = struct.Struct("<IhBB")  # Linux js_event
BUTTON_MAP = {9: 7, 2: 4, 10: 6, 5: 2, 11: 11}  # SDK logical -> Linux physical


def button_binding(value):
    logical, separator, physical = value.partition("=")
    if separator and logical.isdecimal() and physical.isdecimal():
        logical, physical = int(logical), int(physical)
        if logical in BUTTON_MAP and 0 <= physical <= 255:
            return logical, physical
    raise argparse.ArgumentTypeError("expected LOGICAL=PHYSICAL with a logical button from 2, 5, 9, 10, 11")


class LinuxJoystick:
    def __init__(self, path, forward_axis=1, yaw_axis=3, forward_sign=-1, yaw_sign=-1,
                 button_map=(), debug=False):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.raw_axes = np.zeros(256)
        self.raw_buttons = np.zeros(256, dtype=bool)
        self.axes = [yaw_axis, forward_axis]
        self.signs = np.array([yaw_sign, forward_sign])
        self.button_map = BUTTON_MAP | dict(button_map)
        self.debug = debug

    def __enter__(self):
        return self

    def __exit__(self, *_):
        os.close(self.fd)

    def poll(self):
        while True:
            try:
                event = os.read(self.fd, EVENT.size)
            except BlockingIOError:
                break
            if len(event) != EVENT.size:
                raise OSError(f"Joystick disconnected or incomplete event from {self.path}")
            _, value, event_type, index = EVENT.unpack(event)
            event_type &= 0x7f  # Initial events use the same data layout.
            if event_type == 0x01:
                if self.debug and self.raw_buttons[index] != bool(value):
                    print(f"joystick button {index} = {int(bool(value))}", flush=True)
                self.raw_buttons[index] = bool(value)
            elif event_type == 0x02:
                axis = value / (32768 if value < 0 else 32767)
                if self.debug and abs(axis - self.raw_axes[index]) >= 0.2:
                    print(f"joystick axis {index} = {axis:+.2f}", flush=True)
                self.raw_axes[index] = axis
        buttons = np.zeros(14, dtype=bool)
        for logical, physical in self.button_map.items():
            buttons[logical] = self.raw_buttons[physical]
        return SimpleNamespace(axes=self.raw_axes[self.axes] * self.signs, button=buttons)
