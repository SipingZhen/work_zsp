"""Linux joystick events for the N2 MuJoCo adapter."""

import errno
import os
import struct


EVENT = struct.Struct("<IhBB")  # Linux joystick API: time, value, type, number
AXIS = 0x02
BUTTON = 0x01
DEFAULT_BUTTON_MAP = {9: 7, 2: 4, 10: 6, 5: 2, 6: 3, 8: 0}


class LinuxJoystick:
    def __init__(self, path, forward_axis=1, yaw_axis=3, forward_sign=-1, yaw_sign=-1,
                 button_map=None, debug=False):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.axes = [0.0] * 8
        self.logged_axes = [0.0] * 8
        self.raw_buttons = [0] * 32
        self.forward_axis = forward_axis
        self.yaw_axis = yaw_axis
        self.forward_sign = forward_sign
        self.yaw_sign = yaw_sign
        self.button_map = DEFAULT_BUTTON_MAP.copy()
        if button_map:
            self.button_map.update(button_map)
        self.debug = debug

    @property
    def buttons(self):
        return [self.raw_buttons[self.button_map.get(i, i)] for i in range(14)]

    def poll(self):
        while True:
            try:
                event = os.read(self.fd, EVENT.size)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno in (errno.ENODEV, errno.EIO):
                    raise RuntimeError(f"Joystick disconnected: {self.path}") from exc
                raise
            if not event:
                raise RuntimeError(f"Joystick disconnected: {self.path}")
            if len(event) != EVENT.size:
                raise RuntimeError(f"Incomplete joystick event from {self.path}")
            _, value, event_type, number = EVENT.unpack(event)
            event_type &= 0x7f  # JS_EVENT_INIT carries the same initial axis/button state.
            if event_type == AXIS and number < len(self.axes):
                axis = max(-1.0, value / (32768.0 if value < 0 else 32767.0))
                if self.debug and abs(axis - self.logged_axes[number]) >= 0.2:
                    print(f"joystick axis {number}: {axis:+.2f}", flush=True)
                    self.logged_axes[number] = axis
                self.axes[number] = axis
            elif event_type == BUTTON and number < len(self.raw_buttons):
                pressed = int(value != 0)
                if self.debug and pressed != self.raw_buttons[number]:
                    print(f"joystick button {number}: {pressed}", flush=True)
                self.raw_buttons[number] = pressed

    def controller_axes(self):
        # Adapt the Linux event direction to remote_data.axes. The command
        # mapping itself is shared with the robot in usercontroller.cpp.
        return self.yaw_sign * self.axes[self.yaw_axis], self.forward_sign * self.axes[self.forward_axis]

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
