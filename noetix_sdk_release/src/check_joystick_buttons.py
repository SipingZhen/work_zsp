#!/usr/bin/env python3
"""Print the physical button number reported by a Linux joystick device."""

import argparse
import struct
import sys


EVENT = struct.Struct("<IhBB")  # time, value, type, number
JS_EVENT_BUTTON = 0x01
JS_EVENT_INIT = 0x80


def button_presses(device):
    while True:
        event = device.read(EVENT.size)
        if not event:
            return
        if len(event) != EVENT.size:
            raise RuntimeError("收到不完整的手柄事件")

        _, value, event_type, number = EVENT.unpack(event)
        if event_type & JS_EVENT_INIT:
            continue
        if event_type & JS_EVENT_BUTTON and value:
            yield number


def main():
    parser = argparse.ArgumentParser(description="显示手柄按键的物理编号")
    parser.add_argument("device", nargs="?", default="/dev/input/js0",
                        help="手柄设备路径，默认 /dev/input/js0")
    args = parser.parse_args()

    try:
        with open(args.device, "rb", buffering=0) as device:
            print(f"正在监听 {args.device}，按任意手柄按键查看编号；Ctrl+C 退出。", flush=True)
            for number in button_presses(device):
                print(f"按键编号: {number}", flush=True)
    except KeyboardInterrupt:
        print("\n已退出。")
    except (OSError, RuntimeError) as exc:
        print(f"无法读取手柄 {args.device}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
