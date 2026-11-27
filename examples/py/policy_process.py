from contextlib import contextmanager
import multiprocessing as mp
import signal

import numpy as np

from depth_policy import DepthPolicy


def add_inference_argument(parser):
    parser.add_argument("--inference-frames", type=int, choices=range(1, 6), default=1,
                        help="1: synchronous; 2–5: worker process, wait that many 2 ms ticks")


def run_policy(connection, path):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with connection:
        policy = DepthPolicy(path)
        depth_shape = policy.config["inputs"]["depth_m"][-2:]
        policy.step(policy.stand_pos, np.zeros_like(policy.stand_pos), [0, 0, 0, 1],
                    np.zeros(3), np.zeros(3), np.ones(depth_shape, dtype=np.float32))
        connection.send((policy.config, policy.stand_pos, policy.kp, policy.kd, policy.command_ranges))
        while True:
            try:
                reset, inputs = connection.recv()
            except EOFError:
                return
            if reset:
                policy.reset()
            try:
                target = policy.step(*inputs)
                result = target, policy.attention
            except ValueError as error:
                result = str(error)
            try:
                connection.send(result)
            except BrokenPipeError:
                return


class ProcessPolicy:
    def __init__(self, connection, inference_frames):
        self.connection = connection
        self.inference_frames = inference_frames
        if not connection.poll(30):
            raise RuntimeError("Timed out starting the policy worker")
        try:
            self.config, self.stand_pos, self.kp, self.kd, self.command_ranges = connection.recv()
        except EOFError as error:
            raise RuntimeError("Policy worker failed to start") from error
        self.pending = False
        self.reset()

    def reset(self):
        self.attention = None
        self.reset_next = True

    def submit(self, *inputs):
        if self.pending:
            raise ValueError("Policy inference did not finish within one policy period")
        try:
            # One small request can be in flight at a time. send() serializes the sensor snapshot.
            self.connection.send((self.reset_next, inputs))
        except OSError as error:
            raise ValueError("Policy worker disconnected") from error
        self.reset_next = False
        self.pending = True

    def poll(self):
        if not self.pending:
            return None
        try:
            if not self.connection.poll():
                return None
            result = self.connection.recv()
        except (EOFError, OSError) as error:
            raise ValueError("Policy worker disconnected") from error
        self.pending = False
        if self.reset_next:
            return None  # A result from before the most recent mode reset.
        if isinstance(result, str):
            raise ValueError(result)
        target, self.attention = result
        return target


@contextmanager
def open_depth_policy(path, inference_frames):
    if inference_frames == 1:
        yield DepthPolicy(path)
        return

    context = mp.get_context("spawn")
    connection, worker_connection = context.Pipe()
    with connection, worker_connection:
        worker = context.Process(target=run_policy, args=(worker_connection, path), daemon=True)
        worker.start()
        worker_connection.close()
        try:
            yield ProcessPolicy(connection, inference_frames)
        finally:
            connection.close()
            worker.join(timeout=1)
            if worker.is_alive():
                worker.terminate()
                worker.join()
            worker.close()
