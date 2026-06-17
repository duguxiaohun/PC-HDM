import numpy as np


class LinearSchedule:
    def __init__(self, start_val, end_val, total_iters=5):
        self.start_val = start_val
        self.end_val = end_val
        self.total_iters = total_iters
        self.count = 0
        self.last_val = self.start_val

    def step(self):
        if self.count > self.total_iters:
            return self.last_val
        ratio = self.count / self.total_iters
        val = ratio * (self.end_val - self.start_val) + self.start_val
        self.last_val = val
        self.count += 1
        return val

    def val(self):
        return self.last_val


class ExponentialSchedule:
    def __init__(self, start_val, gamma, end_val=None):
        self.start_val = start_val
        self.end_val = end_val
        self.gamma = gamma
        if end_val is not None:
            self.total_iters = int((np.log(end_val) - np.log(start_val)) / np.log(gamma))
        else:
            self.total_iters = None
        self.count = 0
        self.last_val = self.start_val

    def step(self):
        if self.total_iters is not None and self.count > self.total_iters:
            return self.last_val
        val = self.last_val * self.gamma
        self.last_val = val
        self.count += 1
        return val

    def val(self):
        return self.last_val


class TanhSchedule:
    def __init__(self, start_val, end_val, start_step, end_step, gamma=10e-6):
        self.start_val = start_val
        self.end_val = end_val
        self.start_step = start_step
        self.end_step = end_step
        self.gamma = gamma
        self.center_step = (self.end_step - self.start_step) / 2

    def val(self, step):
        val = np.tanh(self.gamma * (step - self.center_step))

        val = (val + 1) / 2
        return val
