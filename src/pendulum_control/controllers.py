"""PID with the actual control interval and saturation-aware integration."""
import math
import numpy as np


class PIDController:
    def __init__(self, dt, low=-170., high=170., kp=.35, ki=.04, kd=.08, deadband=.5, integral_limit=30.):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt, self.low, self.high = dt, low, high
        self.kp, self.ki, self.kd = kp, ki, kd
        self.deadband, self.integral_limit = deadband, integral_limit
        self.integral = 0.
        self.previous = None

    def target(self, acceleration):
        error = float(acceleration)
        if abs(error) < self.deadband:
            error = 0.
        derivative = 0. if self.previous is None else (error-self.previous)/self.dt
        self.previous = error
        candidate = float(np.clip(self.integral+error*self.dt, -self.integral_limit, self.integral_limit))
        raw = -math.degrees(self.kp*error+self.ki*candidate+self.kd*derivative)
        # A positive error drives the target lower; do not integrate further at that limit.
        pushing_lower = raw < self.low and error > 0
        pushing_upper = raw > self.high and error < 0
        if not (pushing_lower or pushing_upper):
            self.integral = candidate
        return float(np.clip(raw, self.low, self.high))
