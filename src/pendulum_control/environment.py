"""Shared deterministic simulator; reading an observation has no side effects."""
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "rope_arm.xml"


@dataclass(frozen=True)
class Config:
    frame_skip: int = 10
    filter_alpha: float = 0.15  # applied once per physics step
    warmup_s: float = 0.25     # undisturbed calibration precedes the kick
    kick_force: float = 60.0
    kick_duration_s: float = 0.15
    max_yaw_rate_deg_s: float = 240.0
    action_step_deg: float = 1.5
    stable_acc: float = 3.0
    stable_steps: int = 40
    max_steps: int = 800


class BallStabilizeEnv:
    def __init__(self, model_path=DEFAULT_MODEL, config=None, direction=(1., 0., 0.)):
        self.config = config or Config()
        c = self.config
        if c.frame_skip < 1 or c.max_steps < 1 or not 0 < c.filter_alpha <= 1:
            raise ValueError("Positive frame_skip/max_steps and filter_alpha in (0,1] required")
        if c.warmup_s < 0 or c.kick_duration_s <= 0 or c.max_yaw_rate_deg_s <= 0 or c.stable_steps < 1:
            raise ValueError("Invalid timing, rate or stability configuration")
        self.direction = np.asarray(direction, dtype=float)
        if self.direction.shape != (3,) or not np.isfinite(self.direction).all() or np.linalg.norm(self.direction) == 0:
            raise ValueError("direction must be a finite nonzero 3-vector")
        self.direction /= np.linalg.norm(self.direction)
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self.actuator = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_pos")
        jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, "joint0")
        self.yaw_adr = int(self.model.jnt_qposadr[jid])
        self.site = self._id(mujoco.mjtObj.mjOBJ_SITE, "imu_site")
        self.body = int(self.model.geom_bodyid[self._id(mujoco.mjtObj.mjOBJ_GEOM, "ball")])
        self.acc_adr = self._sensor("ball_acc")
        self.gyro_adr = self._sensor("ball_gyro")
        self.lo, self.hi = map(float, self.model.actuator_ctrlrange[self.actuator])
        if not np.allclose([self.lo, self.hi], self.model.jnt_range[jid]):
            raise ValueError("Actuator range must match joint range in radians; use models/rope_arm.xml")
        self.dt = float(self.model.opt.timestep)
        self.control_dt = self.dt * c.frame_skip
        self.obs_dim, self.n_actions = 8, 2
        self.reset()

    def _id(self, kind, name):
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"Model is missing {name}")
        return result

    def _sensor(self, name):
        sid = self._id(mujoco.mjtObj.mjOBJ_SENSOR, name)
        if self.model.sensor_dim[sid] != 3:
            raise ValueError(f"{name} must have three channels")
        return int(self.model.sensor_adr[sid])

    def _update_sensors(self):
        rotation = self.data.site_xmat[self.site].reshape(3, 3)
        acc = rotation @ self.data.sensordata[self.acc_adr:self.acc_adr+3]
        gyro = rotation @ self.data.sensordata[self.gyro_adr:self.gyro_adr+3]
        a = self.config.filter_alpha
        self.acc += a * (acc - self.acc)
        self.gyro += a * (gyro - self.gyro)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.acc, self.gyro = np.zeros(3), np.zeros(3)
        self.command_deg = self.prev_command_deg = 0.
        for _ in range(round(self.config.warmup_s / self.dt)):
            mujoco.mj_step(self.model, self.data)
            self._update_sensors()
        self.bias = self.acc.copy()
        self.steps = self.stable_count = 0
        self.start_time = float(self.data.time)
        self.terminated = self.truncated = False
        return self.observation()

    @property
    def elapsed(self):
        return float(self.data.time - self.start_time)

    def observation(self):
        yaw = float(self.data.qpos[self.yaw_adr])
        scale = max(abs(self.lo), abs(self.hi))
        command_rate = (self.command_deg - self.prev_command_deg) / self.control_dt
        return np.array([*self.acc, *self.gyro, np.clip(yaw / scale, -1, 1),
                         np.clip(command_rate / self.config.max_yaw_rate_deg_s, -1, 1)], dtype=np.float32)

    def step(self, action, *, stop_on_success=True):
        if action not in (0, 1):
            raise ValueError("Discrete action must be 0 (left) or 1 (right)")
        delta = self.config.action_step_deg * (1 if action == 1 else -1)
        return self.step_target(self.command_deg + delta, stop_on_success=stop_on_success)

    def step_target(self, target_deg, *, stop_on_success=True):
        if self.truncated or (self.terminated and stop_on_success):
            raise RuntimeError("Episode finished; call reset()")
        if not math.isfinite(target_deg):
            raise ValueError("Target must be finite")
        self.prev_command_deg = self.command_deg
        desired = float(np.clip(target_deg, math.degrees(self.lo), math.degrees(self.hi)))
        max_delta = self.config.max_yaw_rate_deg_s * self.control_dt
        self.command_deg += float(np.clip(desired-self.command_deg, -max_delta, max_delta))
        for _ in range(self.config.frame_skip):
            self.data.xfrc_applied[:] = 0
            if self.elapsed < self.config.kick_duration_s - self.dt * 0.1:
                self.data.xfrc_applied[self.body, :3] = self.config.kick_force * self.direction
            self.data.ctrl[self.actuator] = math.radians(self.command_deg)
            mujoco.mj_step(self.model, self.data)
            self._update_sensors()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Non-finite simulator state")
        horizontal = self.acc[:2] - self.bias[:2]
        axy = float(np.linalg.norm(horizontal))
        yaw_deg = math.degrees(float(self.data.qpos[self.yaw_adr]))
        reward = -math.tanh(axy / 10) - .05 * abs(yaw_deg)/90
        reward -= .02 * abs(self.command_deg-self.prev_command_deg)/self.config.action_step_deg
        after_kick = self.elapsed >= self.config.kick_duration_s
        self.stable_count = self.stable_count + 1 if after_kick and axy < self.config.stable_acc else 0
        if after_kick and axy < self.config.stable_acc:
            reward += 1
        newly_successful = self.stable_count == self.config.stable_steps
        if newly_successful:
            reward += 200
        self.terminated |= self.stable_count >= self.config.stable_steps
        self.steps += 1
        self.truncated = self.steps >= self.config.max_steps and not (self.terminated and stop_on_success)
        info = dict(time_s=self.elapsed, ax=float(horizontal[0]), ay=float(horizontal[1]), axy=axy,
                    yaw_deg=yaw_deg, command_deg=self.command_deg, stable_count=self.stable_count,
                    tilt_deg=math.degrees(math.acos(float(np.clip(self.data.site_xmat[self.site].reshape(3,3)[2,2],-1,1)))))
        return self.observation(), float(reward), bool(self.terminated and stop_on_success), bool(self.truncated), info

    def configuration(self):
        return dict(asdict(self.config), dt=self.dt, control_dt=self.control_dt,
                    direction=self.direction.tolist(), ctrlrange_rad=[self.lo,self.hi])
