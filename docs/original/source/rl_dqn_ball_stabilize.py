# rl_dqn_ball_stabilize_improved.py
import math
import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer

import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# Config
# =========================
XML_PATH = "rope_arm.xml"

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RL (train longer)
TOTAL_EPISODES = 1500
MAX_EP_STEPS = 800

GAMMA = 0.99
LR = 3e-4
BATCH_SIZE = 256
REPLAY_SIZE = 200_000
LEARN_START = 2000
TRAIN_EVERY = 1
TARGET_UPDATE_EVERY = 1000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 120_000

# Environment stepping
FRAME_SKIP = 10
ACTION_STEP_DEG = 1.5
MAX_YAW_RATE_DEG_S = 240.0
ACC_LP_ALPHA = 0.15

# Disturbance
KICK_FORCE = 60.0
KICK_TIME = 0.15
KICK_DIR = np.array([1.0, 0.0, 0.0], dtype=np.float32)

# Stability condition (easier first)
STABLE_ACC_THR = 3.0
STABLE_COUNT_REQ = 40

# Reward shaping (bounded + strong success)
STABLE_BONUS = 1.0
DONE_BONUS = 200.0


# =========================
# Utilities
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def deg2rad(d):
    return d * math.pi / 180.0

def rad2deg(r):
    return r * 180.0 / math.pi


# =========================
# MuJoCo Env (discrete actions)
# =========================
class BallStabilizeEnv:
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        if self.model.nu < 1:
            raise RuntimeError("No actuators found in XML.")

        # actuator (yaw_pos assumed idx 0)
        self.IDX_YAW = 0
        self.dt = float(self.model.opt.timestep)

        # ctrlrange is radians internally
        self.cr_lo_rad = float(self.model.actuator_ctrlrange[self.IDX_YAW, 0])
        self.cr_hi_rad = float(self.model.actuator_ctrlrange[self.IDX_YAW, 1])
        self.cr_lo_deg = rad2deg(self.cr_lo_rad)
        self.cr_hi_deg = rad2deg(self.cr_hi_rad)

        # joint0 (yaw hinge)
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint0")
        if jid < 0:
            raise RuntimeError("Joint 'joint0' not found.")
        self.qpos_adr = int(self.model.jnt_qposadr[jid])

        # imu site
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_site")
        if self.site_id < 0:
            raise RuntimeError("Site 'imu_site' not found.")

        # sensors
        self.sid_acc = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ball_acc")
        if self.sid_acc < 0:
            raise RuntimeError("Sensor 'ball_acc' not found.")
        self.acc_adr = int(self.model.sensor_adr[self.sid_acc])
        self.acc_dim = int(self.model.sensor_dim[self.sid_acc])
        if self.acc_dim != 3:
            raise RuntimeError("Accelerometer dim must be 3.")

        # gyro (recommended)
        self.sid_gyro = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ball_gyro")
        self.have_gyro = self.sid_gyro >= 0
        if self.have_gyro:
            self.gyro_adr = int(self.model.sensor_adr[self.sid_gyro])
            self.gyro_dim = int(self.model.sensor_dim[self.sid_gyro])
            if self.gyro_dim != 3:
                self.have_gyro = False

        # ball geom -> body for disturbance
        gid_ball = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        if gid_ball < 0:
            raise RuntimeError("Geom 'ball' not found.")
        self.ball_body_id = int(self.model.geom_bodyid[gid_ball])

        # internal
        self.acc_filt = np.zeros(3, dtype=np.float32)
        self.gyro_filt = np.zeros(3, dtype=np.float32)
        self.command_deg = 0.0
        self.prev_command_deg = 0.0
        self.sim_time = 0.0
        self.step_count = 0
        self.stable_count = 0

        # observation dim:
        # [ax, ay, az] + [gx,gy,gz if exists] + [yaw_norm] + [yaw_rate_norm]
        self.obs_dim = 3 + (3 if self.have_gyro else 0) + 2
        self.n_actions = 2  # left/right

    def _set_actuator_deg(self, yaw_deg: float):
        yaw_deg = clamp(yaw_deg, self.cr_lo_deg, self.cr_hi_deg)
        self.data.ctrl[self.IDX_YAW] = clamp(deg2rad(yaw_deg), self.cr_lo_rad, self.cr_hi_rad)

    def _read_imu_world(self):
        acc_site = np.array(self.data.sensordata[self.acc_adr:self.acc_adr+3], dtype=np.float32)
        R = np.array(self.data.site_xmat[self.site_id], dtype=np.float32).reshape(3, 3)
        acc_world = R @ acc_site
        self.acc_filt = (1.0 - ACC_LP_ALPHA) * self.acc_filt + ACC_LP_ALPHA * acc_world

        if self.have_gyro:
            gyro_site = np.array(self.data.sensordata[self.gyro_adr:self.gyro_adr+3], dtype=np.float32)
            gyro_world = R @ gyro_site
            self.gyro_filt = (1.0 - ACC_LP_ALPHA) * self.gyro_filt + ACC_LP_ALPHA * gyro_world

        return self.acc_filt.copy(), (self.gyro_filt.copy() if self.have_gyro else None)

    def _get_yaw_deg(self):
        return rad2deg(float(self.data.qpos[self.qpos_adr]))

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.acc_filt[:] = 0
        self.gyro_filt[:] = 0
        self.command_deg = 0.0
        self.prev_command_deg = 0.0
        self.sim_time = 0.0
        self.step_count = 0
        self.stable_count = 0

        # warm-up a bit
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.dt

        return self._get_obs()

    def _get_obs(self):
        acc, gyro = self._read_imu_world()
        yaw_deg = self._get_yaw_deg()

        denom = max(1e-6, max(abs(self.cr_lo_deg), abs(self.cr_hi_deg)))
        yaw_norm = float(np.clip(yaw_deg / denom, -1.0, 1.0))

        dyaw = (self.command_deg - self.prev_command_deg)
        yaw_rate = float((dyaw) / (FRAME_SKIP * self.dt))
        yaw_rate_norm = float(np.clip(yaw_rate / MAX_YAW_RATE_DEG_S, -1.0, 1.0))

        parts = [acc[0], acc[1], acc[2]]
        if self.have_gyro and gyro is not None:
            parts += [gyro[0], gyro[1], gyro[2]]
        parts += [yaw_norm, yaw_rate_norm]
        return np.array(parts, dtype=np.float32)

    def step(self, action: int):
        self.prev_command_deg = self.command_deg

        # discrete action -> change desired yaw
        desired = self.command_deg + (ACTION_STEP_DEG if action == 1 else -ACTION_STEP_DEG)
        desired = clamp(desired, self.cr_lo_deg, self.cr_hi_deg)

        # rate limit (per RL step)
        max_step = MAX_YAW_RATE_DEG_S * (FRAME_SKIP * self.dt)
        d = desired - self.command_deg
        if abs(d) > max_step:
            self.command_deg += max_step if d > 0 else -max_step
        else:
            self.command_deg = desired

        self.command_deg = clamp(self.command_deg, self.cr_lo_deg, self.cr_hi_deg)

        # simulate FRAME_SKIP steps
        for _ in range(FRAME_SKIP):
            self.data.xfrc_applied[:, :] = 0.0
            if self.sim_time < KICK_TIME:
                self.data.xfrc_applied[self.ball_body_id, 0:3] = (KICK_FORCE * KICK_DIR)
            self._set_actuator_deg(self.command_deg)
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.dt

        obs = self._get_obs()

        # reward: bounded accel penalty + mild yaw/smoothness penalties
        acc, _ = self._read_imu_world()
        ax, ay = float(acc[0]), float(acc[1])
        axy = math.sqrt(ax * ax + ay * ay)

        scale = 10.0
        r_acc = -math.tanh(axy / scale)  # [-1, 0]

        yaw_deg = self._get_yaw_deg()
        r_yaw = -0.05 * (abs(yaw_deg) / 90.0)  # mild

        r_dyaw = -0.02 * (abs(self.command_deg - self.prev_command_deg) / max(1e-6, ACTION_STEP_DEG))

        reward = r_acc + r_yaw + r_dyaw

        # stability detection
        if axy < STABLE_ACC_THR:
            self.stable_count += 1
            reward += STABLE_BONUS
        else:
            self.stable_count = 0

        done = False
        if self.stable_count >= STABLE_COUNT_REQ:
            done = True
            reward += DONE_BONUS

        self.step_count += 1
        if self.step_count >= MAX_EP_STEPS:
            done = True

        info = {
            "ax": ax, "ay": ay, "axy": axy,
            "yaw_deg": yaw_deg,
            "cmd_deg": self.command_deg,
            "stable_count": self.stable_count,
        }
        return obs, float(reward), done, info


# =========================
# DQN
# =========================
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    ns: np.ndarray
    d: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s = np.stack([b.s for b in batch]).astype(np.float32)
        a = np.array([b.a for b in batch], dtype=np.int64)
        r = np.array([b.r for b in batch], dtype=np.float32)
        ns = np.stack([b.ns for b in batch]).astype(np.float32)
        d = np.array([b.d for b in batch], dtype=np.float32)
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buf)


def linear_epsilon(step: int):
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    return EPS_START + (EPS_END - EPS_START) * (step / EPS_DECAY_STEPS)


def train_dqn():
    env = BallStabilizeEnv(XML_PATH)

    q = QNet(env.obs_dim, env.n_actions).to(DEVICE)
    tq = QNet(env.obs_dim, env.n_actions).to(DEVICE)
    tq.load_state_dict(q.state_dict())
    tq.eval()

    opt = optim.Adam(q.parameters(), lr=LR)
    rb = ReplayBuffer(REPLAY_SIZE)

    ep_rewards = []
    ep_lengths = []
    ep_final_axy = []

    global_step = 0

    for ep in range(TOTAL_EPISODES):
        s = env.reset()
        total_r = 0.0
        last_info = None

        for t in range(MAX_EP_STEPS):
            eps = linear_epsilon(global_step)
            global_step += 1

            if random.random() < eps:
                a = random.randint(0, env.n_actions - 1)
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, device=DEVICE).unsqueeze(0))
                    a = int(torch.argmax(qs, dim=1).item())

            ns, r, done, info = env.step(a)
            last_info = info
            total_r += r

            rb.push(Transition(s=s, a=a, r=r, ns=ns, d=1.0 if done else 0.0))
            s = ns

            if len(rb) >= LEARN_START and (global_step % TRAIN_EVERY == 0):
                bs, ba, br, bns, bd = rb.sample(BATCH_SIZE)

                bs_t = torch.tensor(bs, device=DEVICE)
                ba_t = torch.tensor(ba, device=DEVICE).unsqueeze(1)
                br_t = torch.tensor(br, device=DEVICE).unsqueeze(1)
                bns_t = torch.tensor(bns, device=DEVICE)
                bd_t = torch.tensor(bd, device=DEVICE).unsqueeze(1)

                q_sa = q(bs_t).gather(1, ba_t)

                with torch.no_grad():
                    next_a = torch.argmax(q(bns_t), dim=1, keepdim=True)
                    next_q = tq(bns_t).gather(1, next_a)
                    target = br_t + (1.0 - bd_t) * GAMMA * next_q

                loss = nn.SmoothL1Loss()(q_sa, target)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()

            if global_step % TARGET_UPDATE_EVERY == 0:
                tq.load_state_dict(q.state_dict())

            if done:
                break

        ep_rewards.append(total_r)
        ep_lengths.append(t + 1)
        ep_final_axy.append(last_info["axy"] if last_info is not None else np.nan)

        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1:4d}/{TOTAL_EPISODES} | eps={linear_epsilon(global_step):.3f} | "
                  f"R={total_r:8.2f} | steps={t+1:4d} | final |a_xy|={ep_final_axy[-1]:6.3f}")

    # plots
    plt.figure()
    plt.plot(ep_rewards)
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.title("DQN training: episode reward")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(ep_lengths)
    plt.xlabel("episode")
    plt.ylabel("episode length (RL steps)")
    plt.title("DQN training: episode length")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(ep_final_axy)
    plt.xlabel("episode")
    plt.ylabel("final horizontal accel magnitude |a_xy|")
    plt.title("DQN training: final |a_xy|")
    plt.grid(True)
    plt.show()

    return q


# =========================
# Evaluate with viewer (stay open, no fast repeat)
# =========================
def run_viewer(policy_net: QNet):
    env = BallStabilizeEnv(XML_PATH)
    s = env.reset()

    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    ts, axy_log, ax_log, ay_log, yaw_log, cmd_log = [], [], [], [], [], []

    while viewer.is_running():
        with torch.no_grad():
            qs = policy_net(torch.tensor(s, device=DEVICE).unsqueeze(0))
            a = int(torch.argmax(qs, dim=1).item())

        s, r, done, info = env.step(a)

        ts.append(env.sim_time)
        ax_log.append(info["ax"])
        ay_log.append(info["ay"])
        axy_log.append(info["axy"])
        yaw_log.append(info["yaw_deg"])
        cmd_log.append(info["cmd_deg"])

        viewer.sync()

        # run at ~60 fps so it doesn't look like "fast repeating"
        time.sleep(1/60)

        if done:
            # pause ~2 seconds then reset (so you SEE it)
            for _ in range(120):
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(1/60)
            s = env.reset()

    viewer.close()

    # plots
    if len(ts) > 10:
        plt.figure()
        plt.plot(ts, ax_log, label="ax")
        plt.plot(ts, ay_log, label="ay")
        plt.plot(ts, axy_log, label="|a_xy|")
        plt.axhline(STABLE_ACC_THR, linestyle="--")
        plt.xlabel("sim time (s)")
        plt.ylabel("acc (m/s^2)")
        plt.title("Evaluation: IMU horizontal acceleration")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(ts, yaw_log, label="yaw (deg)")
        plt.plot(ts, cmd_log, label="cmd (deg)", linestyle="--")
        plt.xlabel("sim time (s)")
        plt.ylabel("deg")
        plt.title("Evaluation: yaw")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    policy = train_dqn()
    run_viewer(policy)