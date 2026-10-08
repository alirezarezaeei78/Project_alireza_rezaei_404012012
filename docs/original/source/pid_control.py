import math
import time
import numpy as np
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer

XML_PATH = "rope_arm.xml"

# ---------------- helpers ----------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def deg2rad(d):
    return d * math.pi / 180.0

def rad2deg(r):
    return r * 180.0 / math.pi

def wrap_deg(a):
    while a > 180: a -= 360
    while a < -180: a += 360
    return a

# ---------------- load ----------------
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

if model.nu < 1:
    raise RuntimeError("No actuators found.")

IDX_YAW = 0
dt = float(model.opt.timestep)

# actuator ctrlrange is radians internally
cr_lo_rad = float(model.actuator_ctrlrange[IDX_YAW, 0])
cr_hi_rad = float(model.actuator_ctrlrange[IDX_YAW, 1])
cr_lo_deg = rad2deg(cr_lo_rad)
cr_hi_deg = rad2deg(cr_hi_rad)

# joint0
jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint0")
if jid < 0:
    raise RuntimeError("Joint 'joint0' not found.")
qpos_adr = int(model.jnt_qposadr[jid])

# imu site + accelerometer sensor
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_site")
if site_id < 0:
    raise RuntimeError("Site 'imu_site' not found.")

sid_acc = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ball_acc")
if sid_acc < 0:
    raise RuntimeError("Sensor 'ball_acc' not found.")
acc_adr = int(model.sensor_adr[sid_acc])
acc_dim = int(model.sensor_dim[sid_acc])
if acc_dim != 3:
    raise RuntimeError("Accelerometer must be 3D.")

# geom "ball" -> owning body
gid_ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
if gid_ball < 0:
    raise RuntimeError("Geom 'ball' not found.")
ball_body_id = int(model.geom_bodyid[gid_ball])

def set_actuator_deg(yaw_deg: float):
    # clamp in degrees to ctrlrange
    yaw_deg = clamp(yaw_deg, cr_lo_deg, cr_hi_deg)
    data.ctrl[IDX_YAW] = deg2rad(yaw_deg)

# ---------------- control goal ----------------
# With 1 input (yaw), control only ONE horizontal axis reliably.
# We'll regulate world-X acceleration to 0 (you can switch to Y easily).
USE_AXIS = 0  # 0 -> ax, 1 -> ay

# PID gains (start stable, then tune)
KP = 0.35
KI = 0.04
KD = 0.08

# Anti-windup limits
I_CLAMP = 30.0   # clamp integral term (in "acc units * sec")
DEADBAND = 0.5   # m/s^2: ignore tiny accel noise

# Smoothing
ACC_LP_ALPHA = 0.15

# Command limits
MAX_YAW_RATE_DEG_S = 180.0  # slower = more stable

HOME_DEG = 0.0

# Disturbance: force kick
KICK_FORCE = 60.0
KICK_TIME  = 0.15

# Run settings
T_TOTAL = 12.0
WARMUP = 0.25

# Render/log rate (THIS FIXES FREEZE)
RENDER_HZ = 60.0
LOG_HZ    = 200.0

# ---------------- init ----------------
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

command_deg = HOME_DEG
i_term = 0.0
prev_e = 0.0
acc_filt = np.zeros(3)

# Bias calibration (optional but helps): measure accel baseline at rest
# We'll collect for a short time before kick/control, then subtract it.
BIAS_TIME = 0.2
bias_acc_sum = np.zeros(3)
bias_n = 0

# logs
ts, ax_log, ay_log, yaw_cmd_log, yaw_cur_log = [], [], [], [], []

viewer = mujoco.viewer.launch_passive(model, data)

sim_time = 0.0
wall0 = time.perf_counter()

next_render = 0.0
next_log = 0.0

# ---------------- main loop ----------------
while sim_time < T_TOTAL and viewer.is_running():
    # Run sim "as fast as needed", but pace to real time overall
    # by stepping multiple mj_step() per outer iteration.
    wall_now = time.perf_counter() - wall0
    # How far simulation should be (real-time pacing)
    target_sim = wall_now
    # Step until we catch up (but don't spiral)
    max_catchup_steps = 4000
    steps = 0
    while sim_time + dt <= min(target_sim, T_TOTAL) and steps < max_catchup_steps:
        # disturbance force (only early)
        data.xfrc_applied[:, :] = 0.0
        if sim_time < KICK_TIME:
            data.xfrc_applied[ball_body_id, 0] = KICK_FORCE  # +X force

        mujoco.mj_step(model, data)
        sim_time += dt
        steps += 1

        # ---- read accel each physics step (cheap) ----
        acc_site = np.array(data.sensordata[acc_adr:acc_adr+3], dtype=float)
        R = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        acc_world = R @ acc_site

        # Low-pass
        acc_filt = (1.0 - ACC_LP_ALPHA) * acc_filt + ACC_LP_ALPHA * acc_world

        # bias collection phase (before kick ends and before control starts)
        if sim_time < BIAS_TIME:
            bias_acc_sum += acc_filt
            bias_n += 1

        # ---- control at LOG_HZ rate (not every dt) ----
        if sim_time >= next_log:
            next_log += 1.0 / LOG_HZ

            # compute bias once available
            if bias_n > 10:
                bias = bias_acc_sum / bias_n
            else:
                bias = np.zeros(3)

            ax = float(acc_filt[0] - bias[0])
            ay = float(acc_filt[1] - bias[1])

            # current yaw
            yaw_cur_deg = rad2deg(float(data.qpos[qpos_adr]))
            yaw_cur_deg = wrap_deg(yaw_cur_deg)

            # PID after warmup
            if sim_time > WARMUP:
                meas = ax if USE_AXIS == 0 else ay
                e = meas  # target 0

                # deadband to reduce chatter
                if abs(e) < DEADBAND:
                    e = 0.0

                # derivative on measurement
                de = (e - prev_e) / dt if dt > 0 else 0.0
                prev_e = e

                # candidate integral
                i_term_candidate = clamp(i_term + e * (1.0 / LOG_HZ), -I_CLAMP, I_CLAMP)

                # PID output
                u = KP * e + KI * i_term_candidate + KD * de

                # map u -> yaw target (deg)
                target_deg = HOME_DEG - rad2deg(u)

                # clamp target to actuator range
                target_deg = clamp(target_deg, cr_lo_deg, cr_hi_deg)

                # anti-windup: only accept integral update if not saturated "pushing further"
                # if command is at limit and error wants more in same direction, freeze integral
                pushing_high = (command_deg >= cr_hi_deg - 1e-6) and (target_deg >= command_deg) and (e > 0)
                pushing_low  = (command_deg <= cr_lo_deg + 1e-6) and (target_deg <= command_deg) and (e < 0)
                if not (pushing_high or pushing_low):
                    i_term = i_term_candidate

                # rate limit command
                max_step = MAX_YAW_RATE_DEG_S * (1.0 / LOG_HZ)
                err_cmd = target_deg - command_deg
                if abs(err_cmd) > max_step:
                    command_deg += max_step if err_cmd > 0 else -max_step
                else:
                    command_deg = target_deg

                command_deg = clamp(command_deg, cr_lo_deg, cr_hi_deg)

            set_actuator_deg(command_deg)

            # log
            ts.append(sim_time)
            ax_log.append(ax)
            ay_log.append(ay)
            yaw_cmd_log.append(command_deg)
            yaw_cur_log.append(yaw_cur_deg)

    # Render at RENDER_HZ (not every step)
    if sim_time >= next_render:
        next_render += 1.0 / RENDER_HZ
        viewer.sync()

    # if we are ahead of real time (rare), sleep a bit
    wall_now2 = time.perf_counter() - wall0
    if sim_time > wall_now2 + 0.01:
        time.sleep(0.005)

# keep window open
while viewer.is_running():
    viewer.sync()
    time.sleep(0.01)

viewer.close()

# ---------------- plots ----------------
ts = np.array(ts)
ax_log = np.array(ax_log)
ay_log = np.array(ay_log)
yaw_cmd_log = np.array(yaw_cmd_log)
yaw_cur_log = np.array(yaw_cur_log)

plt.figure()
plt.plot(ts, ax_log, label="a_world_x (filtered, bias removed)")
plt.plot(ts, ay_log, label="a_world_y (filtered, bias removed)")
plt.axhline(0.0, linewidth=1)
plt.xlabel("time (s)")
plt.ylabel("acc (m/s^2)")
plt.title("Ball IMU acceleration (world) under PID (force kick disturbance)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(ts, yaw_cur_log, label="yaw current (deg)")
plt.plot(ts, yaw_cmd_log, label="yaw command (deg)", linestyle="--")
plt.axhline(0.0, linewidth=1)
plt.xlabel("time (s)")
plt.ylabel("yaw (deg)")
plt.title("Yaw response (clamped + anti-windup + rate limited)")
plt.legend()
plt.grid(True)
plt.show()