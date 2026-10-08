# ui_yaw_green_link_realtime_uiux.py
import time
import math
import tkinter as tk
from tkinter import ttk

import mujoco
import mujoco.viewer

XML_PATH = "rope_arm.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

if model.nu < 1:
    raise RuntimeError("No actuators found. Ensure XML has <position name='yaw_pos' ...> in <actuator>.")

IDX_YAW = 0
dt = float(model.opt.timestep)

# MuJoCo uses radians internally
cr_lo_rad = float(model.actuator_ctrlrange[IDX_YAW, 0])
cr_hi_rad = float(model.actuator_ctrlrange[IDX_YAW, 1])
cr_lo_deg = math.degrees(cr_lo_rad)
cr_hi_deg = math.degrees(cr_hi_rad)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def wrap_deg(a):
    while a > 180:
        a -= 360
    while a < -180:
        a += 360
    return a

jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint0")
if jid < 0:
    raise RuntimeError("Joint 'joint0' not found in XML.")
qpos_adr = int(model.jnt_qposadr[jid])

# State in degrees
target_yaw_deg = 0.0
command_yaw_deg = 0.0

# Auto-return timing
return_at = None
HOME_YAW_DEG = 0.0

# UI helper state
_suppress_ui = False

def set_actuator_deg(yaw_deg: float):
    yaw_rad = math.radians(yaw_deg)
    yaw_rad = clamp(yaw_rad, cr_lo_rad, cr_hi_rad)
    data.ctrl[IDX_YAW] = yaw_rad

# ---------------- UI ----------------
root = tk.Tk()
root.title("MuJoCo • Yaw Control")
root.geometry("720x420")
root.minsize(720, 420)

style = ttk.Style()
try:
    style.theme_use("vista")
except tk.TclError:
    pass

# Slightly nicer defaults
style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
style.configure("Sub.TLabel", font=("Segoe UI", 10))
style.configure("Card.TFrame", padding=12)
style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=(12, 8))
style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))

outer = ttk.Frame(root, padding=14)
outer.pack(fill="both", expand=True)

# Header
header = ttk.Frame(outer)
header.pack(fill="x")

ttk.Label(header, text="Green Stick Yaw Controller", style="Title.TLabel").pack(side="left", anchor="w")

status_var = tk.StringVar(value="IDLE")
status_lbl = ttk.Label(header, textvariable=status_var, style="Status.TLabel")
status_lbl.pack(side="right")

info = ttk.Label(
    outer,
    text=f"Range: {cr_lo_deg:.0f}° to {cr_hi_deg:.0f}°   •   timestep: {dt}s   •   Keys: ←/→ step, Space=Home",
    style="Sub.TLabel"
)
info.pack(anchor="w", pady=(6, 10))

# Cards row
row = ttk.Frame(outer)
row.pack(fill="both", expand=True)

left = ttk.Frame(row, style="Card.TFrame")
left.pack(side="left", fill="both", expand=True, padx=(0, 8))

right = ttk.Frame(row, style="Card.TFrame")
right.pack(side="right", fill="both", expand=True, padx=(8, 0))

# --- Left card: target controls ---
ttk.Label(left, text="Target & Actions", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))

readout = ttk.Frame(left)
readout.pack(fill="x", pady=(0, 8))

lbl_target = ttk.Label(readout, text="Target: 0.0°", font=("Segoe UI", 11))
lbl_target.pack(side="left")
lbl_current = ttk.Label(readout, text="Current: 0.0°", font=("Segoe UI", 11))
lbl_current.pack(side="right")

# Progress bar (how close to target)
ttk.Label(left, text="Target progress", style="Sub.TLabel").pack(anchor="w")
progress = ttk.Progressbar(left, orient="horizontal", mode="determinate", maximum=100)
progress.pack(fill="x", pady=(4, 10))

# Slider
ttk.Label(left, text="Drag to set target", style="Sub.TLabel").pack(anchor="w")
yaw_slider = ttk.Scale(left, from_=cr_lo_deg, to=cr_hi_deg, orient="horizontal")
yaw_slider.pack(fill="x", pady=(4, 10))

# Entry + big buttons
entry_row = ttk.Frame(left)
entry_row.pack(fill="x", pady=(2, 8))

ttk.Label(entry_row, text="Yaw (deg):").pack(side="left")
yaw_entry = ttk.Entry(entry_row, width=10, font=("Segoe UI", 11))
yaw_entry.pack(side="left", padx=8)
yaw_entry.insert(0, "0")

btn_row = ttk.Frame(left)
btn_row.pack(fill="x", pady=(6, 8))

go_btn = ttk.Button(btn_row, text="GO", style="Big.TButton")
go_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

home_btn = ttk.Button(btn_row, text="RETURN NOW", style="Big.TButton")
home_btn.pack(side="right", fill="x", expand=True, padx=(8, 0))

# Small step buttons
steps = ttk.Frame(left)
steps.pack(fill="x", pady=(6, 0))
for txt, d in [("−15°", -15), ("−5°", -5), ("−1°", -1), ("+1°", 1), ("+5°", 5), ("+15°", 15)]:
    ttk.Button(steps, text=txt, command=lambda dd=d: None).pack(side="left", padx=3)

# --- Right card: motion params ---
ttk.Label(right, text="Motion Settings", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))

# Speed
speed_var = tk.DoubleVar(value=600.0)
speed_val = tk.StringVar(value="600")

speed_line = ttk.Frame(right)
speed_line.pack(fill="x", pady=(0, 8))
ttk.Label(speed_line, text="Move speed (deg/s):").pack(side="left")
ttk.Label(speed_line, textvariable=speed_val, font=("Segoe UI", 10, "bold")).pack(side="right")

speed_scale = ttk.Scale(right, from_=10, to=720, variable=speed_var, orient="horizontal")
speed_scale.pack(fill="x", pady=(0, 14))

# Hold time
hold_var = tk.DoubleVar(value=3.0)
hold_val = tk.StringVar(value="3.0")

hold_line = ttk.Frame(right)
hold_line.pack(fill="x", pady=(0, 8))
ttk.Label(hold_line, text="Hold before return (s):").pack(side="left")
ttk.Label(hold_line, textvariable=hold_val, font=("Segoe UI", 10, "bold")).pack(side="right")

hold_scale = ttk.Scale(right, from_=0.0, to=10.0, variable=hold_var, orient="horizontal")
hold_scale.pack(fill="x", pady=(0, 14))

# Tip box
tip = ttk.Label(
    right,
    text="Tip:\n• Set Hold=0 to disable auto-return.\n• Use GO to start the timer.\n• RETURN NOW goes to 0° immediately.",
    justify="left"
)
tip.pack(anchor="w", pady=(8, 0))

def update_labels(*_):
    speed_val.set(f"{speed_var.get():.0f}")
    hold_val.set(f"{hold_var.get():.1f}")

speed_var.trace_add("write", update_labels)
hold_var.trace_add("write", update_labels)
update_labels()

def set_target_deg(v: float, user_initiated: bool = True, arm_timer: bool = True):
    """
    user_initiated: updates UI + target
    arm_timer: if True, starts auto-return timer (GO action). Slider changes can optionally arm too.
    """
    global target_yaw_deg, return_at, _suppress_ui

    v = clamp(float(v), cr_lo_deg, cr_hi_deg)
    target_yaw_deg = v
    lbl_target.config(text=f"Target: {target_yaw_deg:.1f}°")

    # timer logic
    if arm_timer:
        h = float(hold_var.get())
        if h <= 0.0:
            return_at = None
        else:
            return_at = time.perf_counter() + h

    # sync UI widgets
    _suppress_ui = True
    yaw_slider.set(v)
    yaw_entry.delete(0, "end")
    yaw_entry.insert(0, f"{v:.1f}")
    _suppress_ui = False

def slider_changed(_=None):
    if _suppress_ui:
        return
    # slider updates the target, but doesn't start the timer until you press GO
    set_target_deg(float(yaw_slider.get()), user_initiated=True, arm_timer=False)

yaw_slider.configure(command=lambda v: slider_changed())

def go_action():
    # GO uses entry if valid, else slider; starts timer
    try:
        v = float(yaw_entry.get())
    except ValueError:
        v = float(yaw_slider.get())
    set_target_deg(v, user_initiated=True, arm_timer=True)

def return_now():
    global return_at
    return_at = None
    set_target_deg(HOME_YAW_DEG, user_initiated=True, arm_timer=False)

go_btn.configure(command=go_action)
home_btn.configure(command=return_now)

def step_buttons(d):
    set_target_deg(float(yaw_slider.get()) + d, user_initiated=True, arm_timer=False)

# wire step buttons properly (replace the dummy commands)
for child, (_, d) in zip(steps.winfo_children(), [("−15°",-15),("−5°",-5),("−1°",-1),("+1°",1),("+5°",5),("+15°",15)]):
    child.configure(command=lambda dd=d: step_buttons(dd))

# keyboard shortcuts
root.bind("<Left>",  lambda e: step_buttons(-1))
root.bind("<Right>", lambda e: step_buttons(1))
root.bind("<space>", lambda e: return_now())
root.bind("<Return>", lambda e: go_action())

# init state
set_target_deg(0.0, user_initiated=False, arm_timer=False)
command_yaw_deg = 0.0
set_actuator_deg(0.0)

# Viewer
viewer = mujoco.viewer.launch_passive(model, data)

last_wall = time.perf_counter()
accum = 0.0
MAX_STEPS_PER_TICK = 4000

def set_status(text: str):
    status_var.set(text)

def tick():
    global last_wall, accum, command_yaw_deg, return_at

    now = time.perf_counter()
    elapsed = now - last_wall
    last_wall = now
    accum += elapsed

    # auto-return check
    if return_at is not None and now >= return_at:
        return_at = None
        set_target_deg(HOME_YAW_DEG, user_initiated=False, arm_timer=False)

    # status text
    if abs(target_yaw_deg - HOME_YAW_DEG) < 0.5 and abs(command_yaw_deg - HOME_YAW_DEG) < 0.5:
        set_status("IDLE")
    elif return_at is None and abs(target_yaw_deg - HOME_YAW_DEG) < 0.5:
        set_status("RETURNING")
    elif return_at is not None and abs(target_yaw_deg - HOME_YAW_DEG) >= 0.5:
        set_status("HOLDING")
    else:
        set_status("MOVING")

    # ramp command toward target
    max_step_deg = float(speed_var.get()) * elapsed
    err = target_yaw_deg - command_yaw_deg
    if abs(err) <= max_step_deg:
        command_yaw_deg = target_yaw_deg
    else:
        command_yaw_deg += max_step_deg if err > 0 else -max_step_deg

    set_actuator_deg(command_yaw_deg)

    # physics stepping
    steps_n = int(accum / dt)
    if steps_n > MAX_STEPS_PER_TICK:
        steps_n = MAX_STEPS_PER_TICK
        accum = 0.0
    else:
        accum -= steps_n * dt

    for _ in range(steps_n):
        mujoco.mj_step(model, data)

    # read current (rad -> deg)
    cur_deg = math.degrees(float(data.qpos[qpos_adr]))
    cur_disp = wrap_deg(cur_deg)
    lbl_current.config(text=f"Current: {cur_disp:.1f}°")

    # progress: 100 when on target, 0 when far (scaled by range)
    span = max(1e-6, (cr_hi_deg - cr_lo_deg))
    dist = abs(target_yaw_deg - cur_deg)
    # normalize: dist=0 -> 100, dist >= span -> 0
    pct = 100.0 * max(0.0, 1.0 - min(dist / span, 1.0))
    progress["value"] = pct

    viewer.sync()
    root.after(10, tick)

root.after(10, tick)
root.mainloop()
viewer.close()