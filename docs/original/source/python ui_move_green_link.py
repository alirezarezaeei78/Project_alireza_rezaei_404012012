import tkinter as tk
from tkinter import ttk
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "rope_arm.xml"

# Load model + data
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Convenience: control indices (must match actuator order in XML)
IDX_X = 0
IDX_Y = 1
IDX_Z = 2
IDX_YAW = 3

# Default targets
target = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "yaw": 0.0,  # degrees
}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def set_ctrl_from_target():
    # Read ctrlranges directly from model (safer than hardcoding)
    # ctrlrange is shape (nu, 2)
    cr = model.actuator_ctrlrange

    data.ctrl[IDX_X] = clamp(target["x"], cr[IDX_X, 0], cr[IDX_X, 1])
    data.ctrl[IDX_Y] = clamp(target["y"], cr[IDX_Y, 0], cr[IDX_Y, 1])
    data.ctrl[IDX_Z] = clamp(target["z"], cr[IDX_Z, 0], cr[IDX_Z, 1])
    data.ctrl[IDX_YAW] = clamp(target["yaw"], cr[IDX_YAW, 0], cr[IDX_YAW, 1])

def on_slider(name, val):
    target[name] = float(val)
    set_ctrl_from_target()

def on_apply_entries():
    try:
        target["x"] = float(ent_x.get())
        target["y"] = float(ent_y.get())
        target["z"] = float(ent_z.get())
        target["yaw"] = float(ent_yaw.get())
    except ValueError:
        return
    # Update sliders to match
    s_x.set(target["x"])
    s_y.set(target["y"])
    s_z.set(target["z"])
    s_yaw.set(target["yaw"])
    set_ctrl_from_target()

# --- Tk UI ---
root = tk.Tk()
root.title("MuJoCo Green Link Position Controller")

frm = ttk.Frame(root, padding=12)
frm.grid(row=0, column=0, sticky="nsew")

def add_slider(row, label, name, lo, hi):
    ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w")
    s = ttk.Scale(frm, from_=lo, to=hi, orient="horizontal",
                  command=lambda v: on_slider(name, v))
    s.grid(row=row, column=1, sticky="ew", padx=8)
    vlabel = ttk.Label(frm, width=10)
    vlabel.grid(row=row, column=2, sticky="e")

    def update_label(*_):
        vlabel.config(text=f"{s.get():.3f}")
    s.configure(command=lambda v: (on_slider(name, v), update_label()))
    s.set(target[name])
    update_label()
    return s

# Get actuator ctrl ranges (must exist)
cr = model.actuator_ctrlrange
if model.nu < 4:
    raise RuntimeError("Your XML must have 4 actuators: pos_gx,pos_gy,pos_gz,pos_yaw")

frm.columnconfigure(1, weight=1)

s_x = add_slider(0, "X", "x", cr[IDX_X, 0], cr[IDX_X, 1])
s_y = add_slider(1, "Y", "y", cr[IDX_Y, 0], cr[IDX_Y, 1])
s_z = add_slider(2, "Z", "z", cr[IDX_Z, 0], cr[IDX_Z, 1])
s_yaw = add_slider(3, "Yaw (deg)", "yaw", cr[IDX_YAW, 0], cr[IDX_YAW, 1])

# Entry boxes
row = 4
ttk.Separator(frm).grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
row += 1

ttk.Label(frm, text="Type values and Apply:").grid(row=row, column=0, sticky="w")
row += 1

ent_x = ttk.Entry(frm, width=10); ent_x.insert(0, str(target["x"]))
ent_y = ttk.Entry(frm, width=10); ent_y.insert(0, str(target["y"]))
ent_z = ttk.Entry(frm, width=10); ent_z.insert(0, str(target["z"]))
ent_yaw = ttk.Entry(frm, width=10); ent_yaw.insert(0, str(target["yaw"]))

ttk.Label(frm, text="X").grid(row=row, column=0, sticky="w"); ent_x.grid(row=row, column=1, sticky="w")
row += 1
ttk.Label(frm, text="Y").grid(row=row, column=0, sticky="w"); ent_y.grid(row=row, column=1, sticky="w")
row += 1
ttk.Label(frm, text="Z").grid(row=row, column=0, sticky="w"); ent_z.grid(row=row, column=1, sticky="w")
row += 1
ttk.Label(frm, text="Yaw").grid(row=row, column=0, sticky="w"); ent_yaw.grid(row=row, column=1, sticky="w")
row += 1

ttk.Button(frm, text="Apply", command=on_apply_entries).grid(row=row, column=1, sticky="w", pady=6)

# Initialize ctrl
set_ctrl_from_target()

# --- Run viewer in a separate thread-like loop using passive viewer ---
viewer = mujoco.viewer.launch_passive(model, data)

def step_sim():
    # step a few times per UI tick
    for _ in range(5):
        mujoco.mj_step(model, data)
    viewer.sync()
    root.after(10, step_sim)  # ~100 Hz UI tick

root.after(10, step_sim)
root.mainloop()
viewer.close()