"""Replay the inherited checkpoint using archived observation/action logic.

Only the MJCF arena allocation is reduced; control units and historical bugs
remain intact. This is an artifact diagnostic, not a fair controller benchmark.
"""
from pathlib import Path
import importlib.util
import sys
import csv
import hashlib
import json
import torch
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'docs/original/source'
out=ROOT/'results/legacy-replay'
out.mkdir(parents=True,exist_ok=True)
xml=source/'rope_arm.xml'
runtime=out/'runtime-model.xml'
runtime.write_text(xml.read_text().replace('<size njmax="12000" nconmax="3000"/>','<size memory="64M"/>'))
spec=importlib.util.spec_from_file_location('archived_rl',source/'rl_dqn_ball_stabilize.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
env=module.BallStabilizeEnv(str(runtime))
policy=module.QNet(env.obs_dim,env.n_actions)
checkpoint=ROOT/'artifacts/checkpoints/best_dqn_yaw.pt'
weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
policy.load_state_dict(weights);policy.eval();torch.set_num_threads(1)
state=env.reset();rows=[];total=0.
for step in range(module.MAX_EP_STEPS):
    with torch.no_grad():action=int(policy(torch.from_numpy(state).unsqueeze(0)).argmax().item())
    state,reward,done,info=env.step(action);total+=reward
    rows.append(dict(time_s=env.sim_time,action=action,reward=reward,**info))
    if done:break
with (out/'trajectory.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
payload=dict(kind='archived checkpoint diagnostic, not corrected-environment evaluation',
    steps=len(rows),episode_sim_time_s=env.sim_time,total_reward=total,final_axy=info['axy'],
    done=done,stable_count=info['stable_count'],max_abs_command_deg=max(abs(r['cmd_deg']) for r in rows),
    original_ctrlrange_rad=env.model.actuator_ctrlrange.tolist(),
    warning_count=int(sum(w.number for w in env.data.warning)),
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    source_xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
    runtime_xml_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
    observation_dim=env.obs_dim,architecture='8-256-256-2',
    changes='Arena allocation only: legacy njmax/nconmax replaced by memory=64M. Archived Python is unchanged.',
    limitation='Training provenance is absent. Deterministic single episode, historical filtering/units/truncation retained. No performance comparison with corrected PID.')
(out/'inspection.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
