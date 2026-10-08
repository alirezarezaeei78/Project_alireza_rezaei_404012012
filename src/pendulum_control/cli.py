"""Headless benchmarking and explicit opt-in training/viewing."""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
from pathlib import Path
from dataclasses import asdict
import numpy as np
import mujoco
from .environment import BallStabilizeEnv, Config, ROOT, DEFAULT_MODEL
from .controllers import PIDController


def write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def benchmark(out, seconds=12.):
    if seconds <= .15:
        raise ValueError("Benchmark duration must exceed the disturbance")
    out.mkdir(parents=True,exist_ok=True)
    config=Config(max_steps=round(seconds/.005))
    directions={"positive_x":(1,0,0),"negative_x":(-1,0,0),"positive_y":(0,1,0),"negative_y":(0,-1,0)}
    summary=[]
    series={}
    for direction, vector in directions.items():
        for name in ("hold_zero","pid_x"):
            env=BallStabilizeEnv(config=config,direction=vector)
            controller=PIDController(env.control_dt)
            rows=[]
            for _ in range(config.max_steps):
                ax=float(env.acc[0]-env.bias[0])
                target=controller.target(ax) if name=="pid_x" else 0.
                _,_,_,_,info=env.step_target(target,stop_on_success=False)
                rows.append(info)
            write_csv(out/f"{direction}_{name}.csv",rows)
            late=[r for r in rows if r['time_s']>=seconds/2]
            count=0;settling=None
            for row in rows:
                count=count+1 if row['time_s']>=config.kick_duration_s and row['axy']<config.stable_acc else 0
                if count>=config.stable_steps and settling is None:
                    settling=row['time_s']
            summary.append(dict(direction=direction,controller=name,
                rms_axy_second_half=float(np.sqrt(np.mean([r['axy']**2 for r in late]))),
                peak_axy=float(max(r['axy'] for r in rows)),
                max_tilt_deg=float(max(r['tilt_deg'] for r in rows)),
                final_tilt_deg=rows[-1]['tilt_deg'],
                first_threshold_window_end_s=settling,
                max_abs_command_deg=max(abs(r['command_deg']) for r in rows),
                simulator_warning_count=int(sum(w.number for w in env.data.warning))))
            series[(direction,name)]=rows
    write_csv(out/'summary.csv',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    palette={'hold_zero':'#667788','pid_x':'#007F86'}
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for ax,(direction,_) in zip(axes.flat,directions.items()):
        for name in ('hold_zero','pid_x'):
            r=series[(direction,name)]
            ax.plot([x['time_s'] for x in r],[x['axy'] for x in r],color=palette[name],
                    linestyle='--' if name=='hold_zero' else '-',label=name)
        ax.axvspan(0,.15,color='#DAA337',alpha=.15)
        ax.set_title(direction.replace('_',' '));ax.set_ylabel('Horizontal acceleration (m/s²)')
        ax.set_xlabel('Time after kick onset (s)');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'acceleration.png',dpi=180);fig.savefig(out/'acceleration.svg');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for name in ('hold_zero','pid_x'):
        r=series[('positive_x',name)]
        axes[0].plot([x['time_s'] for x in r],[x['tilt_deg'] for x in r],color=palette[name],label=name,linestyle='--' if name=='hold_zero' else '-')
        axes[1].plot([x['time_s'] for x in r],[x['command_deg'] for x in r],color=palette[name],label=name,linestyle='--' if name=='hold_zero' else '-')
    axes[0].set_ylabel('Pendulum tilt (degrees)');axes[1].set_ylabel('Yaw target (degrees)');axes[1].set_xlabel('Time after kick onset (s)')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'tilt-and-command.png',dpi=180);fig.savefig(out/'tilt-and-command.svg');plt.close(fig)
    artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))}
    import torch
    manifest=dict(kind='fresh deterministic simulation benchmark',python=platform.python_version(),
        versions=dict(mujoco=mujoco.__version__,numpy=np.__version__,matplotlib=matplotlib.__version__,torch=torch.__version__),
        platform=platform.platform(),duration_s=seconds,configuration=env.configuration(),directions=directions,
        model_sha256=hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest(),csv_sha256=artifacts,
        protocol='Same model, disturbance, control/filter cadence and limits; PID uses original gains with corrected timing. Eight fixed runs, no fitted policy or statistical replication.',
        limitation='A low-acceleration window is a proxy, not a stability proof. No trained DQN is ranked in this corrected-environment benchmark.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for svg in out.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(summary,indent=2))


def train(out,episodes,seed,max_steps):
    import torch
    from .dqn import QNet,ReplayBuffer,Transition,double_dqn_target
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    rng=random.Random(seed);torch.set_num_threads(1)
    config=Config(max_steps=max_steps)
    env=BallStabilizeEnv(config=config)
    policy,target=QNet(),QNet();target.load_state_dict(policy.state_dict());target.eval()
    opt=torch.optim.Adam(policy.parameters(),lr=3e-4)
    replay=ReplayBuffer(seed=seed);global_step=0;rows=[]
    for ep in range(episodes):
        state=env.reset();total=0.;loss_value=None
        for t in range(max_steps):
            eps=max(.05,1-.95*global_step/120000)
            action=rng.randrange(2) if rng.random()<eps else int(policy(torch.from_numpy(state).unsqueeze(0)).argmax().item())
            next_state,reward,terminated,truncated,info=env.step(action)
            replay.push(Transition(state.copy(),action,reward,next_state.copy(),terminated))
            state=next_state;total+=reward;global_step+=1
            if len(replay)>=2000:
                s,a,r,ns,d=replay.sample(256)
                qs=policy(torch.from_numpy(s)).gather(1,torch.from_numpy(a).long().unsqueeze(1)).squeeze(1)
                target_q=double_dqn_target(torch.from_numpy(r),torch.from_numpy(d),policy(torch.from_numpy(ns)),target(torch.from_numpy(ns)))
                loss=torch.nn.functional.smooth_l1_loss(qs,target_q)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),10.);opt.step()
                loss_value=float(loss.detach())
            if global_step%1000==0:target.load_state_dict(policy.state_dict())
            if terminated or truncated:break
        rows.append(dict(episode=ep+1,reward=total,steps=t+1,terminated=terminated,truncated=truncated,final_axy=info['axy'],last_loss=loss_value))
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'training.csv',rows)
    metadata=dict(seed=seed,episodes=episodes,config=asdict(config),architecture='8-256-256-2',
        model_sha256=hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest(),environment='corrected-v1',
        hyperparameters=dict(gamma=.99,learning_rate=3e-4,batch_size=256,replay_capacity=200000,learn_start=2000,target_update_steps=1000,epsilon_decay_steps=120000),
        note='Training is not an independent evaluation; validate on held-out disturbances before performance claims.')
    torch.save(dict(state_dict=policy.state_dict(),metadata=metadata),out/'dqn.pt')
    (out/'training-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Saved {episodes} training episodes at {out}; updates begin after 2000 transitions.')


def view(target_deg):
    import time
    import mujoco.viewer
    env=BallStabilizeEnv(config=Config(kick_force=0,max_steps=10**9))
    with mujoco.viewer.launch_passive(env.model,env.data) as viewer:
        while viewer.is_running():
            env.step_target(target_deg,stop_on_success=False)
            viewer.sync();time.sleep(env.control_dt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    b=sub.add_parser('benchmark');b.add_argument('--out',type=Path,default=ROOT/'results/local/benchmark');b.add_argument('--seconds',type=float,default=12.)
    t=sub.add_parser('train');t.add_argument('--out',type=Path,default=ROOT/'artifacts/local/training');t.add_argument('--episodes',type=int,default=1500);t.add_argument('--seed',type=int,default=1);t.add_argument('--max-steps',type=int,default=800)
    v=sub.add_parser('view');v.add_argument('--target-deg',type=float,default=0.)
    args=parser.parse_args()
    if args.command=='benchmark':benchmark(args.out,args.seconds)
    elif args.command=='train':
        if args.episodes<1 or args.max_steps<1:parser.error('episodes and max-steps must be positive')
        train(args.out,args.episodes,args.seed,args.max_steps)
    else:view(args.target_deg)


if __name__=='__main__':main()
