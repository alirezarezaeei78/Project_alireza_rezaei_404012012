"""Build scientific diagrams and measured-result figures for the report."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import mujoco
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    svg=OUT/(name+'.svg')
    fig.savefig(svg,bbox_inches='tight');plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')

fig,ax=plt.subplots(figsize=(12,4));ax.set_xlim(0,12);ax.set_ylim(0,4);ax.axis('off')
nodes=[(0.2,2.4,2.5,'Sensor frame','Accelerometer + gyro'),(3.4,2.4,2.5,'World frame','Rotation + filter'),(6.6,2.4,2.5,'Controller','PID / Double DQN'),(9.4,.6,2.4,'Position servo','Yaw target in radians'),(4.8,.3,3.2,'MuJoCo dynamics','Yaw arm + spherical pendulum')]
for x,y,w,a,b in nodes:
    ax.add_patch(FancyBboxPatch((x,y),w,1.05,boxstyle='round,pad=.04',facecolor='#EEF6F5',edgecolor='#007F86',lw=1.5))
    ax.text(x+w/2,y+.7,a,ha='center',weight='bold',color='#173344');ax.text(x+w/2,y+.27,b,ha='center',fontsize=10,color='#526776')
for start,end in [((2.7,2.95),(3.4,2.95)),((5.9,2.95),(6.6,2.95)),((9.1,2.95),(10.6,1.7)),((9.4,1.1),(8.05,.8)),((4.8,.8),(1.45,2.35))]:
    ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1.8,'color':'#173344'})
ax.text(6,.02,'Schematic control loop; one yaw actuator, no direct torque action',ha='center',fontsize=10,color='#526776')
save(fig,'control-loop')

fig,ax=plt.subplots(figsize=(11,3.2));ax.axis('off')
equations=[r'$\mathbf{a}_W=R_{W\leftarrow S}\mathbf{a}_S,\quad \bar{\mathbf{a}}_k=(1-\alpha)\bar{\mathbf{a}}_{k-1}+\alpha\mathbf{a}_{W,k}$',
 r'$\psi^{cmd}_{k+1}=\mathrm{clip}\left(\psi^{cmd}_k+\mathrm{clip}(\psi^{target}_k-\psi^{cmd}_k,\,-\dot\psi_{max}\Delta t_c,\,\dot\psi_{max}\Delta t_c),\,\psi_{min},\psi_{max}\right)$',
 r'$a^*=\arg\max_a Q_{\theta}(s^{\prime},a),\quad y=r+\gamma(1-d_{terminal})Q_{\theta^-}(s^{\prime},a^*)$']
for y,t in zip([.78,.46,.14],equations):ax.text(.02,y,t,fontsize=16,color='#173344',transform=ax.transAxes)
save(fig,'equations')

rows=list(csv.DictReader((ROOT/'results/verified/summary.csv').open()))
directions=['positive_x','negative_x','positive_y','negative_y']
fig,ax=plt.subplots(figsize=(10,4.7));x=np.arange(4)
for offset,name,color,hatch in [(-.18,'hold_zero','#667788','//'),(.18,'pid_x','#007F86','')]:
    values=[float(next(r['rms_axy_second_half'] for r in rows if r['direction']==d and r['controller']==name)) for d in directions]
    bars=ax.bar(x+offset,values,.34,label=name,color=color,hatch=hatch)
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{value:.2f}',ha='center',fontsize=10)
ax.set_xticks(x,['+X','-X','+Y','-Y']);ax.set_ylabel('Horizontal acceleration RMS, 6-12 s (m/s²)')
ax.set_xlabel('60 N disturbance direction; duration 0.15 s');ax.set_ylim(0,160);ax.legend();ax.grid(axis='y',alpha=.2)
save(fig,'benchmark-rms')

model=mujoco.MjModel.from_xml_path(str(ROOT/'models/rope_arm.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
model.vis.global_.offwidth=1000
model.vis.global_.offheight=720
try:
    with mujoco.Renderer(model,height=720,width=1000) as renderer:
        camera=mujoco.MjvCamera();camera.lookat[:]=[.3,0,1.45];camera.distance=4.9;camera.azimuth=135;camera.elevation=-17
        renderer.update_scene(data,camera=camera)
        Image.fromarray(renderer.render()).save(OUT/'simulation-model.png')
    print('Rendered actual MJCF scene')
except Exception as e:
    print('Offscreen render unavailable:',type(e).__name__)
print('Built control loop, equations and measured RMS plot')
