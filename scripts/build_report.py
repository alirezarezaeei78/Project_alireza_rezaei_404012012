"""Build the illustrated English technical report from checked-in evidence."""
from pathlib import Path
from xml.sax.saxutils import escape
import csv,json
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,Image,PageBreak,Preformatted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/Project_Analysis_Report.pdf'
fonts=Path('C:/Windows/Fonts')
if (fonts/'segoeui.ttf').exists():
    pdfmetrics.registerFont(TTFont('Body',str(fonts/'segoeui.ttf')))
    pdfmetrics.registerFont(TTFont('Bold',str(fonts/'segoeuib.ttf')))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold')
    body,bold='Body','Bold'
else:body,bold='Helvetica','Helvetica-Bold'
NAVY=colors.HexColor('#173344');TEAL=colors.HexColor('#007F86');GRAY=colors.HexColor('#526776');PALE=colors.HexColor('#EEF6F5')
styles={
 'body':ParagraphStyle('body',fontName=body,fontSize=9.8,leading=14.4,textColor=NAVY,spaceAfter=9),
 'title':ParagraphStyle('title',fontName=bold,fontSize=25,leading=31,textColor=NAVY,spaceAfter=15),
 'cover':ParagraphStyle('cover',fontName=bold,fontSize=32,leading=39,textColor=NAVY,spaceAfter=14),
 'heading':ParagraphStyle('heading',fontName=bold,fontSize=13,leading=18,textColor=TEAL,spaceBefore=8,spaceAfter=7),
 'small':ParagraphStyle('small',fontName=body,fontSize=8.2,leading=11.8,textColor=GRAY,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName=body,fontSize=8.5,leading=12,textColor=NAVY),
 'th':ParagraphStyle('th',fontName=bold,fontSize=8.4,leading=12,textColor=colors.white),
 'code':ParagraphStyle('code',fontName='Courier',fontSize=8,leading=12,textColor=NAVY,spaceAfter=10)}
def P(text,style='body'):return Paragraph(text,styles[style])
def heading(title):return [P('ALIREZA REZAEI / TECHNICAL PROJECT ANALYSIS','small'),P(title,'title')]
def table(headers,rows,widths):
    t=Table([[P(escape(h),'th') for h in headers]]+[[P(escape(str(v)),'cell') for v in row] for row in rows],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    return t
def img(path,width):
    from PIL import Image as PILImage
    with PILImage.open(path) as i:w,h=i.size
    return Image(str(path),width=width,height=width*h/w)
def section(title,text):return [P(title,'heading'),P(text)]
fig=ROOT/'docs/figures'
rows=list(csv.DictReader((ROOT/'results/verified/summary.csv').open()))
legacy=json.loads((ROOT/'results/legacy-replay/inspection.json').read_text())
smoke=json.loads((ROOT/'results/training-smoke/verification.json').read_text())
story=[P('CONTROL SYSTEMS / REINFORCEMENT LEARNING / REPRODUCIBILITY','small'),P('Yaw-actuated\npendulum control'.replace('\n','<br/>'),'cover'),P('PID and Double DQN in MuJoCo','heading'),P('An illustrated analysis of Project_alireza_rezaei_404012012<br/>Alireza Rezaei | 8 October 2026'),img(fig/'simulation-model.png',475),Spacer(1,8),table(['Source review','Fresh simulation','Learning path'],[['Original code, model, 15-page report, plots and weights','8 matched runs; 12 seconds each; zero simulator warnings','7 regressions passed; 30 short training episodes completed']],[170,170,171]),Spacer(1,12),P('The project connects a physical simulation with classical feedback and value-based learning. This review makes the workflow reproducible and separates historical artifacts from measured evidence. The current gains and inherited policy need further validation before any stabilization claim.'),PageBreak()]

story+=heading('1 / The mechanism and the question')
story+=[P('Can a yaw command reduce motion of a ball hanging from a rotating arm after a force pulse? One actuator influences an unactuated spherical pendulum, making this a nonlinear, underactuated control problem.'),img(fig/'control-loop.png',511),Spacer(1,10),table(['Model element','Specified value / interpretation'],[['Kinematic chain','world -> stand -> green_link -> rope_rigid'],['Arm / attachment','1.00 m capsule, 2.00 kg; pendulum attachment at 1.05 m'],['Suspended segment / ball','0.96 m rigid segment; 0.50 kg ball'],['Degrees of freedom','1 yaw hinge + 3 rotations at the ball joint; 1 actuator'],['Gravity / integration','(0, 0, -9.81) m/s²; implicitfast; 0.0005 s physics step'],['Position servo','Gain 350; damping ratio 1; force range +/-300'],['Yaw command','Target position, not direct torque; limits +/-170 degrees']],[150,361]),Spacer(1,10),P('The word rope refers to a single rigid capsule. It cannot bend, stretch or become slack. Passive joint damping is present, so holding the yaw target at zero is an essential baseline. The simulation does not establish real cable or hardware behavior.'),P('The diagram is a schematic. The cover image is an offscreen render of the actual maintained MJCF model. Model specifications come directly from the supplied XML.','small'),PageBreak()]

story+=heading('2 / Measurements and PID feedback')
story+=section('From local sensors to a bounded command','The site accelerometer and gyro are transformed to world coordinates. The maintained filter updates once per physics step; observation reads are pure. Undisturbed warm-up lasts 0.25 s and establishes the horizontal baseline before the kick begins.')
story+=[img(fig/'equations.png',511),P('Equations: frame rotation and filtering; bounded/rate-limited yaw target; Double DQN bootstrap. Filter coefficient alpha = 0.15. The common control interval is 0.005 s.','small')]
story+=section('The inherited acceleration-feedback law','PID regulates world-X acceleration with gains Kp = 0.35, Ki = 0.04 and Kd = 0.08. The output maps to a yaw target through a negative sign and radians-to-degrees conversion. A 0.5 m/s² deadband, integral clamp of 30 and saturation-aware integration constrain the command.')
story+=[table(['Implementation point','Why it matters'],[['Derivative timing','Use the 5 ms controller period. The archived code divides by the 0.5 ms physics step.'],['Calibration','A rest baseline must precede the force pulse; the historical calibration overlapped it.'],['Control authority','One yaw input cannot independently regulate both horizontal axes. The influence depends on configuration.'],['Interpretation','Correct timing and units do not make the inherited gains a validated stabilizing controller.']],[140,371]),Spacer(1,10),P('The vertical sensor channel retains the gravity-related reading. Its sign should be established from an actual rest test instead of assumed from the original report. The benchmark uses horizontal acceleration and separately logs pendulum tilt.'),PageBreak()]

story+=heading('3 / What makes this Double DQN')
story+=[P('The online network selects the next action and the target network evaluates that action. This is Double DQN, despite the historical filename using DQN. The network is an 8-256-256-2 MLP with ReLU hidden layers.'),table(['Component','Implementation'],[['Observation','World acceleration (3), world gyro (3), normalized actual yaw and normalized target-command rate'],['Actions','Left/right requests of 1.5 degrees; rate cap realizes at most 1.2 degrees per 5 ms step'],['Original budget','1,500 episodes; maximum 800 decisions per episode'],['Discount / optimizer','Gamma 0.99; Adam learning rate 0.0003'],['Replay / batches','Capacity 200,000; batch size 256; learning begins at 2,000 transitions'],['Target / exploration','Synchronize every 1,000 transitions; epsilon 1.00 to 0.05 over 120,000 transitions'],['Loss / clipping','Smooth L1 loss; gradient norm capped at 10']],[140,371])]
story+=section('Reward and episode ending','Reward penalizes horizontal acceleration via -tanh(a_xy / 10), yaw magnitude via -0.05 |yaw| / 90 and target changes via -0.02 |delta yaw| / 1.5. After the pulse, acceleration below 3 m/s² earns +1. Forty consecutive qualifying decisions earn +200 and success termination.')
story+=section('The modeling limits','A 0.20 s acceleration window is a proxy, not a stability proof. Target-command rate is not measured joint velocity; the gyro supplies rotational measurements. The observation has not been proven to be a sufficient Markov state, and the discrete action set has no explicit hold action.')
story+=[P('A true terminal transition masks bootstrapping. An 800-step time limit is truncation and retains it. Changing units, filtering and normalization changes the inherited policy inputs; a fresh trained policy is needed for comparison in the maintained environment.','small'),PageBreak()]

story+=heading('4 / Audit findings and maintenance')
story+=[P('The original source and report remain unmodified in the archive. The maintained code addresses concrete reproduction issues; the new protocol is not presented as identical to the historical experiment.'),table(['Supplied artifact finding','Effect','Maintained change'],[['Target range -170 to 170','Compiles as radians, unlike the degree-specified hinge','Explicit +/-2.967059728 rad'],['PID divides derivative by physics dt','About 10x derivative scale at a 200 Hz update','Use actual 0.005 s interval'],['Calibration overlaps kick','Disturbed signal becomes the rest baseline','Calibrate before disturbance'],['RL filters endpoint twice','Observation and reward use different filter state','Filter once per physics step'],['One done flag for success/time limit','Time-limit target incorrectly loses bootstrap','Separate terminal and truncated'],['Success checked during kick','Can reward early timing rather than post-kick settling','Count after kick completion'],['Four-actuator UI vs one-actuator XML','Historical UI is incompatible with model','One-yaw viewer command'],['Large legacy constraint allocation','Initial benchmark hit a memory allocation failure','Explicit 64 MB arena'],['No checkpoint/log writer in training','Supplied weights lack training provenance','Save metadata, weights and CSVs'],['Working-directory / launch assumptions','Scripts may open viewers during import','Portable model path; explicit CLI']],[157,183,171]),Spacer(1,10),P('The original PID rate cap was 180 degrees/s and RL used 240 degrees/s. The new benchmark uses a common 240-degree/s cap. The original checkpoint diagnostic retains historical control logic and only changes memory allocation.','small'),PageBreak()]

story+=heading('5 / The fresh matched benchmark')
story+=[P('Eight deterministic runs: two conditions for each of +X, -X, +Y and -Y. After 0.25 s undisturbed warm-up, apply 60 N for 0.15 s and observe 12 s after force onset. Both conditions share the same model, filter/control cadence, limits and rate cap. PID uses inherited gains with corrected timing.'),img(fig/'benchmark-rms.png',475),P('Measured from saved trajectories; RMS window = 6-12 s. These are fixed scenarios with no repeated-trial uncertainty estimate. Lower acceleration does not by itself imply stable equilibrium.','small')]
data=[]
for d,label in [('positive_x','+X'),('negative_x','-X'),('positive_y','+Y'),('negative_y','-Y')]:
    a=next(r for r in rows if r['direction']==d and r['controller']=='hold_zero')
    b=next(r for r in rows if r['direction']==d and r['controller']=='pid_x')
    data.append([label,f"{float(a['rms_axy_second_half']):.2f}",f"{float(b['rms_axy_second_half']):.2f}",f"{float(a['final_tilt_deg']):.2f} / {float(b['final_tilt_deg']):.2f}"])
story+=[table(['Kick','Hold RMS','PID RMS','Final tilt: hold / PID'],data,[60,100,100,251]),Spacer(1,10),P('The inherited PID gives higher late acceleration in three of four directions. In -Y its RMS is slightly lower, but its final tilt is larger. All runs include nearly full pendulum rotations, so this is outside a small-angle regime. No trained DQN is ranked against corrected PID here.'),P('Acceleration: m/s². Tilt: degrees from downward vertical. All eight runs reported zero simulator warnings. Numerical execution is distinct from physical validity and successful control.','small'),PageBreak()]

story+=heading('6 / Read the trajectories, not one score')
story+=[img(ROOT/'results/verified/acceleration.png',495),P('Horizontal acceleration for all four force directions. Shading marks the 0.15 s pulse. Dashed gray holds yaw at zero; teal uses inherited PID gains with corrected timing. Full-horizon logging continues after any short success-proxy window.','small'),img(ROOT/'results/verified/tilt-and-command.png',470),P('+X case: pendulum tilt and yaw target. The PID reaches the target limit and sustains motion. The first threshold window is not called a permanent settling time. Raw trajectories and exact settings are saved with the repository.','small'),PageBreak()]

story+=heading('7 / Historical evidence and checkpoint')
story+=[P('Seven supplied plots show training and evaluation traces. Their raw numerical data and exact link to the checkpoint are missing. They remain historical visual evidence; this analysis does not digitize them into new performance claims.'),img(ROOT/'docs/original/figures/RL/RL_1.png',350),P('Historical episode-reward plot from the supplied archive. It is not a fresh training result from the maintained code.','small'),table(['Checkpoint diagnostic','Observed result'],[['Network tensors','Compatible with 8-256-256-2; loaded with weights_only=True'],['Replay horizon',f"{legacy['steps']} decisions; {legacy['episode_sim_time_s']:.3f} s simulated time"],['Total reward / final acceleration',f"{legacy['total_reward']:.2f} / {legacy['final_axy']:.2f} m/s²"],['Success / numerical warnings',f"Stability counter {legacy['stable_count']}; warning count {legacy['warning_count']}"],['Interpretation','Ended at the archived time limit; success was not achieved in this diagnostic']],[163,348]),Spacer(1,10),P('The replay uses original Python logic. Only the model arena is reduced to 64 MB. It does not reproduce the original training or provide a fair comparison with the corrected PID. Checkpoint training configuration and optimizer history are absent.'),P('The original RL viewer appends across resets while simulation time restarts; a combined evaluation trace can therefore contain time discontinuities.','small'),PageBreak()]

story+=heading('8 / Reproduction and verification')
story+=[P('Verified locally with Python 3.12.9, MuJoCo 3.5.0, NumPy 1.26.4, Matplotlib 3.10.0 and PyTorch 2.4.0 on Windows. Install from a checkout; the maintained package uses the adjacent models directory.'),Preformatted('python -m pip install -e .\npython -m unittest discover -s tests -v\npendulum-control benchmark --out results/local/benchmark\npendulum-control train --episodes 30 --max-steps 200\npython scripts/inspect_original.py',styles['code']),table(['Verification','Coverage / evidence'],[['7 passing regression tests','Compiled radians and rate limits; pure observations and repeatable reset; episode endings; PID timing and anti-windup; Double DQN targets; replay wrap'],['8 full benchmark runs','Raw CSVs, summary metrics, plots, model hash, actual configuration and version manifest'],['30 short training episodes',f"{smoke['transitions']:,} transitions; {smoke['episodes_with_optimizer_updates']} episode records with optimizer loss; weights/metadata written locally"],['Original archive integrity','15 original files preserved with SHA-256 hashes'],['Visualization','Actual MJCF offscreen scene; measured plots; editable SVG diagrams']],[150,361])]
story+=section('What these checks establish','The simulator, controller interfaces, replay sampling, gradient updates, target synchronization and checkpoint-writing paths execute. The short training check does not establish convergence or policy quality. Its weights are not promoted as a validated controller.')
story+=section('Evidence boundaries','The original 1,500-episode training was not rerun. There is no independent learned-policy evaluation or hardware trial. Future runs should use local output folders so the checked-in evidence remains traceable. Core version pins are supplied, rather than a complete transitive environment lock.')
story+=[PageBreak()]

story+=heading('9 / A credible next experiment')
story+=[table(['Step','Next deliverable'],[['1. Define the target','Angle, angular velocity, yaw motion and persistent settling criteria, beyond acceleration alone'],['2. Establish baselines','Hold-zero and interpretable feedback controllers under matched conditions; analyze local yaw control authority'],['3. Separate development and testing','Tune on one disturbance/initial-state set; reserve forces and directions for evaluation'],['4. Retrain the policy','Corrected environment, full metadata, multiple seeds; investigate a hold action and angle/velocity observations'],['5. Report generalization','Fixed horizons, repeated trials, uncertainty, constraint violations and matched compute budgets'],['6. Review physical fidelity','Contact, damping, sensor noise and flexible cable assumptions before any hardware transfer']],[148,363]),Spacer(1,10),P('The strongest present outcome is a transparent, executable research prototype with an auditable explanation. The measured failures are useful design evidence. A broad claim that PID or RL solves stabilization would require the experiments above.'),P('References and source evidence','heading')]
for label,url in [('MuJoCo XML reference: sensors, actuator limits and arena allocation','https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html'),('MuJoCo modeling: kinematic frames and compilation','https://mujoco.readthedocs.io/en/stable/modeling.html'),('Van Hasselt, Guez and Silver: Double Q-learning','https://arxiv.org/abs/1509.06461')]:
    story.append(P(f'<a href="{url}" color="#007F86">{escape(label)}</a>','small'))
story+=[P('Project-specific specifications, findings and measurements are supported by the original source/report, archived checkpoint diagnostic and fresh checked-in CSVs. The detailed Markdown analysis includes file-level evidence and reproduction instructions.','small'),P('Original project: Alireza Rezaei. Original academic artifacts remain intact. No additional software reuse license is granted by this report.','small')]

def footer(c,doc):
    c.setStrokeColor(TEAL);c.line(42,38,553,38);c.setFont(body,8);c.setFillColor(GRAY)
    c.drawString(42,24,'MuJoCo pendulum control / source review and measured evidence');c.drawRightString(553,24,str(doc.page))

doc=SimpleDocTemplate(str(OUT),pagesize=(595.28,841.89),leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=53,
    title='MuJoCo Pendulum Control - Technical Analysis',author='Alireza Rezaei')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('Created',OUT)
