# Technical analysis: yaw-actuated suspended-ball control

Prepared 8 October 2026 from the supplied project archive, complete Python/MJCF source, original 15-page Persian report, seven historical plots and inherited model weights. Local execution adds evidence; the original work remains intact.

## Problem and model

The fixed stand carries a yaw-actuated horizontal arm. A 0.96 m rigid segment hangs from a ball joint at the arm tip and carries a 0.50 kg ball. One yaw actuator influences a pendulum with three rotational degrees of freedom. This is an underactuated nonlinear simulation, with passive damping already present in the joints. A reduction in oscillation can therefore occur without a learned controller.

The arm capsule is 1.00 m long and 2.00 kg; the attachment offset is 1.05 m. Gravity is `(0,0,-9.81)` m/s². Physics uses `implicitfast` and a 0.0005 s step. The position actuator has gain 350, damping ratio 1 and a +/-300 force limit. Its command is a target yaw, not a directly applied torque.

The rigid segment cannot bend, stretch or go slack. There is no flexible-cable dynamics or hardware evidence. The 60 N, 0.15 s force pulse corresponds to a nominal 9 N s impulse, although the resulting constrained motion is not equivalent to an unconstrained velocity change.

## Measurement and PID

MuJoCo accelerometer and gyro channels are expressed in their site frame. The code rotates them into the world frame using `site_xmat`. The maintained filter applies `filtered += 0.15 * (raw - filtered)` once per physics step. Reading the current observation does not update the filter.

Undisturbed warm-up lasts 0.25 s. Its final filtered acceleration is saved as the baseline before the force pulse starts. The local stationary test gives horizontal baseline components near zero; the vertical accelerometer channel retains the gravity-related reading. The original report's vertical sign convention should not be used without checking the actual sensor output.

The PID uses the inherited gains `Kp=0.35`, `Ki=0.04`, `Kd=0.08`, a 0.5 m/s² deadband and an integral clamp of 30. It maps measured world-X acceleration to a yaw target through a negative sign and a radians-to-degrees conversion. Differentiation and integration use the actual 0.005 s control interval. Saturation-aware integration stops accumulating in the direction that drives the output farther outside its range. A shared rate cap and position bound act afterward.

These gains were not retuned during this analysis. Correct timing does not make an arbitrary acceleration-feedback law stabilizing. The influence of yaw motion depends on arm orientation and pendulum motion, and one input does not independently regulate both horizontal axes.

## Double DQN and reward

The observation has eight entries: three filtered world accelerometer channels, three world gyro channels, normalized actual yaw and normalized command rate. Command rate is derived from target changes; it is not measured joint angular velocity. The included gyro channels supply rotational measurements. This observation is not proven to be a sufficient Markov state.

The two discrete actions request +/-1.5 degrees relative to the previous target, with a 240-degree/s cap. Ten physics frames per action give a 0.005 s decision interval and a maximum realized target increment of 1.2 degrees. No explicit hold action exists.

The MLP architecture is `8-256-256-2`, with ReLU after each hidden layer. Action selection for the bootstrap uses the online network; evaluation uses the target network, which makes this **Double DQN** despite the historical filename using DQN. The target is `reward + gamma * (1 - terminated) * Q_target(next_state, argmax Q_online(next_state))`. The loss is Smooth L1 and the gradient norm is clipped at 10.

| Hyperparameter | Value |
| --- | --- |
| Original training budget | 1,500 episodes |
| Maximum decision steps | 800, or 4 s after warm-up |
| Discount | 0.99 |
| Adam learning rate | 0.0003 |
| Batch size / replay capacity | 256 / 200,000 |
| Learning starts | 2,000 collected transitions |
| Target synchronization | Every 1,000 transitions |
| Epsilon schedule | 1.00 to 0.05 over 120,000 transitions |

Reward uses `-tanh(horizontal_acceleration / 10)`, a yaw penalty of `-0.05 * abs(yaw_deg) / 90` and a target-change penalty of `-0.02 * abs(delta_command_deg) / 1.5`. After the kick ends, acceleration below 3 m/s² earns +1 per decision. Forty consecutive qualifying decisions produce a +200 success bonus and termination. A time limit produces truncation and does not suppress the bootstrap target.

The criterion is only a short acceleration proxy: 40 decisions equal 0.20 s. It does not prove stable equilibrium or guarantee that the response remains settled. Full-horizon benchmark runs continue after any threshold window and log tilt as an additional diagnostic. Reward shaping and the lack of a hold action deserve separate ablations.

## Source audit and changes

| Finding in supplied artifact | Effect | Maintained implementation |
| --- | --- | --- |
| XML actuator range is `-170 170` while the hinge range is specified in degrees | Compiled target range is +/-170 radians; original RL yaw normalization divides by about 9,740 degrees | Explicit +/-2.967059728 radians matches the +/-170-degree hinge |
| PID derivative divides by physics `dt` at a 200 Hz update | Derivative contribution is scaled about ten times too strongly | Shared 5 ms control interval |
| Bias collection overlaps the initial kick | The supposed rest baseline includes disturbance response | Undisturbed warm-up before the kick |
| RL reads and filters the same endpoint again for reward | Observation and reward can use different filtered values from the same physics state | One filter update per physics step; observation is read-only |
| Success and the 800-step limit both set `done` | Time-limited samples incorrectly lose their bootstrap term | Separate `terminated` and `truncated` |
| Success counter can run before the disturbance ends | Early success can reflect the timing rule rather than post-kick settling | Count after kick completion |
| Four-axis UI expects four actuators, but the XML has one | That archived UI is incompatible with the supplied model | Explicit one-yaw viewer command |
| Relative paths and import-time viewer startup in historical scripts | Depends on working directory and opens windows on import | Repository-relative model path and guarded commands |
| Oversized legacy MJCF constraint allocation | Memory allocation failed during the initial local benchmark | Explicit 64 MB arena; subsequent eight runs had no warnings |
| Training function does not save the supplied checkpoint or raw logs | Exact training provenance cannot be established | New training saves weights, configuration and CSV logs; original checkpoint labeled historical |

The original PID also used a 180-degree/s target cap, while original RL used 240 degrees/s. The corrected benchmark uses a common 240-degree/s cap. Corrections change the observation/dynamics/protocol, so original weights are not a fair comparator for the maintained PID.

## Fresh matched benchmark

Executed in MuJoCo 3.5.0 on Windows with Python 3.12.9. Each run begins at the model's rest configuration, has 0.25 s undisturbed warm-up, receives a 60 N pulse for 0.15 s, and continues for 12 s after force onset. Directions are +X, -X, +Y and -Y. Each direction compares `hold_zero` with `pid_x`. Both use the same model, control cadence, filter cadence, target bounds and rate cap. This is eight deterministic scenarios with no repeated-seed uncertainty estimate.

| Direction | Hold RMS, 6-12 s | PID RMS, 6-12 s | Hold final tilt | PID final tilt |
| --- | ---: | ---: | ---: | ---: |
| +X | 1.04 | 82.62 | 0.59° | 54.33° |
| -X | 1.04 | 90.45 | 0.59° | 131.20° |
| +Y | 1.19 | 140.67 | 0.15° | 63.55° |
| -Y | 1.19 | 1.00 | 0.15° | 1.76° |

Acceleration units are m/s². The PID reaches the +/-170-degree command limit in the +/-X cases. Its smaller acceleration RMS in -Y does not imply a universal improvement; its final tilt is larger than the hold-zero result there. All cases include large pendulum rotations near 180 degrees, so small-angle controller reasoning is insufficient for this disturbance. All eight runs produced zero simulator warnings, which demonstrates numerical execution rather than physical correctness or control stability.

The first qualifying 0.2 s acceleration windows are also saved in the summary, but they are not reported as permanent settling times. The hold-zero condition shows why passive damping must be included as a baseline. The results motivate controller redesign and retuning, rather than a claim that learning has already solved the task.

## Inherited checkpoint and historical plots

The inherited `best_dqn_yaw.pt` contains six tensors compatible with the 8-256-256-2 network. It was loaded with PyTorch's `weights_only=True`. No optimizer state, seed/configuration metadata or training history is embedded.

A diagnostic replay uses the original Python environment and original control/sensor logic, with only the XML arena allocation reduced to 64 MB to make the archived model practical to load. That replay reaches 800 decisions at simulated time 4.025 s, returns approximately -784.07 total reward and ends with horizontal acceleration magnitude 48.70 m/s². The stability counter is zero. The final `done` reflects the historical time limit, not success. No simulator warning occurred in this single replay. It is neither a replication of the original training nor a corrected-environment PID/RL comparison.

The original figures contain training reward, episode length, final acceleration and evaluation traces. Their raw numerical data and exact association with the inherited weights are absent. They remain visual historical evidence; values were not digitized into new numerical claims. The original RL viewer also resets simulation time between episodes while appending to one plotting list, so a plotted time axis may contain discontinuities.

## Verification and reproducibility

Seven regressions passed locally: radians/limits, rate limiting, pure observation reads and repeatable reset, success/truncation timing, PID derivative interval, both anti-windup directions, Double DQN action selection/terminal masking and replay capacity wrap. Several assertions are grouped within one test method.

The 30-episode training-path check collected 6,000 transitions at seed 1 and a 200-step limit. Twenty-one episode records contain an optimizer loss after the learning threshold. This checks replay sampling, gradient updates, target synchronization and checkpoint writing. It does not establish reward convergence, robustness or a learned policy's usefulness. The smoke-test weights are kept local and not promoted as a validated release.

Use the README's editable-install commands. Benchmark manifests store actual versions, settings, model/data hashes and the measurement protocol. `docs/original/manifest.json` records hashes of all 15 preserved original files. New local experiments should write to ignored local directories; checked-in verification results should be replaced only deliberately with an updated analysis.

## Development roadmap

1. Define a physical target: pendulum angles, angular velocity, acceptable yaw motion and persistent settling after the disturbance.
2. Establish hold-zero and classical controller baselines under matched scenarios. Analyze local yaw-to-pendulum control authority before tuning. Separate development and held-out force/direction/initial-angle sets.
3. Reduce disturbance scale for local linear controller experiments, then evaluate the nonlinear regime explicitly. Tune gains using the development set and log every protocol change.
4. Retrain Double DQN under the corrected units/filtering and save complete metadata. Consider a hold action and an angle/velocity observation; evaluate reward components through ablation.
5. Evaluate multiple seeds and held-out disturbances over a fixed horizon. Report angular/acceleration metrics, constraint violations, persistent settling and uncertainty, alongside compute budgets.
6. Investigate model fidelity, contact behavior, sensor noise and flexible cable modeling before any hardware transfer. No real-system validation has been performed.

## References

- [MuJoCo XML reference, version 3.3.1](https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html): sensor frames, actuator ranges, position servos and explicit arena allocation.
- [MuJoCo modeling documentation](https://mujoco.readthedocs.io/en/stable/modeling.html): kinematic frames and model compilation.
- [Van Hasselt, Guez and Silver, Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461): online-action selection with target-network evaluation.
- Supplied original source, MJCF, report, checkpoint and image archive; all numerical results in the fresh benchmark come from saved local runs, not from external benchmark claims.

No hardware trial, full 1,500-episode retraining or independent learned-policy evaluation was completed during this review.
