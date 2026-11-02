# Validation scripts

Regression checks for the ZiMPedance controller (`mpc_params['type'] = 'zimpedance'`) and for the
PACC and two-robot simulations. Run them in the `zimpedance` conda env (`installation/setup_env.sh`),
from the repo root.

| Script | Checks |
|---|---|
| `regression_zimpedance.py [--tasks a,b] [--jobs N] [--out DIR]` | **Main regression.** Model selection and arm parameters; the Pinocchio–MuJoCo check below; then every task preset of `simulation/zimpedance_tasks.py`, run headless in parallel through `zimpedance_sim.py --task <name> --export`, checked for physical behaviour (table below). Exit code 1 on any failure |
| `check_pinocchio_vs_mujoco.py <xml> [--n 200] [--meshdir DIR] [--out res.json]` | Model loading, joint structure, `eef` frame attachment, forward kinematics, mass matrix (with armature) and gravity torques of the Pinocchio model the controller builds, against MuJoCo on the same MJCF |
| `regression_collaborative.py [--skip-closed-loop] [--jobs N] [--out DIR]` | Standalone PACC model, follower robot model, two-robot composition, payload presets and follower guidance; then the closed-loop two-robot scenarios through `collaborative_carrying_sim.py --export` (last section below) |

```bash
conda activate zimpedance
python simulation/validation/regression_zimpedance.py            # ~6 min on 16 cores
python simulation/validation/regression_zimpedance.py --tasks y_tracking,circular_tracking
G=gym_quadruped/robot_model/aliengo_leader
python simulation/validation/check_pinocchio_vs_mujoco.py $G/new_urdf_export.xml
# The hardware XML has no meshes next to it; borrow the sim robot's:
python simulation/validation/check_pinocchio_vs_mujoco.py \
    quadruped_pympc/controllers/zimpedance/new_urdf_export.xml --meshdir $G
```

## `regression_zimpedance.py`: checks and baseline values

The EEF presets run on `aliengo_follower` (arm at the front) with arm k [3, 25, 5], d [0.05, 1.5,
0.5] (`EEF_TASK_ARM`); the others on `aliengo_leader` (arm at the rear) with the MJCF impedance
k [5, 8, 5], d [0.5, 1.5, 0.5] (`simulation/zimpedance_tasks.py`). Every task also checks: the run completes,
states stay finite, every MPC status is 0 or 2, no fall (base z > 0.25 m and |roll|, |pitch| <
0.8 rad after 1 s), and the robot and prediction-model arm impedance are the preset's. Thresholds
are about twice the observed values: they flag a change in behaviour, they are not the
controller's limits. The runs are deterministic; the observed values below reproduce exactly.

| Task | Check | Threshold | Observed |
|---|---|---|---|
| model (each robot) | XML = sim robot's; arm joints; arm k, d, rest; total mass | exact | `<robot>/new_urdf_export.xml`; [5, 8, 5], [0.5, 1.5, 0.5], 0; leader 27.946 kg, follower 29.346 kg |
| scenes (each robot) | leader `flat`, `rough`, `pyramid`, `ramp`, `slope`, `stairs`, `suspend_stairs`; follower `flat`, `rough`, `ramp`, `stairs`: each loads the same actuated robot (12 `<role>/*` motors on their joints, `imu`, a keyframe, arm joints, `eef`; robot physics arrays identical) | exact | all |
| kinematics (each robot) | eef parent; positions; M; gravity torques | `arm_link_3`; 1e-9; 1e-9; 1e-2 N m | ok; 1e-15 m; 1e-15; 5.3e-3 (leader), 5.7e-3 N m (follower) (g = 9.81 vs 9.806) |
| locomotion | mean v_x, ω_z, v_x per segment (fwd 0.1, turn 0.2, back −0.1) | [0.08, 0.12], [0.17, 0.23], [−0.12, −0.08] | 0.098, 0.210, −0.098 |
| | max \|roll, pitch\| while walking; speed 2 s after stop | 0.1 rad; 0.02 m/s | 0.036 rad; 0.004 m/s |
| rough_terrain | x travel; v_x; max \|roll, pitch\| | > 3.0 m; [0.08, 0.12]; 0.3 rad | 3.27 m; 0.099; 0.143 rad |
| y_tracking (follower) | y RMS, max error; measured peak-to-peak | 10, 15 mm; > 180 mm | 1.0, 1.5 mm; 197 mm |
| yz_reaching (follower) | worst settled \|error\| over 10 holds | 15 mm | y 2.0, z 6.0 mm |
| circular_tracking (follower, radius 0.1 m) | y RMS / max; z RMS / max; peak-to-peak y, z | 10 / 18 mm; 16 / 25 mm; > 170, 160 mm | 5.1 / 9.1 mm; 8.1 / 11.7 mm; 188, 178 mm |
| figure_eight_tracking (follower, joint-1 d 0.02, `sim_eef_eight_tracking`) | y RMS / max; z RMS / max; peak-to-peak y, z; base y peak-to-peak | 8 / 12 mm; 20 / 32 mm; > 200, 120 mm; < 30 mm | 2.0 / 2.8 mm; 11.7 / 19.6 mm; 235, 133 mm; 13 mm |
| moving_target_reaching (follower) | y RMS / max; z RMS / max (to the target's current position); peak-to-peak y, z | 17 / 45 mm; 25 / 43 mm; > 80, 120 mm | 9.1 / 24.5 mm; 10.7 / 19.2 mm; 97, 132 mm |
| spatial_reaching (follower, `sim_eef_eight_tracking`) | worst settled \|error\| over the 4 fixed targets; RMS to the moving target's current position | 15 mm; 40 mm | 6.9 mm; y 18.4, z 9.3 mm |
| xy_reaching (follower) | each x–y target reached within 30 mm, in order; last target held over the last 1 s; max \|roll, pitch\| | yes; 30 mm; 0.4 rad | at 6.2, 11.6, 18.3 s; 16.4 mm; 0.173 rad |
| disturbance_rejection | impulses; max \|roll, pitch\|; slowest recovery | 8; 0.15 rad; 1.5 s | 8 × 31.6 N; 0.077 rad; 0.85 s |
| disturbance_sinusoidal | v_x; max \|roll, pitch\| | [0.07, 0.12]; 0.25 rad | 0.092; 0.114 rad |
| pull_release | last 1 s: max \|roll, pitch\|, height error | 0.02 rad, 10 mm | 0.1 mrad, 0 mm |

Wall time: about 6 min for all 12 tasks with the default jobs on 16 cores (rough_terrain is the
longest, 40 s simulated). The first task of each robot builds its acados solver if needed (about
1 min); each robot model has its own solver directory under `c_generated_code/`.

## `regression_collaborative.py`: standalone PACC and two-robot carrying

```bash
python simulation/validation/regression_collaborative.py                     # ~3.5 min on 16 threads, 137 checks
python simulation/validation/regression_collaborative.py --skip-closed-loop  # model checks only, ~7 s
```

Model checks run in-process; the closed-loop scenarios run `collaborative_carrying_sim.py
--export` in parallel. Observed values, deterministic:

| Section | Check | Threshold | Observed |
|---|---|---|---|
| standalone PACC | compiles; ropes carry the payload | nq 27, 2 tendons; tension = weight within 2 % | 98.0 N for 98.1 N |
| | every passive joint moves (lateral impulse) | peak > 0.02 rad | 0.27, 0.06, 0.34 rad |
| | stiffness ×2; payload 10 → 20 kg; springref q2 = 0.3 | q2 at rest < 0.6×; > 1.5×; +0.2 rad | 0.467 → 0.217; → 0.939; → 0.792 rad |
| | damping ×0.25 / ×1 / ×3 | joint-1 peak decreasing | 0.393 > 0.271 > 0.146 rad |
| | one-arm arrangements; 60 s at rest | finite, payload above ground; \|q̇\| < 1e-3 | ok; 4e-10 rad/s |
| aliengo_follower | 12 follower/* motors; arm axes z, y, y; sensors and keyframe | exact | ok |
| | front-mounted arm; impedance | base x > 0.2 m, unrotated; k, d, springref | x 0.285 m; EEF (0.5615, 0, 0.0885) m in trunk; [5, 8, 5], [0.5, 1.5, 0.5], 0 |
| | scenes flat, flat_green, ramp, stairs, rough | 12 motors, 2 sensors, keyframe | all |
| | Pinocchio vs MuJoCo | 1e-9 | 4e-16 m, M 1e-15 |
| two-robot composition | layout; namespaces | nq 51, nu 24, free joints at 0/22/44 | ok |
| | independent arm parameters | as requested per robot | leader [6, 9, 4], follower [3.5, 8.47, 2.75] |
| | default payload | 2 kg, two tendons k 15 N/m, range [0, 0.15] m | ok |
| | initialization | tendons < 0.15 m, spring deadband [0, spawn length], no non-ground contact | 0.119 / 0.115 m, none |
| | tendons hold the payload (bases held) | tensions ≈ weight ±25 %; +20 N push: payload moves < 0.3 m, both arms deflect, tendons ≤ 0.16 m (soft limit) | 10.4 + 10.1 N (19.6 N); 0.20 m; 0.32 / 0.50 rad; 0.155 m |
| | rigid-bar variant: initial violation; +20 N push | < 1e-6 m; 20 ± 2 N through the connections | 2.6e-9 m; 20.0 N |
| | direct elastic variant | one EEF–EEF tendon, k 10 N/m, range [0, 0.4] m | ok |
| payload and impedance presets | `model` preset = robot MJCFs | exact | [5, 8, 5], [0.5, 1.5, 0.5], 0 |
| | per preset: coupling, mass, k/d/springref/masses compiled; static equilibrium with bases held (inside joint limits, link forces ≥ 0.9 × weight, payload z > 0.1 m, settled) | exact; as stated | light 10.4 + 10.2 N / 19.6 N, z 0.269; medium 26.0 + 25.4 / 49.1, 0.232 (`original_elastic`); heavy 34.6 + 34.2 / 68.7, 0.399 m |
| | springref and q0 per robot; wrench-estimator spring torque vs MuJoCo (random q, springref ≠ 0) | exact; 1e-9 N m | ok; 0 |
| | standalone sim with `original_dual` and 7 kg; JSON parameter file, unknown keys rejected | exact | ok |
| follower guidance | levels, dead zones, filter, input convention | exact | ok |
| closed loop (default: 2 kg on elastic tendons) | stand: attitude, follower command, initial link tension | 0.05 rad; 0; < 80 N | 0.021 rad; 0; 42 N (ropes go taut) |
| | straight: x velocity 8–24 s, spacing range, guidance active | [0.08, 0.12]; < 0.2 m; > 90 % | 0.100 / 0.101 m/s; 1.81–1.89 m; 100 % |
| | stop: last 4 s command, speed, spacing range, arm q̇ | 1e-3; 0.01 m/s; 0.01 m; 0.05 rad/s | 0; 0.0004 m/s; 0 m; 0.001 rad/s |
| | turn / turn_right (0.1 m/s, ±0.1 rad/s, 60 s), steady state 40–60 s: follower yaw rate vs leader; follower yaw command mean, share of its yaw, sign | ≥ 0.7×; same sign, > 0.02 rad/s, > 40 %, never wrong | left 0.066 / 0.081, +0.045, 65 %; right −0.079 / −0.081, −0.061, 74 % |
| | turn_stop (left 4–44 s, stop): command after 46 s; relative-heading drift and spacing range over the last 6 s | < 1e-3; < 0.03 rad/s, < 0.1 m | 0; 0.019 rad/s, 0.047 m |
| | impedance (follower: PACC paper Table II) | as straight, values applied | 0.100 / 0.101 m/s |
| | rigid_bar straight | as straight | 0.099 / 0.099 m/s; 1.48–1.58 m |
| | elastic (EEF–EEF tendon) straight | x velocity [0.07, 0.12] | 0.085 / 0.085 m/s |
| | `medium_5kg` / `heavy_7kg` (experimental) stand: max \|roll, pitch\| < 2× observed, no follower command; straight: x velocity [0.08, 0.16], spacing range < 0.3 m, attitude < 1.6× observed | as stated | stand 0.033 / 0.047 rad; straight 0.119 / 0.116 and 0.134 / 0.133 m/s, 0.235 / 0.137 rad |
| all closed loop | no MuJoCo instability reset (`unstable_at`) | none | none |
| all closed loop | coupling holds; solver status; finite | rigid bar 0.28 ± 0.005 m; tendons: EEF distance within the taut-tendon bound, payload above 0.1 m | e.g. default 0.51–0.66 m (< 0.705), payload ≥ 0.218 m |
