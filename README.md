# ZiMPedance

Code for **ZiMPedance: Impedance-Aware ZMP Modeling and Control for Payload Carrying with Quadruped
Robots** (Giovanni B. Dessy, Lorenzo Amatucci, Claudio Semini, Victor Barasuol).

## Overview

A quadruped carrying a payload through a passive spring-based arm: the arm's dynamics store and
release energy during locomotion and move the ZMP. This repository contains the MuJoCo simulation
of the paper and the controller it proposes:

- **ZiMPedance MPC** (`mpc_params['type'] = 'zimpedance'`, the paper's ARMPC): an MPC on the single
  rigid body dynamics extended with the passive-arm joint states, their springs and dampers and the
  base-arm inertial coupling, with a ZMP constraint. It is written with
  [acados](https://github.com/acados/acados) and [Pinocchio](https://github.com/stack-of-tasks/pinocchio).
- **Single-robot tasks** on an Aliengo with a 3-DoF passive arm: locomotion on flat and rough
  terrain, disturbances on the arm's end effector, and end-effector tracking and reaching with the arm.
- **Standalone PACC model**: the two passive arms on fixed blocks, sharing a payload.
- **Two-robot collaborative carrying**: a leader and a follower that carry a payload between their
  arms; the follower is guided by its arm deflection. Both robots run the PACC collaborative MPC
  (`'collaborative'`), not ZiMPedance.
- **Passive arm hardware**: printable STLs of the arm and the Arduino sketch of its joint encoders
  ([hardware/passive_arm](hardware/passive_arm)).

The controller framework is derived from [Quadruped-PyMPC](https://github.com/iit-DLSLab/Quadruped-PyMPC)
and the simulation environment from [gym-quadruped](https://github.com/iit-DLSLab/gym-quadruped);
both are reduced here to what this code uses.

## Dependencies

- Linux (tested on Ubuntu 24.04, x86-64), conda or mamba, and a C compiler
- Python 3.12, [MuJoCo](https://mujoco.org/) 3.4.0, Pinocchio 4.0 (must be >= 4.0), casadi 3.7.2,
  liecasadi, gymnasium, numpy, scipy, matplotlib: see [installation/environment.yml](installation/environment.yml)
- [acados](https://github.com/acados/acados) v0.4.0, as a git submodule, built by the setup script

## Installation

```bash
git clone --recurse-submodules https://github.com/iit-DLSLab/ZiMPedance.git
cd ZiMPedance
bash installation/setup_env.sh          # --lock for the exact tested versions
conda activate zimpedance
```

The setup script creates the `zimpedance` conda environment, builds acados in
`quadruped_pympc/acados` and sets `PYTHONPATH`, `LD_LIBRARY_PATH` and `ACADOS_SOURCE_DIR` when the
environment is activated. Nothing is pip-installed. The first run of each controller and robot
generates and compiles its acados solver (about 1 min); later runs reuse it.

## How to run

Single robot, ZiMPedance MPC (`simulation/zimpedance_sim.py`):

```bash
python simulation/zimpedance_sim.py --task figure_eight_tracking
python simulation/zimpedance_sim.py --task locomotion --headless --export run.npz
python simulation/zimpedance_sim.py                    # interactive: arrow keys set the velocity command
```

| Task | Robot | What it does |
|---|---|---|
| `locomotion` | leader | crawl forward, turn in place, backward, stop |
| `rough_terrain` | leader | crawl forward over tilted boxes up to 0.1 m |
| `disturbance_rejection` | leader | crawl under eight 30 N impulses on the end effector (EEF) |
| `disturbance_sinusoidal` | leader | crawl under a 15 N, 1 Hz force on the EEF |
| `pull_release` | leader | standing, the EEF is pulled sideways and released four times |
| `y_tracking` | follower | standing, lateral EEF sine, 0.1 m at 0.4 Hz (paper Fig. 6) |
| `yz_reaching` | follower | y-z EEF targets, held 2 s each |
| `circular_tracking` | follower | EEF circle in the y-z plane, radius 0.1 m |
| `figure_eight_tracking` | follower | EEF figure eight, drawn by the arm while the base stays nearly still |
| `moving_target_reaching` | follower | EEF follows a target moving at 0.05 m/s |
| `spatial_reaching` | follower | four fixed EEF targets, then a moving one |
| `xy_reaching` | follower | walks to put the EEF on x-y targets up to 0.6 m away |

The leader has the arm at the rear, the follower at the front. The end-effector tasks use a soft
arm (joint 1 nearly undamped) so the arm motion is visible. Presets, weights and arm impedance:
[simulation/zimpedance_tasks.py](simulation/zimpedance_tasks.py). Options: `--duration S`,
`--scene rough|stairs|ramp|...`, `--headless`, `--export run.npz`.

Standalone PACC and two-robot carrying:

```bash
python simulation/pacc_standalone_sim.py                         # passive arms on blocks, shared payload
python simulation/collaborative_carrying_sim.py --scenario straight
    # scenarios: stand, straight, stop, turn, turn_right, turn_slow, turn_stop, showcase
    # --payload-preset light_2kg (default) | medium_5kg | heavy_7kg (experimental)
```

Regression checks (about 6 and 3.5 min on 16 cores; see
[simulation/validation/README.md](simulation/validation/README.md)):

```bash
python simulation/validation/regression_zimpedance.py
python simulation/validation/regression_collaborative.py
```

## Repository layout

| Path | Content |
|---|---|
| `quadruped_pympc/config.py` | controller and simulation parameters, ZiMPedance weight profiles |
| `quadruped_pympc/controllers/zimpedance/` | ZiMPedance MPC (acados OCP and Pinocchio model) |
| `quadruped_pympc/controllers/collaborative/` | PACC collaborative MPC (two-robot carrying) |
| `quadruped_pympc/` (rest) | gait generator, foothold and swing control, passive-arm wrench estimate |
| `gym_quadruped/` | MuJoCo environments and robot models: `aliengo_leader`, `aliengo_follower`, `aliengo_two_robots` |
| `simulation/` | the simulations above, their task presets and the regression checks |
| `hardware/passive_arm/` | arm STLs (mm) and encoder sketch |
| `installation/` | conda environment and setup script |

## Citing this work

If you use this code, please cite:

```bibtex
@article{dessy2026zimpedance,
  author  = {Dessy, Giovanni B. and Amatucci, Lorenzo and Semini, Claudio and Barasuol, Victor},
  title   = {{ZiMPedance}: Impedance-Aware {ZMP} Modeling and Control for Payload Carrying with Quadruped Robots},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026}
}
```

## License

BSD 3-Clause, see [LICENSE](LICENSE).

## Maintainer

This repository is maintained by [Giovanni B. Dessy](https://github.com/Giovadess) (giovanni.dessy@outlook.com).
