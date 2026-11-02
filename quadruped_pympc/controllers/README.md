# Controllers

Selected by `config.mpc_params['type']` and loaded in `quadruped_pympc/interfaces/srbd_controller_interface.py`
(single robot) or `srbd_controller_interface_collaborative.py` (collaborative wrapper).

| `type` | Directory | Role |
|---|---|---|
| `zimpedance` | `zimpedance/` | ZiMPedance MPC: single rigid body extended with the passive-arm joints, springs, dampers and base-arm coupling (Schur complement), ZMP constraint. Used by `simulation/zimpedance_sim.py` |
| `collaborative` | `collaborative/` | PACC collaborative MPC, with the passive-arm wrench in its model. Drives both robots in `simulation/collaborative_carrying_sim.py` |

Each controller writes its generated acados solver under `<directory>/c_generated_code/`
(git-ignored); the ZiMPedance MPC keeps one solver per robot model, `c_generated_code/<robot>/`. It
regenerates the solver when its model XML, its two OCP source files or `ACADOS_SOURCE_DIR` change
(`build_stamp.txt`); after changing an OCP option in `config.py` (horizon, constraints), delete the
directory to force a rebuild.

`zimpedance/new_urdf_export.xml` is the hardware robot (arm mounted at the front). In simulation
the controller uses the simulated robot's MJCF instead, through `mpc_params['armpc_model_xml']`.

**`zimpedance` configuration keys** (`config.mpc_params`):
- `armpc_model_xml`: the MJCF of the prediction model (default: the simulated robot's; `None`: the
  hardware robot above).
- `armpc_arm_params`: `'model'` (default) reads the arm stiffness, damping and `springref` and the
  total mass from that MJCF; `'hw_as_run'` uses the `collaborative_task_params` gains, the arm pose
  calibrated at 1 s as spring rest and a 25.523 kg reference mass, as on the hardware.
- `armpc_weight_profile`: a name in `config.armpc_weight_profiles` (`paper_table3`, `hw_as_run`,
  `hw_eef_y_tracking`, `paper_eef_tracking`, `sim_eef_yz_tracking`, `sim_eef_eight_tracking`,
  `sim_eef_xy_reaching`).

End-effector tasks are passed in the MPC reference: `eef_task_mode` plus world-frame `ref_eef_x/y/z`
(scalar or one value per node). The task presets in `simulation/zimpedance_tasks.py` build them.
