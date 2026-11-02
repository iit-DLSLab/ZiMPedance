# gym_quadruped

A reduced copy of [gym-quadruped](https://github.com/iit-DLSLab/gym-quadruped) (DLS Lab, IIT): the
single-robot MuJoCo environment (`quadruped_env.py`), its two-robot variant (`quadruped_env_coll.py`)
and the utilities they use, with the robot models of this work only:

| Model | Content |
|---|---|
| `robot_model/aliengo_leader/` | Aliengo with the 3-DoF passive arm mounted at the rear; `scene_<terrain>.xml` |
| `robot_model/aliengo_follower/` | Aliengo with the passive arm mounted at the front; `scene_<terrain>.xml` |
| `robot_model/aliengo_two_robots/` | the two arms on fixed blocks with a shared payload (standalone PACC), and the rigid bar of the two-robot scene |

`new_urdf_export.xml` in each robot folder is the MJCF that both MuJoCo (through the scenes) and the
ZiMPedance prediction model (Pinocchio) load.
