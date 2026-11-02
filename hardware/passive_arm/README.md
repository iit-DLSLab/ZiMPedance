# PACC passive arm: printable parts and encoders

STL files of the passive arm's four parts, in millimetres, in `stl/`. They come from the arm's CAD,
through the simulation meshes in `gym_quadruped/robot_model/aliengo_follower/assets/`
(in metres); `export_stl.py` rescales those and reports the mesh checks below.

| Part | File | Size [mm] | Volume [cm³] | Mesh |
|---|---|---|---|---|
| Base (mounts on the robot trunk) | `stl/base_arm_link.stl` | 145.0 × 80.4 × 82.0 | 148.8 | 1 open, 2 non-manifold edges |
| Link 1 (joint 1, vertical axis) | `stl/arm_link_1.stl` | 105.5 × 86.4 × 82.5 | 173.1 | closed |
| Link 2 | `stl/arm_link_2.stl` | 283.3 × 46.7 × 222.0 | 566.1 | 2 non-manifold edges; one enclosed cavity (0.8 cm³) |
| Link 3 (carries the end effector) | `stl/arm_link_3.stl` | 110.0 × 35.0 × 323.7 | 313.9 | 4 open, 2 non-manifold edges; one enclosed cavity (0.3 cm³) |

- **One file, one part.** The second body in links 2 and 3 is an inner shell with inverted normals:
  a cavity inside the part, not a separate piece. Slicers keep it as a void.
- **Mesh defects:** a handful of edges out of 27 000–83 000 triangles; slicers repair them on import.
- **Placement:** each part is in its CAD frame (joint axis at the origin), not oriented for printing.

## Joint encoders

Each joint carries a **Bourns EMS22A** non-contacting absolute magnetic encoder (1024 positions,
5 V, SSI-like serial output). Datasheet: Bourns, *EMS22A Non-Contacting Absolute Encoder*.

`arduino/sketch_3joints_full/sketch_3joints_full.ino` (Arduino Uno) reads the three encoders and prints
`rad1<TAB>rad2<TAB>rad3` (0 to 2π each, 6 decimals) over USB serial at 115200 baud, every 50 ms.
`PLOTTER_MODE = true` prints raw counts for the Arduino Serial Plotter instead.

| Encoder pin | Wire | Joint 1 | Joint 2 | Joint 3 |
|---|---|---|---|---|
| 1 DI | blue | D8 (driven LOW) | D4 (driven LOW) | GND |
| 2 CLK | grey | D11 | D7 | A0 |
| 3 GND | black | GND | GND | GND |
| 4 DO | white | D10 | D6 | A1 |
| 5 VCC | pink | 5 V | 5 V | 5 V |
| 6 CS | brown | D9 | D5 | A2 |

The robot side (a ROS 2 node, not part of this release)
unwraps the 0–2π readings into continuous angles and flips joint 1 (sign −1, +1, +1).

The parts were printed in PLA. Print settings, the remaining bought parts (bearings, shafts,
springs, fasteners) and assembly notes are not documented yet; open an issue if you build the arm.

```bash
python hardware/passive_arm/export_stl.py   # regenerate stl/ from the simulation meshes
```
