"""Task presets for the ZiMPedance controller (Schur ARMPC, mpc_params['type'] = 'zimpedance') in the
single-robot MuJoCo simulation.

    python simulation/zimpedance_sim.py --task <name> [--duration S] [--headless] [--export out.npz]

A preset is a few config overrides plus, optionally, a base-velocity schedule and an end-effector
(EEF) reference. The y tracking and circle use the amplitudes and frequencies of the paper
(Sec. V-D, Fig. 6). The disturbances are implemented in zimpedance_sim.py (apply_impulse, apply_sin,
release_from_position).
"""
import json

import numpy as np

# The EEF tasks run on the follower robot (arm at the front, as on hardware) with an arm impedance
# chosen to make the motion visible: joint 1 soft and nearly undamped
# (k 3, d 0.05: the arm swings freely sideways while the base does the tracking), joint 2 stiff
# (k 25: it holds the 3.6 kg distal link up, q2 ~0.3 rad instead of ~0.94 with the MJCF's k 8).
# Set on both the MuJoCo plant and the prediction model; the MJCF has k [5, 8, 5], d [0.5, 1.5, 0.5].
# The locomotion and disturbance presets keep the rear-arm leader with its MJCF impedance, as
# validated by the regression; with com_centred_footholds and velocity_trim_gain (below) they also run
# on the follower (locomotion 0.097 / -0.108 m/s for +-0.1 commanded).
DEFAULT_ROBOT = 'aliengo_leader'
EEF_TASK_ARM = dict(robot='aliengo_follower', arm_stiffness=[3, 25, 5], arm_damping=[0.05, 1.5, 0.5])

# Settings shared by every preset: the ZiMPedance controller, arm parameters from the robot model,
# and stance torques that realize the MPC's GRFs.
COMMON = {
    'mpc_params': {'type': 'zimpedance', 'armpc_arm_params': 'model', 'armpc_weight_profile': 'paper_table3'},
    'simulation_params': {'dt': 0.002, 'scene': 'flat', 'gait': 'crawl', 'stance_leg_bias_compensation': True,
                          'com_centred_footholds': True, 'velocity_trim_gain': 0.5},
    'collaborative_task_params': {'apply_impulse': False, 'apply_sin': False, 'release_from_position': False,
                                  'sin_traj_tracking': False},
}

TASKS = {
    'locomotion': dict(
        description='Flat ground crawl: forward 0.1 m/s, turn in place 0.2 rad/s, backward 0.1 m/s, stop',
        duration=34.0,
        velocity=[(2.0, 14.0, 0.1, 0.0, 0.0), (14.0, 22.0, 0.0, 0.0, 0.2), (22.0, 30.0, -0.1, 0.0, 0.0)]),
    'rough_terrain': dict(
        description='Forward 0.1 m/s crawl over scene_rough (tilted boxes up to 0.1 m)',
        duration=40.0,
        config={'simulation_params': {'scene': 'rough'}},
        velocity=[(2.0, 36.0, 0.1, 0.0, 0.0)]),
    'y_tracking': dict(
        **EEF_TASK_ARM,
        description='EEF lateral sine, 0.1 m amplitude at 0.4 Hz (paper Fig. 6, left), standing',
        duration=22.0,
        config={'mpc_params': {'armpc_weight_profile': 'paper_eef_tracking'}},
        eef=dict(mode='eef_y_sine_tracking', start=2.0, amplitude=0.1, frequency=0.4)),
    'yz_reaching': dict(
        **EEF_TASK_ARM,
        description='EEF y-z targets held 2 s each, cycled twice: (dy, dz) from the initial EEF position',
        duration=22.0,
        config={'mpc_params': {'armpc_weight_profile': 'sim_eef_yz_tracking'}},
        eef=dict(mode='eef_yz_reaching', start=2.0, hold=2.0,
                 dy=[0.0, 0.04, -0.04, -0.04, 0.04], dz=[0.0, -0.1, -0.1, 0.08, 0.08])),
    'circular_tracking': dict(
        **EEF_TASK_ARM,
        description='EEF y-z circle, radius 0.1 m at 0.25 Hz, reached in 2 s (paper Fig. 6, right, used a '
                    '0.03 x 0.08 m ellipse)',
        duration=22.0,
        config={'mpc_params': {'armpc_weight_profile': 'sim_eef_yz_tracking'}},
        eef=dict(mode='eef_circle_tracking', start=2.0, radius_y=0.1, radius_z=0.1, frequency=0.25, ramp=2.0)),
    'figure_eight_tracking': dict(
        **{**EEF_TASK_ARM, 'arm_damping': [0.02, 1.5, 0.5]},  # joint 1 even less damped: the arm swings
        description='EEF y-z figure eight (lemniscate), 0.12 m in y and 0.08 m in z at 0.2 Hz, made by '
                    'swinging the arm while the base stays nearly still',
        duration=22.0,
        config={'mpc_params': {'armpc_weight_profile': 'sim_eef_eight_tracking'}},
        eef=dict(mode='eef_eight_tracking', start=2.0, radius_y=0.12, radius_z=0.16, frequency=0.2)),
    'moving_target_reaching': dict(
        **EEF_TASK_ARM,
        description='EEF reaches a target moving at 0.05 m/s between y-z waypoints (1 s pauses); '
                    'the MPC sees only where the target is now',
        duration=24.0,
        config={'mpc_params': {'armpc_weight_profile': 'sim_eef_yz_tracking'}},
        eef=dict(mode='eef_moving_target_reaching', start=2.0, speed=0.05, pause=1.0,
                 dy=[0.0, 0.05, 0.05, -0.05, -0.05, 0.0], dz=[0.0, 0.06, -0.08, -0.08, 0.06, 0.0])),
    'spatial_reaching': dict(
        **EEF_TASK_ARM,
        description='EEF reaches four fixed y-z targets (2.5 s each), then follows a target moving at '
                    '0.06 m/s around a diamond; the MPC sees only where the target is now',
        duration=26.0,
        config={'mpc_params': {'armpc_weight_profile': 'sim_eef_eight_tracking'}},
        eef=dict(mode='eef_spatial_reaching', start=2.0, hold=2.5,
                 dy=[0.08, -0.08, -0.06, 0.06], dz=[0.05, 0.05, -0.08, -0.08],
                 speed=0.06, pause=0.0, move_dy=[0.0, 0.08, 0.0, -0.08], move_dz=[0.07, 0.0, -0.07, 0.0])),
    'xy_reaching': dict(
        **EEF_TASK_ARM,
        description='Walk to place the EEF on x-y targets up to 0.6 m away, holding each 3 s once reached',
        duration=30.0,
        config={'mpc_params': {'armpc_weight_profile': 'sim_eef_xy_reaching'}},
        eef=dict(mode='eef_xy_reaching', start=2.0, dx=[0.6, 0.6, 0.0], dy=[0.0, 0.3, 0.0],
                 tolerance=0.03, dwell=3.0, gain=0.6, max_speed=0.1, reach=0.08)),
    'disturbance_rejection': dict(
        description='Eight 30 N, 0.2 s EEF impulses 2 s apart (from 3 s) while crawling forward at 0.1 m/s',
        duration=24.0,
        config={'collaborative_task_params': {'apply_impulse': True, 'amplitude': 30.0}},
        velocity=[(2.0, 24.0, 0.1, 0.0, 0.0)]),
    'disturbance_sinusoidal': dict(
        description='15 N, 1 Hz sinusoidal EEF force along x while crawling forward at 0.1 m/s',
        duration=20.0,
        config={'collaborative_task_params': {'apply_sin': True, 'amplitude': 15.0, 'frequency': 1.0}},
        velocity=[(2.0, 20.0, 0.1, 0.0, 0.0)]),
    'pull_release': dict(
        description='Four held EEF pulls (~19 N, alternating left/right) and releases, standing; 7 s to settle',
        duration=26.0,
        config={'collaborative_task_params': {'release_from_position': True}}),
}


def apply_task(cfg, name):
    """Apply a preset to the config module. Call before the controller is created."""
    task = dict(TASKS[name])
    task.setdefault('robot', DEFAULT_ROBOT)
    task.setdefault('arm_stiffness', None)  # None: the MJCF's
    task.setdefault('arm_damping', None)
    cfg.select_robot(task['robot'])
    for overrides in (COMMON, task.get('config', {})):
        for section, values in overrides.items():
            getattr(cfg, section).update(values)
    cfg.simulation_params['dt'] = cfg.sim_dt_by_controller[cfg.mpc_params['type']]
    return task


def set_arm_impedance(mj_model, wrapper, stiffness=None, damping=None):
    """Passive-arm joint stiffness [N m/rad] and damping [N m s/rad] of the MuJoCo plant and of the
    Schur ARMPC's prediction model (which otherwise reads them from the MJCF). None keeps the MJCF's."""
    model = wrapper.srbd_controller_interface.controller.centroidal_model
    joints = [mj_model.joint(name).id for name in model.arm_joint_names]
    if stiffness is not None:
        mj_model.jnt_stiffness[joints] = stiffness
        model.arm_stiffness = np.asarray(stiffness, dtype=float)
    if damping is not None:
        mj_model.dof_damping[mj_model.jnt_dofadr[joints]] = damping
        model.arm_damping = np.asarray(damping, dtype=float)


class VelocitySchedule:
    """Piecewise-constant base velocity command (vx, vy in the heading frame, yaw rate)."""

    def __init__(self, segments):
        self.segments = segments

    def __call__(self, t, **state):
        for t0, t1, vx, vy, wz in self.segments:
            if t0 <= t < t1:
                return np.array([vx, vy, 0.0]), wz
        return np.zeros(3), 0.0


class EEFTask:
    """World-frame EEF reference sampled at the MPC nodes.

    The reference is anchored at the measured EEF position when the task starts. (The original
    one-way ARMPC tasks, on the rear-arm leader, anchored z at the base height; the follower's EEF
    hangs 0.13 m below it, and reaching up there takes a 0.2 rad base pitch.) eef_xy_reaching also
    commands the base velocity (call the task like a VelocitySchedule) so that the robot walks the
    EEF to its targets.
    """

    def __init__(self, mode, start, horizon, mpc_dt, **params):
        self.mode, self.start, self.params = mode, start, params
        self.node_times = np.arange(horizon) * mpc_dt
        self.anchor = None
        self.target_idx, self.reached_since = 0, None  # eef_xy_reaching progress
        self.last_tau, self.markers_moving = None, None  # eef_spatial_reaching phase, cached path markers

    def reference(self, t, eef_pos, base_pos):
        if t < self.start:
            return None
        if self.anchor is None:
            self.anchor = dict(x=float(eef_pos[0]), y=float(eef_pos[1]), z=float(eef_pos[2]))
        tau = t - self.start + self.node_times
        self.last_tau = tau[0]
        p, y0, z0 = self.params, self.anchor['y'], self.anchor['z']
        ref = {'eef_task_mode': self.mode}
        if self.mode == 'eef_y_sine_tracking':
            ref['ref_eef_y'] = y0 + p['amplitude'] * np.sin(2 * np.pi * p['frequency'] * tau)
        elif self.mode == 'eef_y_step_tracking':
            steps = np.array([0.0, p['amplitude'], 0.0, -p['amplitude']])
            ref['ref_eef_y'] = y0 + steps[np.floor(tau / p['hold']).astype(int) % 4]
        elif self.mode == 'eef_yz_reaching':
            idx = np.floor(tau / p['hold']).astype(int) % len(p['dy'])
            ref['ref_eef_y'] = y0 + np.asarray(p['dy'])[idx]
            ref['ref_eef_z'] = z0 + np.asarray(p['dz'])[idx]
        elif self.mode in ('eef_circle_tracking', 'eef_eight_tracking'):
            ref['ref_eef_y'], ref['ref_eef_z'] = self._periodic(tau)
        elif self.mode == 'eef_moving_target_reaching':
            # no preview: every node holds the target's current position
            y, z = self._waypoint_path(np.full_like(tau, tau[0]), p['dy'], p['dz'])
            ref['ref_eef_y'], ref['ref_eef_z'] = y, z
        elif self.mode == 'eef_spatial_reaching':
            # fixed targets, then a moving one; no preview in either phase
            fixed = len(p['dy']) * p['hold']
            if tau[0] < fixed:
                k = int(tau[0] // p['hold'])
                y, z = np.full_like(tau, y0 + p['dy'][k]), np.full_like(tau, z0 + p['dz'][k])
            else:
                y, z = self._waypoint_path(np.full_like(tau, tau[0] - fixed), p['move_dy'], p['move_dz'])
            ref['ref_eef_y'], ref['ref_eef_z'] = y, z
        elif self.mode == 'eef_xy_reaching':
            self._update_target(t, eef_pos)
            # the arm leans towards the target, at most `reach` away from where the EEF is
            error = self._xy_target() - np.asarray(eef_pos[:2])
            error *= min(1.0, p['reach'] / max(np.linalg.norm(error), 1e-9))
            ref['ref_eef_x'], ref['ref_eef_y'] = (np.full_like(tau, eef_pos[i] + error[i]) for i in (0, 1))
        else:
            raise ValueError(f"unknown EEF task mode {self.mode}")
        return ref

    def __call__(self, t, eef_pos=None, heading_R=None, **state):
        """Base command (heading-frame velocity, yaw rate) of eef_xy_reaching: a saturated proportional
        law on the EEF's distance to the target, zero inside the tolerance."""
        if self.mode != 'eef_xy_reaching' or self.anchor is None or eef_pos is None:
            return np.zeros(3), 0.0
        p = self.params
        error = self._xy_target() - np.asarray(eef_pos[:2])
        distance = np.linalg.norm(error)
        if distance < p['tolerance']:
            return np.zeros(3), 0.0
        v = p['gain'] * error * min(1.0, p['max_speed'] / (p['gain'] * distance))
        return heading_R.T @ np.array([v[0], v[1], 0.0]), 0.0

    def _periodic(self, tau):
        p = self.params
        w = 2 * np.pi * p['frequency'] * tau
        if self.mode == 'eef_circle_tracking':
            # the radius grows from 0 over `ramp` seconds: no jump from the EEF to the circle at the start
            grow = np.clip(tau / p['ramp'], 0.0, 1.0) if p.get('ramp') else 1.0
            return (self.anchor['y'] + grow * p['radius_y'] * np.cos(w),
                    self.anchor['z'] + grow * p['radius_z'] * np.sin(w))
        # lemniscate of Gerono
        return self.anchor['y'] + p['radius_y'] * np.sin(w), self.anchor['z'] + p['radius_z'] * np.sin(w) * np.cos(w)

    def _waypoint_path(self, tau, dy, dz):
        """A moving target: constant-speed segments between the waypoints (dy, dz from the anchor),
        each followed by a pause, looping back to the first waypoint."""
        p = self.params
        points = np.column_stack((self.anchor['y'] + np.asarray(dy), self.anchor['z'] + np.asarray(dz)))
        points = np.vstack((points, points[:1]))
        move = np.linalg.norm(np.diff(points, axis=0), axis=1) / p['speed']
        t_end = np.cumsum(move + p['pause'])
        tau = np.mod(tau, t_end[-1])
        seg = np.searchsorted(t_end, tau, side='right')
        frac = np.clip((tau - (t_end[seg] - move[seg] - p['pause'])) / np.maximum(move[seg], 1e-9), 0.0, 1.0)
        target = points[seg] + frac[:, None] * (points[seg + 1] - points[seg])
        return target[:, 0], target[:, 1]

    def _waypoint_period(self, dy, dz):
        points = np.column_stack((dy, dz))
        points = np.vstack((points, points[:1]))
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1) / self.params['speed']
                            + self.params['pause']))

    def _xy_target(self):
        p = self.params
        return np.array([self.anchor['x'] + p['dx'][self.target_idx], self.anchor['y'] + p['dy'][self.target_idx]])

    def _update_target(self, t, eef_pos):
        """Move on to the next eef_xy_reaching target after `dwell` seconds within `tolerance`."""
        p = self.params
        if np.linalg.norm(self._xy_target() - np.asarray(eef_pos[:2])) > p['tolerance']:
            self.reached_since = None
        elif self.reached_since is None:
            self.reached_since = t
        elif t - self.reached_since >= p['dwell'] and self.target_idx < len(p['dx']) - 1:
            self.target_idx, self.reached_since = self.target_idx + 1, None

    def markers(self, eef_pos):
        """Spheres (position, diameter, rgba) that show the task in the viewer: the path
        or targets in light green, the current target in solid green."""
        if self.anchor is None:
            return []
        p, x0, y0, z0 = self.params, self.anchor['x'], self.anchor['y'], self.anchor['z']
        path, light = [], [0.2, 0.8, 0.2, 0.35]
        if self.mode in ('eef_circle_tracking', 'eef_eight_tracking', 'eef_y_sine_tracking'):
            tau = (p.get('ramp') or 0.0) + np.linspace(0.0, 1.0 / p['frequency'], 60, endpoint=False)  # full size
            if self.mode == 'eef_y_sine_tracking':
                y, z = y0 + p['amplitude'] * np.sin(2 * np.pi * p['frequency'] * tau), np.full_like(tau, eef_pos[2])
            else:
                y, z = self._periodic(tau)
            path = [([x0, yi, zi], 0.015, light) for yi, zi in zip(y, z)]
        elif self.mode in ('eef_yz_reaching', 'eef_moving_target_reaching'):
            path = [([x0, y0 + dy, z0 + dz], 0.03, light) for dy, dz in zip(p['dy'], p['dz'])]
        elif self.mode == 'eef_spatial_reaching':
            # the fixed targets, then the moving target's path
            n = len(p['dy'])
            if self.markers_moving is None:
                tau = np.linspace(0.0, 1.0, 80) * self._waypoint_period(p['move_dy'], p['move_dz'])
                y, z = self._waypoint_path(tau, p['move_dy'], p['move_dz'])
                self.markers_moving = [([x0, yi, zi], 0.012, light) for yi, zi in zip(y, z)]
            fixed_phase = self.last_tau is None or self.last_tau < n * p['hold']
            path = ([([x0, y0 + dy, z0 + dz], 0.03, light) for dy, dz in zip(p['dy'], p['dz'])]
                    if fixed_phase else self.markers_moving)
        elif self.mode == 'eef_xy_reaching':
            path = [([x0 + dx, y0 + dy, eef_pos[2]], 0.04, light) for dx, dy in zip(p['dx'], p['dy'])]
        return path

    def target_marker(self, reference, eef_pos):
        if reference is None:
            return []
        if self.mode == 'eef_xy_reaching':  # the target itself, not the leaning reference
            target = [*self._xy_target(), eef_pos[2]]
        else:
            target = [np.asarray(reference.get(k, eef_pos[i])).flat[0]
                      for i, k in enumerate(('ref_eef_x', 'ref_eef_y', 'ref_eef_z'))]
        return [(target, 0.04, [0.1, 0.7, 0.1, 0.9])]


def add_markers(scene, markers, geom_ids=None):
    """Draw marker spheres into an MjvScene. With geom_ids (a list, filled on the first call) the
    same scene slots are reused, as the viewer's user_scn keeps its geoms between frames."""
    import mujoco
    for i, (pos, diameter, rgba) in enumerate(markers):
        if geom_ids is not None and i < len(geom_ids):
            idx = geom_ids[i]
        else:
            if scene.ngeom >= scene.maxgeom:
                return
            idx, scene.ngeom = scene.ngeom, scene.ngeom + 1
            if geom_ids is not None:
                geom_ids.append(idx)
        mujoco.mjv_initGeom(scene.geoms[idx], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.full(3, diameter / 2),
                            pos=np.asarray(pos, float), mat=np.eye(3).flatten(), rgba=np.asarray(rgba, float))
        scene.geoms[idx].category = mujoco.mjtCatBit.mjCAT_DECOR


def make_task_runtime(task, horizon, mpc_dt):
    eef = EEFTask(horizon=horizon, mpc_dt=mpc_dt, **task['eef']) if 'eef' in task else None
    if eef is not None and eef.mode == 'eef_xy_reaching':
        return eef, eef  # the EEF task drives the base
    velocity = VelocitySchedule(task.get('velocity', []))  # no schedule: hold still
    return velocity, eef


def run_metadata(cfg, task, wrapper):
    """Task and configuration of a run, for the exported log."""
    meta = dict(task=task, robot=cfg.robot,
                mpc_params={k: cfg.mpc_params.get(k) for k in (
                    'type', 'horizon', 'dt', 'armpc_weight_profile', 'armpc_arm_params', 'armpc_model_xml',
                    'use_zmp_stability', 'crawl_stability_margin')},
                simulation_params={k: cfg.simulation_params.get(k) for k in (
                    'dt', 'scene', 'gait', 'mpc_frequency', 'stance_leg_bias_compensation', 'com_centred_footholds',
                    'velocity_trim_gain')},
                gait_params=cfg.simulation_params['gait_params'].get(cfg.simulation_params['gait']))
    model = getattr(wrapper.srbd_controller_interface.controller, 'centroidal_model', None)
    if hasattr(model, 'arm_stiffness'):
        meta['arm_model'] = dict(stiffness=model.arm_stiffness.tolist(), damping=model.arm_damping.tolist(),
                                 spring_rest=model.arm_spring_rest.tolist(), total_mass=model.total_mass)
    meta['units'] = 'SI; angles in rad; base_ori is roll-pitch-yaw; eef_ref is the node-0 reference (NaN: none)'
    return meta


def export_npz(path, log, metadata):
    """Write a trajectory log to a compressed NPZ, with metadata as a JSON string."""
    arrays = {}
    for key, values in log.items():
        if len(values) and all(hasattr(values[0], leg) for leg in ('FL', 'FR', 'RL', 'RR')):
            # per-leg entries (LegsAttr): stack to (steps, 4, ...) in FL, FR, RL, RR order
            values = [[np.asarray(v[leg], dtype=float) for leg in ('FL', 'FR', 'RL', 'RR')] for v in values]
        try:
            arrays[key] = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            continue  # per-leg dicts and other ragged entries are not exported
    np.savez_compressed(path, metadata=json.dumps(metadata, default=str), **arrays)
