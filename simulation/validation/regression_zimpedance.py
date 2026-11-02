"""Regression checks for the ZiMPedance controller (Schur ARMPC) in the single-robot MuJoCo sim.

    python simulation/validation/regression_zimpedance.py [--tasks y_tracking,circular_tracking] [--jobs 6] [--out DIR]

Runs each task preset of simulation/zimpedance_tasks.py headless, in its own process, through
`zimpedance_sim.py --task <name> --export`, then checks physical behaviour from the exported
log: attitude, height, velocity or EEF tracking, disturbance recovery, solver status and finite
states. Thresholds are about twice the values observed when the baseline was recorded (listed in README.md);
they flag a change in behaviour, not the limits of the controller. Exit code 1 if a check fails.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIM = os.path.join(REPO, 'simulation', 'zimpedance_sim.py')
ROBOT_MODELS = os.path.join(REPO, 'gym_quadruped', 'robot_model')
OK_STATUS = (0, 2)  # SQP converged, or stopped at its single iteration (nlp_solver_max_iter = 1)
sys.path.insert(0, os.path.join(REPO, 'simulation'))
from zimpedance_tasks import TASKS, DEFAULT_ROBOT  # noqa: E402
ROBOTS = sorted({task.get('robot', DEFAULT_ROBOT) for task in TASKS.values()})  # the robots the presets use
MJCF_ARM_STIFFNESS, MJCF_ARM_DAMPING = [5, 8, 5], [0.5, 1.5, 0.5]  # both robot MJCFs
TOTAL_MASS = {'aliengo_leader': 27.946,  # arm link 3 of 0.2 kg and a 2 kg EEF payload body
              'aliengo_follower': 29.346}  # arm link 3 of 3.5 kg and a 0.1 kg EEF body


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files if k != 'metadata'}, json.loads(str(d['metadata']))


def heading_velocity(d):
    yaw = d['base_ori'][:, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    v = d['base_lin_vel']
    return np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], d['base_ang_vel'][:, 2]], 1)


def common_checks(d, meta):
    t, p, o = d['time'], d['base_pos'], d['base_ori']
    late = t > 1.0
    status = d['mpc_status']
    return {
        'completed': (t[-1] >= meta['task']['duration'] - 0.01, f"{t[-1]:.2f} s"),
        'finite': (bool(np.isfinite(d['qpos']).all()), ''),
        'solver_status': (bool(np.isin(status, OK_STATUS).all()),
                          str({int(k): int(n) for k, n in zip(*np.unique(status, return_counts=True))})),
        'no_fall': (p[late, 2].min() > 0.25 and np.abs(o[late, :2]).max() < 0.8,
                    f"z_min {p[late, 2].min():.3f} m, max|roll,pitch| {np.abs(o[late, :2]).max():.3f} rad"),
        'robot_and_arm_impedance': (meta['robot'] == meta['task']['robot']
                                    and np.allclose(meta['arm_model']['stiffness'],
                                                    meta['task']['arm_stiffness'] or MJCF_ARM_STIFFNESS)
                                    and np.allclose(meta['arm_model']['damping'],
                                                    meta['task']['arm_damping'] or MJCF_ARM_DAMPING),
                                    f"{meta['robot']}, prediction model k {meta['arm_model']['stiffness']}, "
                                    f"d {meta['arm_model']['damping']}"),
    }


def check_velocity_segments(d, meta, tolerances):
    t, o = d['time'], d['base_ori']
    vh = heading_velocity(d)
    out = {}
    for (t0, t1, vx, vy, wz), (axis, lo, hi) in zip(meta['task']['velocity'], tolerances):
        w = (t > t0 + 1.5) & (t < t1)
        val = vh[w, axis].mean()
        out[f'velocity {[vx, vy, wz]}'] = (lo <= val <= hi, f"{['vx', 'vy', 'wz'][axis]} {val:.3f} in [{lo}, {hi}]")
        out[f'attitude {[vx, vy, wz]}'] = (np.abs(o[w, :2]).max() < 0.1,
                                           f"max|roll,pitch| {np.abs(o[w, :2]).max():.3f} rad")
    stop = t > meta['task']['velocity'][-1][1] + 2.0
    if stop.any():
        speed = np.linalg.norm(d['base_lin_vel'][stop, :2], axis=1).mean()
        out['stops'] = (speed < 0.02, f"mean speed {speed:.4f} m/s after the last command")
    return out


def check_locomotion(d, meta):
    # forward 0.098, yaw rate 0.210, backward -0.098 observed (CoM-centred footholds and velocity trim)
    return check_velocity_segments(d, meta, [(0, 0.08, 0.12), (2, 0.17, 0.23), (0, -0.12, -0.08)])


def check_rough_terrain(d, meta):
    p, o, t = d['base_pos'], d['base_ori'], d['time']
    out = check_velocity_segments(d, meta, [(0, 0.08, 0.12)])
    out.pop('attitude [0.1, 0.0, 0.0]')
    out['traversed'] = (p[-1, 0] - p[0, 0] > 3.0, f"x travel {p[-1, 0] - p[0, 0]:.2f} m (observed 3.27)")
    out['attitude'] = (np.abs(o[t > 1, :2]).max() < 0.3, f"max|roll,pitch| {np.abs(o[t > 1, :2]).max():.3f} rad (observed 0.143)")
    return out


def eef_errors(d, meta, axis, settle=2.0):
    t, e, r = d['time'], d['end_eff_pos'], d['eef_ref']
    w = (t > meta['task']['eef']['start'] + settle) & ~np.isnan(r[:, axis])
    err = e[w, axis] - r[w, axis]
    return np.sqrt((err ** 2).mean()), np.abs(err).max(), np.ptp(e[w, axis])


def check_y_tracking(d, meta):
    rms, mx, ptp = eef_errors(d, meta, 1)
    return {'y_rms': (rms < 0.010, f"{1e3 * rms:.1f} mm (observed 1.0)"),
            'y_max': (mx < 0.015, f"{1e3 * mx:.1f} mm (observed 1.5)"),
            'y_amplitude': (ptp > 0.18, f"measured peak-to-peak {1e3 * ptp:.0f} mm of 200")}


def check_yz_path(d, meta, limits):
    out = {}
    for axis, name, rms_tol, max_tol, ptp_min, obs in limits:
        rms, mx, ptp = eef_errors(d, meta, axis)
        out[f'{name}_rms_max'] = (rms < rms_tol and mx < max_tol,
                                  f"{1e3 * rms:.1f} / {1e3 * mx:.1f} mm (observed {obs})")
        out[f'{name}_amplitude'] = (ptp > ptp_min, f"measured peak-to-peak {1e3 * ptp:.0f} mm")
    return out


def check_circular_tracking(d, meta):
    return check_yz_path(d, meta, ((1, 'y', 0.010, 0.018, 0.17, '5.1 / 9.1'), (2, 'z', 0.016, 0.025, 0.16, '8.1 / 11.7')))


def check_figure_eight_tracking(d, meta):
    out = check_yz_path(d, meta, ((1, 'y', 0.008, 0.012, 0.20, '2.0 / 2.8'), (2, 'z', 0.020, 0.032, 0.12, '11.7 / 19.6')))
    # the arm makes the shape: the base barely sways
    w = d['time'] > meta['task']['eef']['start'] + 2.0
    sway = np.ptp(d['base_pos'][w, 1])
    out['base_sway'] = (sway < 0.03, f"base y peak-to-peak {1e3 * sway:.0f} mm (observed 13)")
    return out


def check_moving_target_reaching(d, meta):
    # the MPC sees the target's current position only, so the EEF lags it by about the move speed x delay
    return check_yz_path(d, meta, ((1, 'y', 0.017, 0.045, 0.08, '9.1 / 24.5'), (2, 'z', 0.025, 0.043, 0.12, '10.7 / 19.2')))


def check_spatial_reaching(d, meta):
    t, e, r = d['time'], d['end_eff_pos'], d['eef_ref']
    task = meta['task']['eef']
    fixed_end = task['start'] + len(task['dy']) * task['hold']
    worst = 0.0
    for k in range(len(task['dy'])):  # settled part of each fixed target
        w = (t > task['start'] + k * task['hold'] + 1.5) & (t < task['start'] + (k + 1) * task['hold'])
        worst = max(worst, np.abs(e[w, 1:] - r[w, 1:]).mean(0).max())
    w = t > fixed_end + 1.0
    rms = np.sqrt(((e[w, 1:] - r[w, 1:]) ** 2).mean(0))
    return {'fixed_targets': (worst < 0.015, f"worst settled |error| {1e3 * worst:.1f} mm (observed 6.9)"),
            'moving_target': (rms.max() < 0.040, f"RMS to the target's current position y {1e3 * rms[0]:.1f}, "
                                                 f"z {1e3 * rms[1]:.1f} mm (observed 18.4, 9.3)")}


def check_xy_reaching(d, meta):
    t, e, o = d['time'], d['end_eff_pos'], d['base_ori']
    task = meta['task']['eef']
    anchor = e[np.searchsorted(t, task['start']), :2]
    targets = anchor + np.column_stack((task['dx'], task['dy']))
    distance = np.linalg.norm(e[:, None, :2] - targets[None], axis=2)  # (steps, targets)
    first, since = [], task['start']
    for k in range(len(targets)):  # each target after the previous one, from the task start
        r = np.flatnonzero((t > since) & (distance[:, k] < task['tolerance']))
        since = t[r[0]] if r.size else np.inf
        first.append(since)
    end = t > t[-1] - 1.0
    return {'targets_reached_in_order': (bool(np.isfinite(first[-1])),
                                         f"first within {1e3 * task['tolerance']:.0f} mm at {np.round(first, 1).tolist()} s "
                                         f"(observed 6.2, 11.6, 18.3)"),
            'holds_last_target': (distance[end, -1].max() < task['tolerance'],
                                  f"last 1 s: max distance {1e3 * distance[end, -1].max():.1f} mm"),
            'attitude': (np.abs(o[t > 1, :2]).max() < 0.4,
                         f"max|roll,pitch| {np.abs(o[t > 1, :2]).max():.3f} rad (observed 0.173)")}


def check_yz_reaching(d, meta):
    t, e, r = d['time'], d['end_eff_pos'], d['eef_ref']
    task = meta['task']['eef']
    start, hold, n = task['start'], task['hold'], len(task['dy'])
    errs = []
    for k in range(int((t[-1] - start) // hold)):
        # settled part of each hold, before the next target enters the 0.5 s horizon
        w = (t > start + k * hold + 1.0) & (t <= start + (k + 1) * hold - 0.5)
        errs.append(np.abs((e[w, 1:] - r[w, 1:]).mean(0)))
    errs = np.array(errs)
    return {'targets_reached': (errs.max() < 0.015,
                                f"worst steady |error| y {1e3 * errs[:, 0].max():.1f}, z {1e3 * errs[:, 1].max():.1f} mm "
                                f"over {len(errs)} holds of {n} targets (observed <= 6.0)")}


def disturbance_events(d):
    t, f = d['time'], d['impulse']
    on = np.linalg.norm(f, axis=1) > 1e-9
    starts = np.flatnonzero(np.diff(on.astype(int)) == 1) + 1
    ends = np.flatnonzero(np.diff(on.astype(int)) == -1) + 1
    return t, on, starts, ends


def check_disturbance_rejection(d, meta):
    t, on, starts, ends = disturbance_events(d)
    o, q = d['base_ori'], d['joint_pos_arm']
    q0 = q[np.searchsorted(t, 1.0)]  # arm pose calibrated at 1 s
    recover = []
    for a in starts:
        b = ends[ends > a][0]
        nxt = starts[starts > a][0] if (starts > a).any() else len(t)
        ok = np.flatnonzero((np.abs(o[b:nxt, :2]).max(1) < 0.05) & (np.abs(q[b:nxt] - q0).max(1) < 0.1))
        recover.append(t[b + ok[0]] - t[b] if ok.size else np.inf)
    return {'impulses': (len(starts) == 8, f"{len(starts)} impulses of {np.linalg.norm(d['impulse'], axis=1).max():.1f} N"),
            'attitude': (np.abs(o[t > 1, :2]).max() < 0.15, f"max|roll,pitch| {np.abs(o[t > 1, :2]).max():.3f} rad (observed 0.077)"),
            'recovery': (max(recover) < 1.5, f"slowest recovery {max(recover):.2f} s (observed 0.85)")}


def check_disturbance_sinusoidal(d, meta):
    t, o = d['time'], d['base_ori']
    out = check_velocity_segments(d, meta, [(0, 0.07, 0.12)])
    out.pop('attitude [0.1, 0.0, 0.0]')
    out['attitude'] = (np.abs(o[t > 1, :2]).max() < 0.25, f"max|roll,pitch| {np.abs(o[t > 1, :2]).max():.3f} rad (observed 0.114)")
    return out


def check_pull_release(d, meta):
    t, o, p = d['time'], d['base_ori'], d['base_pos']
    end = t > t[-1] - 1.0
    return {'recovers': (np.abs(o[end, :2]).max() < 0.02 and abs(p[end, 2].mean() - 0.324) < 0.01,
                         f"last 1 s: max|roll,pitch| {np.abs(o[end, :2]).max():.4f} rad, z {p[end, 2].mean():.3f} m")}


CHECKS = {
    'locomotion': check_locomotion, 'rough_terrain': check_rough_terrain,
    'y_tracking': check_y_tracking, 'yz_reaching': check_yz_reaching, 'circular_tracking': check_circular_tracking,
    'figure_eight_tracking': check_figure_eight_tracking, 'moving_target_reaching': check_moving_target_reaching,
    'spatial_reaching': check_spatial_reaching, 'xy_reaching': check_xy_reaching,
    'disturbance_rejection': check_disturbance_rejection, 'disturbance_sinusoidal': check_disturbance_sinusoidal,
    'pull_release': check_pull_release,
}


def check_model(robot_name):
    """Model selection and the arm parameters the controller reads, without building the solver."""
    sys.path.insert(0, REPO)
    from quadruped_pympc import config as cfg
    from quadruped_pympc.controllers.zimpedance.arm_augmented_mpc_nominal_model import (
        Arm_Augmented_Centroidal_Model)
    cfg.select_robot(robot_name)
    role = robot_name.split('_')[1]
    model = Arm_Augmented_Centroidal_Model()
    expected_xml = os.path.join(ROBOT_MODELS, robot_name, 'new_urdf_export.xml')
    return {
        'model_xml': (os.path.realpath(model.model_path) == os.path.realpath(expected_xml), model.model_path),
        'arm_joints': (model.arm_joint_names == [f'{role}/arm_link_{i}_joint' for i in (1, 2, 3)],
                       str(model.arm_joint_names)),
        # the MJCF's impedance; the EEF presets replace the damping (checked on every run)
        'arm_impedance': (np.allclose(model.arm_stiffness, MJCF_ARM_STIFFNESS) and np.allclose(model.arm_damping, MJCF_ARM_DAMPING)
                          and np.allclose(model.arm_spring_rest, 0),
                          f"k {model.arm_stiffness}, d {model.arm_damping}, rest {model.arm_spring_rest}"),
        'total_mass': (abs(model.total_mass - TOTAL_MASS[robot_name]) < 1e-3, f"{model.total_mass:.3f} kg"),
    }


SCENES = {'aliengo_leader': ('flat', 'rough', 'pyramid', 'ramp', 'slope', 'stairs', 'suspend_stairs'),
          'aliengo_follower': ('flat', 'rough', 'ramp', 'stairs')}


def check_scenes(robot_name):
    """Every terrain scene of a robot instantiates the same complete, actuated robot."""
    import mujoco
    role = robot_name.split('_')[1]
    base = os.path.join(ROBOT_MODELS, robot_name)
    names = lambda m, obj, n: [mujoco.mj_id2name(m, obj, i) for i in range(n)]
    legs = [f'{role}/{leg}_{j}' for leg in ('FL', 'FR', 'RL', 'RR') for j in ('hip', 'thigh', 'calf')]
    out, ref = {}, None
    for scene in SCENES[robot_name]:
        try:
            m = mujoco.MjModel.from_xml_path(os.path.join(base, f'scene_{scene}.xml'))
        except Exception as e:
            out[scene] = (False, f"does not load: {str(e)[:120]}")
            continue
        # robot physics: bodies 1.. (robot) and all joints; terrain geoms are excluded
        robot = np.concatenate([m.body_mass[1:], m.body_inertia[1:].ravel(), m.jnt_stiffness, m.dof_damping,
                                m.dof_armature, m.qpos_spring, m.actuator_ctrlrange.ravel()])
        ref = robot if ref is None else ref
        acts = names(m, mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu)
        act_joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) for i in range(m.nu)]
        ok = (acts == legs and act_joints == [a + '_joint' for a in legs]
              and 'imu' in names(m, mujoco.mjtObj.mjOBJ_SENSOR, m.nsensor)
              and m.nkey >= 1 and (m.nq, m.nv) == (22, 21)
              and names(m, mujoco.mjtObj.mjOBJ_JOINT, m.njnt)[-3:] == [f'{role}/arm_link_{i}_joint' for i in (1, 2, 3)]
              and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'eef') >= 0
              and robot.shape == ref.shape and np.array_equal(robot, ref))
        out[scene] = (ok, f"nu {m.nu}, sensors {names(m, mujoco.mjtObj.mjOBJ_SENSOR, m.nsensor)}, "
                          f"keyframe 0 {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, 0) if m.nkey else None}, "
                          f"mass {m.body_subtreemass[1]:.3f} kg")
    return out


def run_task(name, out):
    path = os.path.join(out, f'{name}.npz')
    t0 = time.time()
    proc = subprocess.run([sys.executable, SIM, '--task', name, '--headless', '--export', path],
                          cwd=REPO, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0 or not os.path.isfile(path):
        return name, wall, {'run': (False, proc.stderr.strip().splitlines()[-1] if proc.stderr else 'no output')}
    d, meta = load(path)
    results = common_checks(d, meta)
    results.update(CHECKS[name](d, meta))
    return name, wall, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default=','.join(CHECKS), help='comma-separated presets (default: all)')
    ap.add_argument('--jobs', type=int, default=min(len(CHECKS), os.cpu_count() or 1))
    ap.add_argument('--out', help='keep the exported logs here (default: a temporary directory)')
    ap.add_argument('--skip-kinematics', action='store_true', help='skip check_pinocchio_vs_mujoco.py')
    args = ap.parse_args()
    tasks = [t for t in args.tasks.split(',') if t]
    out = args.out or tempfile.mkdtemp(prefix='zimpedance_regression_')
    os.makedirs(out, exist_ok=True)

    report, failed = {}, False
    for robot_name in ROBOTS:
        report[f'{robot_name} model'] = check_model(robot_name)
    for robot_name in SCENES:
        report[f'{robot_name} scenes (same actuated robot)'] = check_scenes(robot_name)
    for robot_name in ([] if args.skip_kinematics else ROBOTS):
        role = robot_name.split('_')[1]
        xml = os.path.join(ROBOT_MODELS, robot_name, 'new_urdf_export.xml')
        kin_json = os.path.join(out, f'kinematics_{robot_name}.json')
        subprocess.run([sys.executable, os.path.join(REPO, 'simulation', 'validation', 'check_pinocchio_vs_mujoco.py'),
                        xml, '--role', role, '--out', kin_json], cwd=REPO, capture_output=True, text=True)
        if os.path.isfile(kin_json):
            kin = json.load(open(kin_json))
            e = kin['max_error']
            report[f'{robot_name} kinematics'] = {
                'eef_parent': (kin['eef_parent_joint_pin'] == f'{role}/arm_link_3_joint', kin['eef_parent_joint_pin']),
                'positions': (max(e['eef'], e['all_bodies']) < 1e-9, f"eef {e['eef']:.1e} m, bodies {e['all_bodies']:.1e} m"),
                'mass_matrix': (e['M_act'] < 1e-9, f"{e['M_act']:.1e}"),
                # Pinocchio uses g = 9.81, the XML 9.806: 5.3e-3 N m observed
                'gravity_torques': (e['g_act'] < 1e-2, f"{e['g_act']:.1e} N m"),
            }
        else:
            report[f'{robot_name} kinematics'] = {'pinocchio_vs_mujoco': (False, 'check_pinocchio_vs_mujoco.py wrote no result')}

    # The first run of each robot generates its acados solver if needed; build them before going parallel.
    first = [next(t for t in tasks if TASKS[t].get('robot', DEFAULT_ROBOT) == r) for r in ROBOTS
             if any(TASKS[t].get('robot', DEFAULT_ROBOT) == r for t in tasks)]
    rest = [t for t in tasks if t not in first]
    results = [run_task(name, out) for name in first]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results += list(pool.map(lambda n: run_task(n, out), rest))
    for name, wall, res in results:
        report[f'{name} ({wall:.0f} s wall)'] = res

    for section, res in report.items():
        print(f"\n[{section}]")
        for check, (ok, detail) in res.items():
            failed |= not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {check}: {detail}")
    print(f"\nlogs: {out}\n{'FAILED' if failed else 'ALL PASSED'}")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
