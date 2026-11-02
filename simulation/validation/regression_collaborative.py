"""Regression checks for the standalone PACC simulation and the two-robot collaborative carrying.

    python simulation/validation/regression_collaborative.py [--skip-closed-loop] [--jobs 5] [--out DIR]

Model checks run in-process (seconds); the closed-loop scenarios run in parallel subprocesses
through `collaborative_carrying_sim.py --export` (about 1.5 min). Thresholds sit around twice the
values observed when the baseline was recorded (README.md); they flag a change in behaviour. Exit code 1 on failure.
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
sys.path.insert(0, REPO)
import mujoco

from simulation.pacc_models import (ROBOT_MODELS, ARM_PRESETS, PAYLOAD_PRESETS, arm_joint_names, estimator_rest_position,
                                    load_arm_params, payload_setup, resolve_arm_params, set_arm_qpos)
from simulation import pacc_standalone_sim as pacc
from simulation.collaborative_scene import build_collaborative_model, link_tensions, PAYLOADS, ROLES

CC_SIM = os.path.join(REPO, 'simulation', 'collaborative_carrying_sim.py')
LEG_ACTUATORS = lambda role: [f'{role}/{leg}_{j}' for leg in ('FL', 'FR', 'RL', 'RR') for j in ('hip', 'thigh', 'calf')]


# ---------------------------------------------------------------------------------------------- standalone PACC
def _settle_and_impulse(params=None, mass=None, impulse=(60, 60, 0), arms=pacc.ROLES):
    model = pacc.build_model(arms, params, mass)
    log = pacc.simulate(model, arms=arms, duration=10.0, impulse=impulse, impulse_start=4.0, impulse_duration=0.2)
    t, q = log['time'], log[f'{arms[0]}_q']
    q_eq = q[(t > 3) & (t < 4)].mean(0)
    return log, q_eq, np.abs(q[t > 4.2] - q_eq).max(0)


def check_standalone():
    out = {}
    model = pacc.build_model()
    out['compiles'] = (model.nq == 27 and model.ntendon == 2,
                       f"nq {model.nq}, 2 arms x 3 passive joints, payload {model.body('mountpoint').mass[0]:.1f} kg, {model.ntendon} rope tendons")
    base, q_eq, dev = _settle_and_impulse()
    out['all_joints_move'] = (dev.min() > 0.02, f"peak deviation after a lateral impulse {dev.round(3)} rad (observed [0.27, 0.06, 0.34])")
    t = base['time']
    weight = model.body('mountpoint').mass[0] * 9.806
    tension = base['follower_rope_force'][(t > 3) & (t < 4)].mean() + base['leader_rope_force'][(t > 3) & (t < 4)].mean()
    out['ropes_carry_payload'] = (abs(tension - weight) < 0.02 * weight, f"rope tensions {tension:.1f} N, payload weight {weight:.1f} N")
    _, q_k, _ = _settle_and_impulse({'follower': dict(stiffness=[20, 50, 0.002])})
    out['stiffness_effect'] = (q_k[1] < 0.6 * q_eq[1], f"follower q2 at rest {q_eq[1]:.3f} -> {q_k[1]:.3f} rad with stiffness x2")
    _, q_m, _ = _settle_and_impulse(mass=20.0)
    out['payload_mass_effect'] = (q_m[1] > 1.5 * q_eq[1], f"follower q2 at rest {q_eq[1]:.3f} -> {q_m[1]:.3f} rad with 10 -> 20 kg")
    _, _, dev_lo = _settle_and_impulse({'follower': dict(damping=[1.05, 2.5, 1.0])})
    _, _, dev_hi = _settle_and_impulse({'follower': dict(damping=[12.6, 30, 12])})
    out['damping_effect'] = (dev_lo[0] > dev[0] > dev_hi[0], f"joint-1 peak deviation {dev_lo[0]:.3f} > {dev[0]:.3f} > {dev_hi[0]:.3f} rad for damping x0.25, x1, x3")
    _, q_s, _ = _settle_and_impulse({'follower': dict(springref=[0, 0.3, 0])})
    out['springref_effect'] = (q_s[1] - q_eq[1] > 0.2, f"follower q2 at rest {q_eq[1]:.3f} -> {q_s[1]:.3f} rad with springref q2 = 0.3")
    for arm in pacc.ROLES:
        log, _, _ = _settle_and_impulse(arms=(arm,), impulse=(20, 20, 0))
        out[f'one_arm_{arm}'] = (np.isfinite(log['payload_pos']).all() and log['payload_pos'][-1, 2] > 0.05,
                                 f"payload z at end {log['payload_pos'][-1, 2]:.3f} m")
    long = pacc.simulate(model, duration=60.0, impulse=(0, 0, 0))
    calm = long['time'] > 50
    out['stable_60s'] = (np.isfinite(long['payload_pos']).all() and np.abs(long['follower_qd'][calm]).max() < 1e-3,
                         f"max |qd| over 50-60 s {np.abs(long['follower_qd'][calm]).max():.1e} rad/s")
    return out


# ------------------------------------------------------------------------------------------------- follower model
def check_follower():
    m = mujoco.MjModel.from_xml_path(f'{ROBOT_MODELS}/aliengo_follower/aliengo_follower_actuated.xml')
    d = mujoco.MjData(m)
    name = lambda t, i: mujoco.mj_id2name(m, t, i)
    acts = [name(mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    act_joints = [name(mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) for i in range(m.nu)]
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    eef_trunk = d.xmat[1].reshape(3, 3).T @ (d.site('eef').xpos - d.xpos[1])
    base = m.body('follower/pa_link_0_ext')
    out = {
        'actuators': (acts == LEG_ACTUATORS('follower') and act_joints == [a + '_joint' for a in acts], f"{m.nu} motors on follower/* leg joints"),
        'arm_joints': ([name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)][-3:] == arm_joint_names('follower')
                       and m.jnt_axis[-3:].tolist() == [[0, 0, 1], [0, 1, 0], [0, 1, 0]], "joint 1 about z, joints 2-3 about y"),
        'sensors_keyframe': (sorted(name(mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)) == ['follower/imu_gyro', 'imu'] and m.nkey == 1,
                             f"sensors {[name(mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)]}, keyframe {name(mujoco.mjtObj.mjOBJ_KEY, 0)}"),
        'front_mounted': (m.body_pos[base.id][0] > 0.2 and np.allclose(m.body_quat[base.id], [1, 0, 0, 0]) and eef_trunk[0] > 0.5,
                          f"arm base at x {m.body_pos[base.id][0]:.3f} m, unrotated; eef in trunk frame {eef_trunk.round(4)}"),
        'impedance': (np.allclose(m.jnt_stiffness[-3:], [5, 8, 5]) and np.allclose(m.dof_damping[-3:], [0.5, 1.5, 0.5]) and np.allclose(m.qpos_spring[-3:], 0),
                      f"k {m.jnt_stiffness[-3:]}, d {m.dof_damping[-3:]}, springref {m.qpos_spring[-3:]}"),
    }
    for scene in ('flat', 'flat_green', 'ramp', 'stairs', 'rough'):
        ms = mujoco.MjModel.from_xml_path(f'{ROBOT_MODELS}/aliengo_follower/scene_{scene}.xml')
        out[f'scene_{scene}'] = (ms.nu == 12 and ms.nsensor == 2 and ms.nkey >= 1, f"nu {ms.nu}, nsensor {ms.nsensor}, nkey {ms.nkey}")
    return out


def check_follower_kinematics(out_dir):
    res = os.path.join(out_dir, 'kinematics_follower.json')
    subprocess.run([sys.executable, os.path.join(REPO, 'simulation', 'validation', 'check_pinocchio_vs_mujoco.py'),
                    f'{ROBOT_MODELS}/aliengo_follower/new_urdf_export.xml', '--role', 'follower', '--out', res],
                   cwd=REPO, capture_output=True, text=True)
    if not os.path.isfile(res):
        return {'pinocchio_vs_mujoco': (False, 'no result')}
    k = json.load(open(res))
    e = k['max_error']
    return {'eef_parent': (k['eef_parent_joint_pin'] == 'follower/arm_link_3_joint', k['eef_parent_joint_pin']),
            'positions_mass_matrix': (max(e['eef'], e['all_bodies'], e['M_act']) < 1e-9,
                                      f"eef {e['eef']:.1e} m, bodies {e['all_bodies']:.1e} m, M {e['M_act']:.1e}")}


# --------------------------------------------------------------------------------------------- two-robot model
def check_composition():
    out = {}
    params = {'leader': dict(stiffness=[6, 9, 4], damping=[0.4, 1.2, 0.6]), 'follower': dict(stiffness=[3.5, 8.47, 2.75], damping=[0.26, 1.43, 0.05])}
    m, info = build_collaborative_model(arm_params=params)
    d = mujoco.MjData(m)
    name = lambda t, i: mujoco.mj_id2name(m, t, i)
    acts = [name(mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    joints = [n for n in (name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)) if n]
    free = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    out['layout'] = (m.nq == 51 and m.nu == 24 and free == [0, 22, 44], f"nq {m.nq}, nu {m.nu}, free joints at qpos {free} (leader, follower, payload)")
    out['namespaces'] = (acts == LEG_ACTUATORS('leader') + LEG_ACTUATORS('follower') and len(set(joints)) == len(joints)
                         and sorted(name(mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)) == ['imu_follower', 'imu_gyro_follower', 'imu_gyro_leader', 'imu_leader'],
                         f"24 motors leader/* then follower/*, {len(joints)} unique named joints, sensors renamed per role")
    k = {r: m.jnt_stiffness[[m.joint(n).id for n in arm_joint_names(r)]] for r in ROLES}
    c = {r: m.dof_damping[m.jnt_dofadr[[m.joint(n).id for n in arm_joint_names(r)]]] for r in ROLES}
    out['independent_arm_params'] = (all(np.allclose(k[r], params[r]['stiffness']) and np.allclose(c[r], params[r]['damping']) for r in ROLES),
                                     f"leader k {k['leader']}, follower k {k['follower']}")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    tendon_ids = [m.tendon(f'link_eef_{r}').id for r in ROLES]
    out['payload_default'] = (abs(m.body('mountpoint').mass[0] - 2.0) < 1e-9 and m.neq == 0 and m.ntendon == 2
                              and np.allclose(m.tendon_stiffness[tendon_ids], 15) and np.allclose(m.tendon_range[tendon_ids], [0, 0.15]),
                              f"{m.body('mountpoint').mass[0]:.1f} kg payload hung by {m.ntendon} tendons, k {m.tendon_stiffness[tendon_ids]} N/m, "
                              f"range {m.tendon_range[tendon_ids][0]} m")
    body = lambda g: name(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    extra = [(body(cn.geom1), body(cn.geom2)) for cn in d.contact[:d.ncon] if 'world' not in (body(cn.geom1), body(cn.geom2))]
    rest = m.tendon_lengthspring[tendon_ids]
    out['initialization'] = (not extra and np.all(d.ten_length[tendon_ids] < 0.15) and np.allclose(rest[:, 1], d.ten_length[tendon_ids])
                             and np.allclose(rest[:, 0], 0),
                             f"tendons {d.ten_length[tendon_ids].round(3)} m (< 0.15), spring deadband [0, spawn length]; non-ground contacts {extra}")
    out['force_transmission_tendons'] = _hold_test_tendons(m)
    bar, _ = build_collaborative_model(payload='rigid_bar')
    db = mujoco.MjData(bar)
    mujoco.mj_resetDataKeyframe(bar, db, 0)
    mujoco.mj_forward(bar, db)
    viol = np.abs(db.efc_pos[db.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]).max()
    out['rigid_bar_variant'] = (bar.neq == 2 and viol < 1e-6, f"{bar.neq} connect constraints, initial violation {viol:.1e} m")
    out['force_transmission_rigid_bar'] = _force_transmission(bar)
    el, _ = build_collaborative_model(payload='elastic')
    out['elastic_variant'] = (el.ntendon == 1 and el.nq == 44 and np.isclose(el.tendon_stiffness[0], 10) and np.allclose(el.tendon_range[0], [0, 0.4]),
                              f"one EEF-EEF tendon, k {el.tendon_stiffness[0]} N/m, range {el.tendon_range[0]} m, no payload body (nq {el.nq})")
    return out


def _hold(m):
    """Data at the keyframe with both robots' bases and legs held there at every step."""
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    hold = np.r_[0:19, 22:41]
    q_hold = d.qpos[hold].copy()

    def step(n, force=np.zeros(3), body=None):
        for _ in range(n):
            if body is not None:
                d.xfrc_applied[body, :3] = force
            mujoco.mj_step(m, d)
            d.qpos[hold], d.qvel[np.r_[0:18, 21:39]] = q_hold, 0.0
    return d, step


def _hold_test_tendons(m):
    """Bases held: the tendons carry the payload's weight; a 20 N push along +x moves the payload by a
    bounded amount, changes the tensions and deflects both arms."""
    d, step = _hold(m)
    pid = m.body('mountpoint').id
    arm_q = lambda: {r: d.qpos[[m.jnt_qposadr[m.joint(n).id] for n in arm_joint_names(r)]].copy() for r in ROLES}
    step(2500)
    t0, q0, x0 = link_tensions(m, d), arm_q(), d.xpos[pid].copy()
    step(2500, np.array([20.0, 0, 0]), pid)
    t1, q1, x1 = link_tensions(m, d), arm_q(), d.xpos[pid].copy()
    weight = m.body('mountpoint').mass[0] * 9.81
    dq = {r: np.abs(q1[r] - q0[r]).max() for r in ROLES}
    lengths = d.ten_length[[m.tendon(f'link_eef_{r}').id for r in ROLES]]
    ok = (abs(sum(t0.values()) - weight) < 0.25 * weight and abs(x1[0] - x0[0]) < 0.3 and min(dq.values()) > 0.01
          and np.all(lengths <= 0.15 + 0.01))  # MuJoCo length limits are soft (solref 0.02 s): ~5 mm under load
    return (ok, f"tensions at rest {t0['leader']:.1f} + {t0['follower']:.1f} N (payload weight {weight:.1f} N, ropes inclined); "
                f"+20 N push: payload moves {x1[0] - x0[0]:.3f} m, tensions {t1['leader']:.1f} / {t1['follower']:.1f} N, "
                f"arm deflection {dq['leader']:.3f} / {dq['follower']:.3f} rad, tendon lengths {lengths.round(3)} m (limit 0.15, soft)")


def _force_transmission(m):
    """Hold both robots' bases and legs at the keyframe; the payload hangs on the two arms. A 20 N push
    on the payload along +x must come out through the two connections and deflect both arms."""
    d, hold_step = _hold(m)
    pid = m.body('mountpoint').id
    eq = {r: m.equality(f'connect_eef_{r}').id for r in ROLES}
    step = lambda n, force: hold_step(n, force, pid)

    def state():
        f = {r: d.efc_force[(d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == eq[r])][:3] for r in ROLES}
        return f, {r: d.qpos[[m.jnt_qposadr[m.joint(n).id] for n in arm_joint_names(r)]].copy() for r in ROLES}
    step(2500, np.zeros(3))
    f0, q0 = state()
    step(2500, np.array([20.0, 0, 0]))
    f1, q1 = state()
    dfx = sum(f1[r][0] - f0[r][0] for r in ROLES)
    dq = {r: np.abs(q1[r] - q0[r]).max() for r in ROLES}
    weight = sum(np.linalg.norm(f0[r]) for r in ROLES)
    return (abs(abs(dfx) - 20.0) < 2.0 and min(dq.values()) > 0.01,
            f"static connection forces {weight:.1f} N (payload weight {m.body('mountpoint').mass[0] * 9.81:.1f} N); "
            f"+20 N on the payload changes them by {abs(dfx):.1f} N in x; arm deflection leader {dq['leader']:.3f}, follower {dq['follower']:.3f} rad")


# ---------------------------------------------------------------------------------- payload and impedance presets
def _arm_ids(m, role):
    return [m.joint(n).id for n in arm_joint_names(role)]


def check_presets():
    from quadruped_pympc.interfaces.pacc_interface import PassiveArmInterface
    out = {}
    # 'model' = the robot MJCFs' own arm impedance
    own = {}
    for role in ROLES:
        mr = mujoco.MjModel.from_xml_path(f'{ROBOT_MODELS}/aliengo_{role}/new_urdf_export.xml')
        ids = _arm_ids(mr, role)
        own[role] = (mr.jnt_stiffness[ids], mr.dof_damping[mr.jnt_dofadr[ids]], mr.qpos_spring[mr.jnt_qposadr[ids]])
    p = ARM_PRESETS['model']
    out['model_preset_is_robot_mjcf'] = (all(np.allclose(own[r][0], p[r]['stiffness']) and np.allclose(own[r][1], p[r]['damping'])
                                             and np.allclose(own[r][2], p[r]['springref']) for r in ROLES),
                                         f"aliengo_leader/follower new_urdf_export.xml k {own['leader'][0]}, d {own['leader'][1]}")
    # every payload preset: selection, compiled parameters, static equilibrium with the bases held
    for name, preset in PAYLOAD_PRESETS.items():
        payload, mass, arm_preset = payload_setup(name)
        m, info = build_collaborative_model(payload=payload, payload_mass=mass, arm_preset=arm_preset)
        want = ARM_PRESETS[arm_preset]
        got = info['arm_params']
        applied = all(np.allclose(got[r][k], want[r][k]) for r in ROLES for k in ('stiffness', 'damping', 'springref', 'link3_mass', 'eef_mass'))
        coupling = (m.neq == 2) if payload == 'rigid_bar' else (m.ntendon == 2 and m.neq == 0)
        out[f'{name}_selected'] = ((payload, mass, arm_preset) == (preset['payload'], preset['payload_mass'], preset['arm_preset'])
                                   and abs(m.body('mountpoint').mass[0] - mass) < 1e-9 and applied and coupling,
                                   f"{payload}, {m.body('mountpoint').mass[0]:.1f} kg, arms '{arm_preset}' k {got['leader']['stiffness']} "
                                   f"d {got['leader']['damping']} (both robots), {preset['status']}")
        out[f'{name}_static'] = _static_equilibrium(m)
    # independence and spring reference / initial angles, and the estimator's rest position
    params = {'leader': dict(springref=[0, 0.3, -0.2], q0=[0.05, 0.1, -0.1], stiffness=[6, 9, 4]),
              'follower': dict(springref=[0, 0.2, 0.4], q0=[-0.05, 0.2, 0.1], stiffness=[3.5, 8.47, 2.75])}
    m, info = build_collaborative_model(payload='rigid_bar', arm_params=params)  # no tendon forces on the arms
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    ok_ref = all(np.allclose(m.qpos_spring[m.jnt_qposadr[_arm_ids(m, r)]], params[r]['springref'])
                 and np.allclose(d.qpos[m.jnt_qposadr[_arm_ids(m, r)]], params[r]['q0']) for r in ROLES)
    out['springref_q0_per_robot'] = (ok_ref, f"springref leader {params['leader']['springref']}, follower {params['follower']['springref']}; "
                                              f"keyframe arm q = q0 for each robot")
    rng = np.random.default_rng(0)
    err = 0.0
    for r in ROLES:
        g = info['arm_params'][r]
        est = PassiveArmInterface(spring_gains=np.array(g['stiffness']), robot=f'aliengo_{r}', joint_pos0=estimator_rest_position(r, g['springref']),
                                  mass_matrix=np.array([0.554, 0.554, g['link3_mass']]), damping_gains=np.array(g['damping']), com_offset=np.zeros(3))
        dofs = m.jnt_dofadr[_arm_ids(m, r)]
        sign = np.array([1, 1, -1]) if r == 'follower' else np.ones(3)  # estimator coordinates (q1, |q2|, -q3) for the follower
        for _ in range(20):
            q = rng.uniform([-0.5, 0.05, -0.8], [0.5, 0.8, 0.3])
            mujoco.mj_resetDataKeyframe(m, d, 0)
            set_arm_qpos(m, d, r, q)
            mujoco.mj_forward(m, d)
            qe = q.copy()
            est.updateJacobian(np.zeros((3, 3)), qe, est.p1, est.p2, est.p3)
            err = max(err, np.abs(est.updateSpringTorque(qe, est.spring_gains) - sign * d.qfrc_spring[dofs]).max())
    out['estimator_spring_matches_mujoco'] = (err < 1e-9, f"wrench-estimator spring torque vs MuJoCo with springref and per-robot k: max error {err:.1e} N m")
    # the standalone sim takes the same parameter sets
    sa = pacc.build_model(arm_params=resolve_arm_params('original_dual'), payload_mass=PAYLOAD_PRESETS['heavy_7kg']['payload_mass'])
    sa_ok = all(np.allclose(pacc.compiled_arm_params(sa, r)[k], ARM_PRESETS['original_dual'][r][k])
                for r in ROLES for k in ('stiffness', 'damping', 'springref', 'link3_mass', 'eef_mass'))
    out['standalone_uses_presets'] = (sa_ok and abs(sa.body('mountpoint').mass[0] - 7.0) < 1e-9,
                                      f"arms-on-blocks with 'original_dual': {pacc.compiled_arm_params(sa, 'leader')}; bar {sa.body('mountpoint').mass[0]:.1f} kg")
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump({'follower': {'stiffness': [3.5, 8.47, 2.75]}}, f)
    from_json = resolve_arm_params('model', load_arm_params(f.name))
    os.unlink(f.name)
    try:
        resolve_arm_params('model', {'leader': {'stifness': [1, 1, 1]}})
        typo_caught = False
    except KeyError:
        typo_caught = True
    out['json_format'] = (from_json['follower']['stiffness'] == [3.5, 8.47, 2.75] and from_json['leader'] == ARM_PRESETS['model']['leader']
                          and typo_caught, "JSON overrides one robot only; unknown keys rejected")
    return out


def _static_equilibrium(m):
    """Bases and legs held at the keyframe for 5 s: the arms settle inside their joint limits, the links carry
    the payload's weight and the payload hangs clear of the ground."""
    d, step = _hold(m)
    step(2500)
    weight = m.body('mountpoint').mass[0] * 9.81
    tension = link_tensions(m, d)
    q, qd, lim = {}, 0.0, True
    for r in ROLES:
        ids = _arm_ids(m, r)
        q[r] = d.qpos[m.jnt_qposadr[ids]]
        qd = max(qd, np.abs(d.qvel[m.jnt_dofadr[ids]]).max())
        rng = m.jnt_range[ids]
        lim &= all(not m.jnt_limited[j] or (lo + 0.02 < x < hi - 0.02) for j, x, (lo, hi) in zip(ids, q[r], rng))
    z = d.xpos[m.body('mountpoint').id][2]
    ok = lim and qd < 1e-2 and sum(tension.values()) > 0.9 * weight and z > 0.1
    return (ok, f"arm q leader {q['leader'].round(3)}, follower {q['follower'].round(3)} rad (inside limits); |qd| {qd:.0e}; "
                f"link forces {tension['leader']:.1f} + {tension['follower']:.1f} N for {weight:.1f} N weight; payload z {z:.3f} m")


# --------------------------------------------------------------------------------------------- guidance mapping
def check_guidance():
    from quadruped_pympc.interfaces.pacc_interface import PassiveArmInterface
    from simulation.collaborative_carrying_sim import guidance_input
    g = PassiveArmInterface(spring_gains=np.ones(3), robot='aliengo_follower', joint_pos0=np.zeros(3),
                            mass_matrix=np.ones(3), damping_gains=np.ones(3), com_offset=np.zeros(3))
    deg = np.pi / 180

    def steady(dq1, dq3):
        v = w = 0.0
        for _ in range(30):  # the first-order filter converges to the level
            lin, ang = g.compute_reference_velocity(np.array([dq1, 0.0, dq3]), v, w)
            v, w = lin[0], ang[2]
        return float(round(v, 3)), float(round(w, 3))
    cases = {(0, 5): (0, 0), (0, -15): (0.1, 0), (0, -25): (0.2, 0), (0, 15): (0.1, 0), (0, 25): (0.15, 0),
             (15, 0): (0, 0.4), (-15, 0): (0, -0.4), (25, 0): (0, 0.5), (-25, 0): (0, -0.5), (5, 5): (0, 0)}
    got = {k: steady(k[0] * deg, k[1] * deg) for k in cases}
    lin, _ = g.compute_reference_velocity(np.array([0, 0, -15 * deg]), 0.0, 0.0)
    return {'levels_and_dead_zones': (all(np.allclose(got[k], cases[k], atol=1e-3) for k in cases),
                                      'steady commands (vx, wz) for (dq1, dq3) deg: ' + ', '.join(f"{k}: {got[k]}" for k in cases)),
            'low_pass_filter': (abs(lin[0] - 0.05) < 1e-9, f"first step from rest to the 0.1 level gives {lin[0]:.3f} (alpha 0.5)"),
            'input_convention': (np.allclose(guidance_input(np.array([0.1, -0.2, 0.3])), [0.1, 0.2, -0.3]),
                                 "(q1, q2, q3) -> (q1, |q2|, -q3), as the original pipeline fed it")}


# ------------------------------------------------------------------------------------------------- closed loop
SCENARIOS = {
    'stand': [], 'straight': [], 'stop': [], 'turn': [], 'turn_right': [], 'turn_stop': [],
    'impedance': ['--scenario', 'straight', '--stiffness-follower', '3.5,8.47,2.75', '--damping-follower', '0.26,1.43,0.05'],
    'rigid_bar': ['--scenario', 'straight', '--payload', 'rigid_bar'],
    'elastic': ['--scenario', 'straight', '--payload', 'elastic'],
    # experimental payload presets: only the scenarios they pass (pacc_models.PAYLOAD_PRESETS)
    **{f'{p}_{sc}': ['--payload-preset', p, '--scenario', sc] for p, v in PAYLOAD_PRESETS.items()
       if v['status'] != 'validated' for sc in v['scenarios']},
}


def run_closed_loop(name, out):
    path = os.path.join(out, f'cc_{name}.npz')
    args = SCENARIOS[name] or ['--scenario', name]
    t0 = time.time()
    proc = subprocess.run([sys.executable, CC_SIM, *args, '--headless', '--export', path], cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.isfile(path):
        return name, time.time() - t0, {'run': (False, (proc.stderr.strip().splitlines() or ['no output'])[-1])}
    d = np.load(path)
    meta = json.loads(str(d['metadata']))
    return name, time.time() - t0, check_run(name, {k: d[k] for k in d.files if k != 'metadata'}, meta)


def _span_velocity(t, p, t0, t1):
    i0, i1 = np.searchsorted(t, t0), min(np.searchsorted(t, t1), len(t) - 1)
    return (p[i1, :2] - p[i0, :2]) / (t[i1] - t[i0])


def _connected(d, meta, late):
    """The coupling holds: rigid bar at its length; tendons within their length limits (EEF distance
    bounded by mount points + two tendon lengths) and the payload off the ground."""
    dist = d['distance_eef'][late]
    p = PAYLOADS[meta['payload']]
    if p['coupling'] == 'connect':
        return (np.abs(dist - 0.28).max() < 0.005, f"EEF distance {dist.min():.3f}-{dist.max():.3f} m (bar 0.28 m)")
    if not p['file']:
        return (dist.max() < p['max_length'] + 0.005, f"EEF distance {dist.min():.3f}-{dist.max():.3f} m (tendon limit {p['max_length']} m)")
    z = d['payload_pos'][late, 2]
    bound = 0.4 + 2 * p['max_length'] + 0.005
    return (dist.max() < bound and z.min() > 0.1,
            f"EEF distance {dist.min():.3f}-{dist.max():.3f} m (< {bound:.3f} with taut tendons); payload z >= {z.min():.3f} m")


def check_run(name, d, meta):
    t = d['time']
    late = t > 1.0
    pL, pF = d['leader_base_pos'], d['follower_base_pos']
    oL, oF = d['leader_base_ori'], d['follower_base_ori']
    att = max(np.abs(oL[late, :2]).max(), np.abs(oF[late, :2]).max())
    out = {
        'completed_finite': (t[-1] >= meta['duration'] - 0.01 and all(np.isfinite(v).all() for k, v in d.items()
                                                                       if v.dtype != object and not k.startswith('payload_')),
                             f"{t[-1]:.1f} s"),
        'solver_status': (all(np.isin(d[f'{r}_mpc_status'], (0, 2)).all() for r in ROLES), 'status 0 or 2 for both robots'),
        'payload_connected': _connected(d, meta, late),
    }
    out['no_mujoco_reset'] = (meta.get('unstable_at') is None, f"MuJoCo instability at {meta.get('unstable_at')}")
    if meta.get('payload_preset') in PAYLOAD_PRESETS and meta['payload_preset'] != 'light_2kg':
        return {**out, **_check_preset_run(meta['payload_preset'], meta['scenario'], d, meta, att)}
    if name == 'stand':
        f0 = max(d['link_tension_leader'][:500].max(), d['link_tension_follower'][:500].max())
        out['stable'] = (att < 0.05, f"max |roll, pitch| {att:.3f} rad (observed 0.021)")
        out['no_follower_command'] = (np.abs(d['follower_cmd']).max() < 1e-9, f"max follower command {np.abs(d['follower_cmd']).max():.1e}")
        out['initial_link_force'] = (f0 < 80, f"max link tension over the first 1 s {f0:.1f} N (payload drops ~3 cm into the "
                                               f"ropes at spawn: observed 42 N; static 11 N per side)")
    if name in ('straight', 'impedance', 'rigid_bar', 'elastic'):
        vL, vF = _span_velocity(t, pL, 8, 24)[0], _span_velocity(t, pF, 8, 24)[0]
        gap = d['distance_base'][(t > 8)]
        lo = 0.07 if name == 'elastic' else 0.08  # direct EEF tendon: 0.085 observed
        out['follower_follows'] = (lo < vL < 0.12 and lo < vF < 0.12 and np.ptp(gap) < 0.2,
                                   f"x velocity 8-24 s leader {vL:.3f}, follower {vF:.3f} m/s; base spacing {gap.min():.2f}-{gap.max():.2f} m "
                                   f"(observed: elastic_payload 0.100 / 0.101, 1.81-1.89; rigid_bar 0.099 / 0.099, 1.48-1.58; elastic 0.085)")
        out['guidance_drives_follower'] = ((d['follower_cmd'][t > 8, 0] > 0).mean() > 0.9,
                                           f"follower forward command active {100 * (d['follower_cmd'][t > 8, 0] > 0).mean():.0f} % of 8-24 s")
        out['stable'] = (att < 0.15, f"max |roll, pitch| {att:.3f} rad (observed 0.072)")
        if name == 'impedance':
            k = meta['arm_params']['follower']['stiffness']
            out['impedance_applied'] = (np.allclose(k, [3.5, 8.47, 2.75]) and np.allclose(meta['arm_params']['leader']['stiffness'], [5, 8, 5]),
                                        f"leader k {meta['arm_params']['leader']['stiffness']}, follower k {k}")
    if name == 'stop':
        end = t > t[-1] - 4
        gap = d['distance_base'][end]
        out['follower_stops'] = (np.abs(d['follower_cmd'][end]).max() < 1e-3 and np.linalg.norm(d['follower_base_lin_vel'][end, :2], axis=1).mean() < 0.01,
                                 f"last 4 s: follower command {np.abs(d['follower_cmd'][end]).max():.1e}, speed {np.linalg.norm(d['follower_base_lin_vel'][end, :2], axis=1).mean():.4f} m/s")
        out['spacing_settles'] = (np.ptp(gap) < 0.01, f"base spacing over the last 4 s {gap.min():.3f}-{gap.max():.3f} m")
        out['arm_oscillation_decays'] = (np.abs(d['follower_arm_qd'][end]).max() < 0.05 and np.abs(d['leader_arm_qd'][end]).max() < 0.05,
                                         f"max |arm qd| last 4 s leader {np.abs(d['leader_arm_qd'][end]).max():.3f}, follower {np.abs(d['follower_arm_qd'][end]).max():.3f} rad/s")
        out['stable'] = (att < 0.15, f"max |roll, pitch| {att:.3f} rad")
    if name in ('turn', 'turn_right'):
        # steady state (40-60 s): the follower turns with the leader, mostly through its own yaw command
        sign = 1 if name == 'turn' else -1
        w = t > 40
        wzL, wzF = d['leader_base_ang_vel'][w, 2].mean(), d['follower_base_ang_vel'][w, 2].mean()
        cmd = d['follower_cmd'][w, 1]
        share = np.abs(cmd).sum() / max(np.abs(d['follower_base_ang_vel'][w, 2]).sum(), 1e-9)
        out['follower_turns_with_leader'] = (sign * wzL > 0.06 and sign * wzF > 0.7 * sign * wzL,
                                             f"yaw rate 40-60 s leader {wzL:+.3f}, follower {wzF:+.3f} rad/s "
                                             f"(observed left +0.081 / +0.066, right -0.081 / -0.079)")
        out['guidance_commands_heading'] = (sign * cmd.mean() > 0.02 and share > 0.4 and (sign * cmd >= -1e-9).all(),
                                            f"follower yaw command mean {cmd.mean():+.3f} rad/s, active {100 * (np.abs(cmd) > 1e-3).mean():.0f} %, "
                                            f"{100 * share:.0f} % of the follower's yaw, never of the wrong sign "
                                            f"(observed left +0.045, 65 %; right -0.061, 74 %)")
        out['stable'] = (att < 0.3, f"max |roll, pitch| {att:.3f} rad (observed 0.072)")
    if name == 'turn_stop':
        stop_t = meta['leader_command'][-1][1]
        yawL, yawF = np.unwrap(oL[:, 2]), np.unwrap(oF[:, 2])
        after, end = t > stop_t + 2, t > t[-1] - 6
        rel = yawL - yawF
        rate = abs(rel[end][-1] - rel[end][0]) / 6
        out['follower_command_returns_to_zero'] = (np.abs(d['follower_cmd'][after]).max() < 1e-3,
                                                   f"max follower command from {stop_t + 2:.0f} s: {np.abs(d['follower_cmd'][after]).max():.1e}")
        out['formation_settles'] = (rate < 0.03 and np.ptp(d['distance_base'][end]) < 0.1,
                                    f"last 6 s: relative heading drifts {rate:.3f} rad/s (leader arm spring unwinding; observed 0.019), "
                                    f"spacing range {np.ptp(d['distance_base'][end]):.3f} m")
        out['stable'] = (att < 0.3, f"max |roll, pitch| {att:.3f} rad")
    return out


# Observed (stand / straight): medium_5kg max |roll, pitch| 0.033 / 0.235 rad, x velocity
# 0.119 / 0.116 m/s; heavy_7kg 0.047 / 0.137 rad, 0.134 / 0.133 m/s (both faster than the 0.1 m/s command).
PRESET_EXPECTED = {'medium_5kg': dict(stand_att=0.033, att=0.235), 'heavy_7kg': dict(stand_att=0.047, att=0.137)}


def _check_preset_run(preset, scenario, d, meta, att):
    t, out = d['time'], {}
    exp = PRESET_EXPECTED[preset]
    p = PAYLOAD_PRESETS[preset]
    out['preset_applied'] = (meta['payload'] == p['payload'] and abs(meta['payload_mass'] - p['payload_mass']) < 1e-9
                             and meta['arm_params']['leader']['stiffness'] == ARM_PRESETS[p['arm_preset']]['leader']['stiffness'],
                             f"{meta['payload']} {meta['payload_mass']} kg, arms '{p['arm_preset']}' k {meta['arm_params']['leader']['stiffness']}")
    if scenario == 'stand':
        out['stable'] = (att < 2 * exp['stand_att'] and np.abs(d['follower_cmd']).max() < 1e-9,
                         f"max |roll, pitch| {att:.3f} rad (observed {exp['stand_att']}); follower command 0")
    if scenario == 'straight':
        vL, vF = _span_velocity(t, d['leader_base_pos'], 8, 24)[0], _span_velocity(t, d['follower_base_pos'], 8, 24)[0]
        gap = d['distance_base'][t > 8]
        out['follower_follows'] = (0.08 < vL < 0.16 and 0.08 < vF < 0.16 and np.ptp(gap) < 0.3,
                                   f"x velocity 8-24 s leader {vL:.3f}, follower {vF:.3f} m/s; spacing {gap.min():.2f}-{gap.max():.2f} m")
        out['stable'] = (att < min(1.6 * exp['att'], 0.45), f"max |roll, pitch| {att:.3f} rad (observed {exp['att']})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-closed-loop', action='store_true')
    ap.add_argument('--jobs', type=int, default=len(SCENARIOS))
    ap.add_argument('--out')
    args = ap.parse_args()
    out = args.out or tempfile.mkdtemp(prefix='collaborative_regression_')
    os.makedirs(out, exist_ok=True)
    report = {}
    t0 = time.time()
    report['standalone PACC'] = check_standalone()
    report['aliengo_follower'] = {**check_follower(), **check_follower_kinematics(out)}
    report['two-robot composition'] = check_composition()
    report['payload and impedance presets'] = check_presets()
    report['follower guidance'] = check_guidance()
    report['__model_time'] = time.time() - t0
    if not args.skip_closed_loop:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for name, wall, res in pool.map(lambda n: run_closed_loop(n, out), SCENARIOS):
                report[f'closed loop: {name} ({wall:.0f} s wall)'] = res
    failed = False
    print(f"model checks: {report.pop('__model_time'):.0f} s")
    for section, res in report.items():
        print(f"\n[{section}]")
        for check, (ok, detail) in res.items():
            failed |= not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {check}: {detail}")
    print(f"\nlogs: {out}\n{'FAILED' if failed else 'ALL PASSED'}")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
