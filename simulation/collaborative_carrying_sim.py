"""Two-robot collaborative carrying: leader (rear PACC) commanded externally, follower (front PACC)
guided by its passive-arm deflection, a shared payload between the two end effectors.

    python simulation/collaborative_carrying_sim.py --scenario straight [--headless] [--export run.npz]
        [--payload-preset light_2kg|medium_5kg|heavy_7kg] [--payload elastic_payload|rigid_bar|elastic]
        [--payload-mass 2] [--terrain flat] [--arm-preset model|original_elastic|original_dual] [--arm-params arms.json]
        [--stiffness-leader 5,8,5] [--damping-leader 0.5,1.5,0.5]
        [--springref-leader 0,0,0] [--q0-leader 0,0,0] (same for --*-follower) [--leader-vx V] [--leader-wz W]

Scenarios (leader command, heading frame): stand, straight (0.1 m/s), stop (0.1 m/s then 0),
turn / turn_right / turn_slow (0.1 m/s with +0.1 / -0.1 / +0.05 rad/s), turn_stop (left turn, then 0). The scene is composed by simulation/collaborative_scene.py.

Payload and arms: a payload preset (simulation/pacc_models.py PAYLOAD_PRESETS; default light_2kg, the
validated 2 kg configuration) sets the coupling, the payload mass and the arm preset; --payload,
--payload-mass and --arm-preset replace one of them, and --arm-params (JSON) and the per-robot
options override single arm parameters on top.

Locomotion: both robots use the existing 'collaborative' controller (the PACC MPC of the original
dual-robot sim, quadruped_pympc_wrapper_collaborative), one wrapper per robot, each with its own
passive-arm wrench estimate. Follower guidance: PassiveArmInterface.compute_reference_velocity,
unchanged (joint-3 deflection -> forward velocity, joint-1 deflection -> yaw rate, dead zones,
levels and low-pass filter as implemented), measured from the follower arm pose calibrated once
the payload has settled.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from quadruped_pympc import config as cfg

cfg.mpc_params['type'] = 'collaborative'
cfg.simulation_params['dt'] = cfg.sim_dt_by_controller['collaborative']

import mujoco
from gym_quadruped.quadruped_env_coll import QuadrupedEnvColl
from gym_quadruped.utils.quadruped_utils import LegsAttr
from gym_quadruped.utils.quadruped_utils_collab import LegsAttrTwoRbots
from quadruped_pympc.interfaces.pacc_interface import PassiveArmInterface
from quadruped_pympc.quadruped_pympc_wrapper_collaborative import QuadrupedPyMPC_Wrapper

try:
    from simulation.collaborative_scene import build_collaborative_model, link_tensions, ROLES
    from simulation.pacc_models import (arm_state, estimator_rest_position, load_arm_params, payload_setup,
                                        ARM_PRESETS, PAYLOAD_PRESETS)
except ImportError:  # repo root not on sys.path
    from collaborative_scene import build_collaborative_model, link_tensions, ROLES
    from pacc_models import (arm_state, estimator_rest_position, load_arm_params, payload_setup,
                             ARM_PRESETS, PAYLOAD_PRESETS)

SCENARIOS = {  # leader command segments (t0, t1, vx, wz), heading frame
    'stand': dict(duration=10.0, leader=[]),
    'straight': dict(duration=24.0, leader=[(4.0, 24.0, 0.1, 0.0)]),
    'stop': dict(duration=26.0, leader=[(4.0, 14.0, 0.1, 0.0)]),
    # Turns last 56 s: a ~1.8 m train entering a 1 m radius turn needs ~30 s of pursuit transient before the
    # follower turns at the leader's rate; steady state is evaluated over 40-60 s.
    'turn': dict(duration=60.0, leader=[(4.0, 60.0, 0.1, 0.1)]),  # left
    'turn_right': dict(duration=60.0, leader=[(4.0, 60.0, 0.1, -0.1)]),
    'turn_slow': dict(duration=60.0, leader=[(4.0, 60.0, 0.1, 0.05)]),
    'turn_stop': dict(duration=60.0, leader=[(4.0, 44.0, 0.1, 0.1)]),  # left turn, then the leader stops
    # demonstration: straight, a left turn, stop
    'showcase': dict(duration=30.0, leader=[(3.0, 11.0, 0.1, 0.0), (11.0, 23.0, 0.1, 0.1)]),
}
RENDER_FREQ = 30  # Hz
GUIDANCE_START = 3.0  # calibrate the follower arm's neutral pose once the payload has settled


def guidance_input(q):
    """Follower arm angles as the original guidance received them. In the original two-robot sim the same array
    went to PassiveArmInterface.calculate_force_estimates_damping first, whose updateJacobian mirrors
    the follower arm in place (q2 -> |q2|, q3 -> -q3), and then to compute_reference_velocity."""
    return np.array([q[0], abs(q[1]), -q[2]])
SUFFIX = {'leader': '1', 'follower': '2'}
LEGS = ('FL', 'FR', 'RL', 'RR')


def split(attr, role):
    """Per-robot LegsAttr from the two-robot LegsAttrTwoRbots."""
    return LegsAttr(**{leg: attr[leg + SUFFIX[role]] for leg in LEGS})


def leader_command(segments, t):
    for t0, t1, vx, wz in segments:
        if t0 <= t < t1:
            return np.array([vx, 0.0, 0.0]), np.array([0.0, 0.0, wz])
    return np.zeros(3), np.zeros(3)


def run(scenario='straight', terrain='flat', payload='elastic_payload', payload_mass=2.0, arm_params=None,
        duration=None, render=False, leader_vx=None, leader_wz=None, arm_preset='model', guidance=None,
        payload_preset=None, sim_dt=None):
    """payload_preset: recorded in the metadata only; resolve it with pacc_models.payload_setup."""
    sc = dict(SCENARIOS[scenario])
    guidance_params = dict(guidance or {})
    # optional overrides of the scenario's leader speed and yaw rate (segments with a nonzero value)
    sc['leader'] = [(t0, t1, leader_vx if (leader_vx is not None and vx) else vx,
                     leader_wz if (leader_wz is not None and wz) else wz) for t0, t1, vx, wz in sc['leader']]
    if duration and duration > sc['duration']:  # a command that lasts to the end keeps lasting
        sc['leader'] = [(t0, duration if t1 >= sc['duration'] else t1, vx, wz) for t0, t1, vx, wz in sc['leader']]
    duration = duration or sc['duration']
    model, info = build_collaborative_model(terrain=terrain, payload=payload, payload_mass=payload_mass,
                                            arm_params=arm_params, arm_preset=arm_preset)
    if sim_dt:  # before the wrappers are built: they read it from the config
        cfg.simulation_params['dt'] = sim_dt
    dt = cfg.simulation_params['dt']
    env = QuadrupedEnvColl(robot='aliengo_two_robots', hip_height=cfg.hip_height,
                           legs_joint_names={**info['leg_joints']['leader'], **info['leg_joints']['follower']},
                           feet_geom_name={**info['feet_geoms']['leader'], **info['feet_geoms']['follower']},
                           scene=f'composed_{terrain}', sim_dt=dt, base_vel_command_type='forward',
                           ground_friction_coeff=1.5, mj_model=model)
    env.reset(random=False)
    if render:
        env.render()
    m, d = env.mjModel, env.mjData
    trunk_id = {role: m.body(f'{role}/trunk').id for role in ROLES}
    has_payload = info['payload_mass'] > 0  # 'elastic' couples the end effectors without a payload body
    payload_id = m.body('mountpoint').id if has_payload else None
    nq_robots = m.joint('weight_joint').qposadr[0] if has_payload else m.nq  # robots' qpos, as the wrappers expect

    params = info['arm_params']
    obs_names = ('ref_base_height', 'ref_base_angles', 'ref_feet_pos', 'nmpc_GRFs', 'nmpc_footholds',
                 'swing_time', 'phase_signal', 'lift_off_positions')
    feet0 = env.feet_pos(frame='world')
    wrapper, estimator = {}, {}
    for role in ROLES:
        wrapper[role] = QuadrupedPyMPC_Wrapper(initial_feet_pos=split(feet0, role), legs_order=list(LEGS),
                                               quadrupedpympc_observables_names=obs_names)
        p = params[role]
        # wrench estimate: the arm's actual impedance and spring rest
        estimator[role] = PassiveArmInterface(spring_gains=np.array(p['stiffness'], float), robot=f'aliengo_{role}',
                                              joint_pos0=estimator_rest_position(role, p['springref']), mass_matrix=np.array([0.554, 0.554, p['link3_mass']]),
                                              damping_gains=np.array(p['damping'], float), com_offset=np.zeros(3))
    guidance = None  # PassiveArmInterface used only for compute_reference_velocity, neutral pose = calibrated
    jac_feet_prev = LegsAttrTwoRbots(*[np.zeros((3, m.nv)) for _ in range(8)])
    tau = LegsAttrTwoRbots(*[np.zeros((m.nv, 1)) for _ in range(8)])
    jeef = {role: np.zeros((3, 3)) for role in ROLES}
    ref_prev = np.zeros(2)  # follower (vx, yaw rate), for the guidance's low-pass filter
    eq_rows = {}
    log = {k: [] for k in ('time', 'payload_pos', 'payload_quat', 'leader_cmd', 'follower_cmd', 'follower_cmd_raw_x', 'follower_guidance_q',
                           'link_tension_leader', 'link_tension_follower', 'distance_eef', 'distance_base')}
    for role in ROLES:
        for k in ('base_pos', 'base_ori', 'base_lin_vel', 'base_ang_vel', 'arm_q', 'arm_qd', 'eef', 'wrench_est', 'mpc_status'):
            log[f'{role}_{k}'] = []

    unstable_at = None
    last_render = 0.0
    while d.time < duration:
        t = d.time
        qpos, qvel = d.qpos[:nq_robots], d.qvel
        feet_pos, hip_pos = env.feet_pos(frame='world'), env.hip_positions(frame='world')
        feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
        jac_feet_dot = (feet_jac - jac_feet_prev) / dt
        jac_feet_prev = feet_jac
        feet_vel = LegsAttrTwoRbots(**{leg: feet_jac[leg] @ d.qvel for leg in env.legs_order})
        state = {
            'leader': dict(pos=env.base_pos.copy(), lin_vel=env.base_lin_vel(frame='world').copy(),
                           ang_vel=env.base_ang_vel(frame='world').copy(), ori=env.base_ori_euler_xyz.copy(),
                           heading=env.heading_orientation_SO3),
            'follower': dict(pos=env.base_pos_follower.copy(), lin_vel=env.base_lin_vel_follower(frame='world').copy(),
                             ang_vel=env.base_ang_vel_follower(frame='world').copy(), ori=env.base_ori_euler_xyz_follower.copy(),
                             heading=env.heading_orientation_SO3_follower),
        }
        # leader: external command; follower: arm-deflection guidance once calibrated
        lin_l, ang_l = leader_command(sc['leader'], t)
        state['leader']['ref'] = (state['leader']['heading'] @ lin_l, ang_l)
        q_f = guidance_input(arm_state(m, d, 'follower')[0])
        raw_x = 0.0
        if t >= GUIDANCE_START:
            if guidance is None:
                p = params['follower']
                guidance = PassiveArmInterface(spring_gains=np.array(p['stiffness'], float), robot='aliengo_follower',
                                               joint_pos0=q_f.copy(), mass_matrix=np.array([0.554, 0.554, p['link3_mass']]),
                                               damping_gains=np.array(p['damping'], float), com_offset=np.zeros(3))
                guidance.guidance.update(guidance_params)
            lin_f, ang_f = guidance.compute_reference_velocity(q_f, ref_prev[0], ref_prev[1])
            raw_x = lin_f[0]
            ref_prev = np.array([lin_f[0], ang_f[2]])
        else:
            lin_f, ang_f = np.zeros(3), np.zeros(3)
        state['follower']['ref'] = (state['follower']['heading'] @ lin_f, ang_f)

        action = np.zeros(m.nu)
        for role in ROLES:
            s = state[role]
            q_arm, qd_arm = arm_state(m, d, role)
            # a copy: the estimator mirrors the follower's angles in place
            wrench, jeef[role], eef_est = estimator[role].calculate_force_estimates_damping(jeef[role], q_arm.copy(), qd_arm, s['ori'])
            com = d.subtree_com[trunk_id[role]].copy()
            com[2] = s['pos'][2]  # MPC position: current CoM in x, y and base height (as in zimpedance_sim)
            tau_role, _ = wrapper[role].compute_actions(
                com, s['pos'], s['lin_vel'], s['ori'], s['ang_vel'], split(feet_pos, role), split(hip_pos, role),
                LegsAttr(**{leg: qpos[env.legs_qpos_idx[leg + SUFFIX[role]]] for leg in LEGS}), None, list(LEGS), dt,
                s['ref'][0], s['ref'][1], env.step_num, qpos, qvel, split(feet_jac, role), split(jac_feet_dot, role),
                split(feet_vel, role), split(env.legs_qfrc_passive, role), split(env.legs_qfrc_bias, role),
                split(env.legs_mass_matrix, role), split(env.legs_qpos_idx, role), split(env.legs_qvel_idx, role),
                split(tau, role), cfg.inertia.flatten(), wrench, jeef[role], eef_est)
            for leg in LEGS:
                action[env.legs_tau_idx[leg + SUFFIX[role]]] = tau_role[leg]
            solver = getattr(wrapper[role].srbd_controller_interface.controller, 'acados_ocp_solver', None)
            log[f'{role}_base_pos'].append(s['pos'])
            log[f'{role}_base_ori'].append(s['ori'])
            log[f'{role}_base_lin_vel'].append(s['lin_vel'])
            log[f'{role}_base_ang_vel'].append(s['ang_vel'])
            log[f'{role}_arm_q'].append(q_arm)
            log[f'{role}_arm_qd'].append(qd_arm)
            log[f'{role}_eef'].append(d.site(f'eef_{role}').xpos.copy())
            log[f'{role}_wrench_est'].append(np.asarray(wrench, float).ravel()[:6])
            log[f'{role}_mpc_status'].append(solver.status if solver is not None else np.nan)
        env.step(action=action)
        if d.time < t:  # MuJoCo reset the state after a numerical blow-up (warning 'Nan, Inf or huge value')
            unstable_at = t
            break

        log['time'].append(d.time)
        log['payload_pos'].append(d.xpos[payload_id].copy() if has_payload else np.full(3, np.nan))
        log['payload_quat'].append(d.xquat[payload_id].copy() if has_payload else np.full(4, np.nan))
        log['leader_cmd'].append(np.r_[lin_l[0], ang_l[2]])
        log['follower_cmd'].append(np.r_[lin_f[0], ang_f[2]])
        log['follower_cmd_raw_x'].append(raw_x)
        log['follower_guidance_q'].append(q_f)
        for role, tension in link_tensions(m, d).items():  # force carried by each robot's connection
            log[f'link_tension_{role}'].append(tension)
        log['distance_eef'].append(np.linalg.norm(d.site('eef_leader').xpos - d.site('eef_follower').xpos))
        log['distance_base'].append(np.linalg.norm(d.qpos[0:2] - d.qpos[22:24]))
        if render and time.time() - last_render >= 1.0 / RENDER_FREQ:  # wall-clock rate, as the other sims
            env.render()
            last_render = time.time()
    env.close()
    if unstable_at is not None:  # drop the robots' last entries of the step that blew up
        n = len(log['time'])
        log = {k: v[:n] for k, v in log.items()}
    meta = dict(scenario=scenario, sim_dt=dt, payload_preset=payload_preset, unstable_at=unstable_at,
                payload_preset_status=PAYLOAD_PRESETS[payload_preset]['status'] if payload_preset else None,
                estimator_rest_position={r: estimator[r].joint_pos0.tolist() for r in ROLES}, leader_command=sc['leader'], duration=duration, guidance_start=GUIDANCE_START,
                guidance_neutral_pose=None if guidance is None else guidance.joint_pos0.tolist(),
                guidance_params=None if guidance is None else guidance.guidance,
                guidance_convention='follower_guidance_q = (q1, |q2|, -q3) of follower_arm_q; neutral pose in the same convention', controller='collaborative',
                **{k: v for k, v in info.items() if k not in ('leg_joints', 'feet_geoms')},
                units='SI; base_ori roll-pitch-yaw; *_cmd = (forward velocity heading frame, yaw rate); '
                      'link_tension_* is the force carried by each connection (connect constraint, or tendon spring + limit)')
    return {k: np.asarray(v) for k, v in log.items()}, meta


def _vec(text):
    return None if text is None else [float(x) for x in text.split(',')]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--scenario', choices=sorted(SCENARIOS), default='straight')
    ap.add_argument('--terrain', default='flat', help='aliengo_leader scene: flat, rough, ramp, ...')
    ap.add_argument('--payload-preset', choices=sorted(PAYLOAD_PRESETS), default='light_2kg',
                    help='coupling, payload mass and arm preset (light_2kg validated; medium_5kg, heavy_7kg experimental)')
    ap.add_argument('--payload', choices=('elastic_payload', 'rigid_bar', 'elastic'),
                    help='elastic_payload: payload hung by two 15 N/m tendons (<= 0.15 m); rigid_bar: connect constraints; '
                         'elastic: one 10 N/m tendon from EEF to EEF (<= 0.4 m), no payload')
    ap.add_argument('--payload-mass', type=float, help='kg (2 kg: the ZiMPedance payload; PACC Sec. IV-A used 7 kg)')
    ap.add_argument('--arm-preset', choices=sorted(ARM_PRESETS),
                    help="model: the robot MJCFs' arm impedance; original_elastic / original_dual: the arms of the "
                         "original elastic (5 kg) and last two-robot scenes")
    ap.add_argument('--arm-params', metavar='JSON', help='{role: {stiffness, damping, springref, q0, link3_mass, eef_mass}} overrides')
    ap.add_argument('--duration', type=float)
    ap.add_argument('--leader-vx', type=float, help="override the scenario's leader forward speed [m/s]")
    ap.add_argument('--leader-wz', type=float, help="override the scenario's leader yaw rate [rad/s]")
    ap.add_argument('--guidance', action='append', default=[], metavar='KEY=VALUE',
                    help='follower guidance parameter (theta1, theta2, vx1, vx2_pos, vx2_neg, psi1, psi2, wz1, wz2)')
    ap.add_argument('--sim-dt', type=float, help=f"simulation step [s] (config: {cfg.simulation_params['dt']})")
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--export', metavar='NPZ')
    for role in ROLES:
        for quantity in ('stiffness', 'damping', 'springref', 'q0'):
            ap.add_argument(f'--{quantity}-{role}', help='three comma-separated values, one per arm joint')
    args = ap.parse_args()
    payload, payload_mass, arm_preset = payload_setup(args.payload_preset, args.payload, args.payload_mass, args.arm_preset)
    arm_params = load_arm_params(args.arm_params) if args.arm_params else {}
    for role in ROLES:  # command-line values override the JSON file
        arm_params.setdefault(role, {}).update({q: _vec(getattr(args, f'{q}_{role}')) for q in ('stiffness', 'damping', 'springref', 'q0')
                                                if getattr(args, f'{q}_{role}')})
    log, meta = run(args.scenario, args.terrain, payload, payload_mass, arm_params, args.duration,
                    render=not args.headless, leader_vx=args.leader_vx, leader_wz=args.leader_wz,
                    arm_preset=arm_preset, guidance={k: float(v) for k, v in (a.split('=') for a in args.guidance)},
                    payload_preset=args.payload_preset, sim_dt=args.sim_dt)
    print(f"simulated {log['time'][-1]:.2f} s; leader x {log['leader_base_pos'][-1][0]:.2f} m, follower x "
          f"{log['follower_base_pos'][-1][0]:.2f} m, EEF distance {log['distance_eef'][-1]:.3f} m")
    if meta['unstable_at'] is not None:
        print(f"stopped: MuJoCo became unstable at {meta['unstable_at']:.3f} s")
    if args.export:
        np.savez_compressed(args.export, metadata=json.dumps(meta, default=str), **log)
        print("exported to", args.export)


if __name__ == '__main__':
    main()
