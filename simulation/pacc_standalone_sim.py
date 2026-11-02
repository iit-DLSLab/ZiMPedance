"""Standalone PACC simulation: passive arms on fixed blocks carrying a shared payload, no
quadruped locomotion controller.

    python simulation/pacc_standalone_sim.py [--headless] [--duration 15] [--arms both|follower|leader]
        [--stiffness-follower 10,25,0.001] [--damping-follower 4.2,10.02,4] [--springref-follower 0,0,0]
        [--q0-follower 0,0,0] (same for --*-leader) [--payload-mass 10]
        [--arm-preset model|original_elastic|original_dual] [--payload-preset light_2kg|medium_5kg|heavy_7kg]
        [--arm-params params.json] [--impulse 200,0,0] [--impulse-start 5] [--impulse-duration 0.5] [--export run.npz]

Model: gym_quadruped/robot_model/aliengo_two_robots/scene_pacc_arms_on_blocks.xml (two_arm_only.xml:
front-mounted follower arm, two_arm_only_2.xml: rear-mounted leader arm, each on a 500 kg block;
weight.xml: a 10 kg bar hanging from both end effectors by 0.1 m rope tendons). Stiffness,
damping, spring reference and payload mass are set on the MjSpec before compiling, so they are
the values MuJoCo integrates; unspecified values are the MJCF's. Arm parameters use the format and
presets of the two-robot sim (pacc_models.ARM_PRESETS, PAYLOAD_PRESETS, a JSON file of
{role: {stiffness, damping, springref, q0, link3_mass, eef_mass}}); a payload preset sets the arms
and the bar mass, while the coupling stays this scene's ropes. This replaces two_arm_tes_sim.py,
which pointed at another machine's path and a scene that no longer loaded.
"""
import argparse
import json
import os
import time

import mujoco
import numpy as np

try:
    from simulation.pacc_models import (ROBOT_MODELS, ARM_PRESETS, PAYLOAD_PRESETS, resolve_arm_params, load_arm_params,
                                        set_arm_params, set_body_mass, tendon, arm_state, set_arm_qpos)
except ImportError:  # repo root not on sys.path
    from pacc_models import (ROBOT_MODELS, ARM_PRESETS, PAYLOAD_PRESETS, resolve_arm_params, load_arm_params,
                             set_arm_params, set_body_mass, tendon, arm_state, set_arm_qpos)

SCENE = os.path.join(ROBOT_MODELS, 'aliengo_two_robots', 'scene_pacc_arms_on_blocks.xml')
ROLES = ('follower', 'leader')
TENDONS = {'follower': 'link_arms_follower', 'leader': 'link_arms1'}  # rope from each EEF to the bar
PAYLOAD = 'mountpoint'  # the bar body of weight.xml
MASS_BODIES = {'link3_mass': '{role}/pa_link_2_ext', 'eef_mass': 'eef_{role}'}


def build_model(arms=ROLES, arm_params=None, payload_mass=None):
    """Compile the arms-on-blocks scene. arm_params: {role: {stiffness, damping, springref, link3_mass,
    eef_mass}} (pacc_models format; q0 is the initial state, see simulate); None keeps the MJCF value."""
    spec = mujoco.MjSpec.from_file(SCENE)
    for key in list(spec.keys):  # the file's keyframe is stale (blocks swapped); qpos0 is the intended pose
        spec.delete(key)
    for role in ROLES:
        if role not in arms:
            spec.delete(tendon(spec, TENDONS[role]))
            spec.delete(spec.body(f'{role}/trunk'))
    for role, params in (arm_params or {}).items():
        if role in arms:
            set_arm_params(spec, role, *(params.get(k) for k in ('stiffness', 'damping', 'springref')))
            for key, body in MASS_BODIES.items():
                if params.get(key) is not None:
                    set_body_mass(spec, body.format(role=role), params[key])
    if payload_mass is not None:
        set_body_mass(spec, PAYLOAD, payload_mass)
    return spec.compile()


def tendon_force(model, data, name):
    """Rope tension [N] from the tendon-limit constraint (0 when slack)."""
    tid = model.tendon(name).id
    rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON) & (data.efc_id == tid)
    return float(np.abs(data.efc_force[rows]).sum())


def simulate(model, arms=ROLES, q0=None, duration=15.0, impulse=(200.0, 0.0, 0.0), impulse_start=5.0,
             impulse_duration=0.5, data=None, viewer=None):
    """Run from qpos0 (arm joints at q0), apply the impulse to the payload, return the log."""
    data = mujoco.MjData(model) if data is None else data
    for role, q in (q0 or {}).items():
        if role in arms:
            set_arm_qpos(model, data, role, q)
    mujoco.mj_forward(model, data)
    payload = model.body(PAYLOAD).id
    log = {k: [] for k in ('time', 'payload_pos', 'payload_quat', 'applied_force')}
    for role in arms:
        for k in ('q', 'qd', 'eef', 'rope_length', 'rope_force'):
            log[f'{role}_{k}'] = []
    while data.time < duration:
        on = impulse_start <= data.time < impulse_start + impulse_duration
        data.xfrc_applied[payload, 0:3] = impulse if on else 0.0
        step_start = time.time()
        mujoco.mj_step(model, data)
        log['time'].append(data.time)
        log['payload_pos'].append(data.xpos[payload].copy())
        log['payload_quat'].append(data.xquat[payload].copy())
        log['applied_force'].append(data.xfrc_applied[payload, 0:3].copy())
        for role in arms:
            q, qd = arm_state(model, data, role)
            log[f'{role}_q'].append(q)
            log[f'{role}_qd'].append(qd)
            log[f'{role}_eef'].append(data.site(f'eef_{role}').xpos.copy())
            log[f'{role}_rope_length'].append(data.ten_length[model.tendon(TENDONS[role]).id])
            log[f'{role}_rope_force'].append(tendon_force(model, data, TENDONS[role]))
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - step_start)))
    return {k: np.asarray(v) for k, v in log.items()}


def compiled_arm_params(model, role):
    """The arm parameters MuJoCo integrates, in the pacc_models format (without q0)."""
    ids = [model.joint(f'{role}/{j}').id for j in ('arm_link_1_joint', 'arm_link_2_joint', 'arm_link_3_joint')]
    return dict(stiffness=model.jnt_stiffness[ids].tolist(), damping=model.dof_damping[model.jnt_dofadr[ids]].tolist(),
                springref=model.qpos_spring[model.jnt_qposadr[ids]].tolist(),
                **{key: float(model.body(body.format(role=role)).mass[0]) for key, body in MASS_BODIES.items()})


def _vec(text):
    return None if text is None else [float(x) for x in text.split(',')]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--duration', type=float, default=15.0)
    ap.add_argument('--arms', choices=('both', 'follower', 'leader'), default='both')
    for role in ROLES:
        for quantity in ('stiffness', 'damping', 'springref', 'q0'):
            ap.add_argument(f'--{quantity}-{role}', help='three comma-separated values, one per arm joint')
    ap.add_argument('--payload-mass', type=float, help='bar mass [kg] (MJCF: 10; default: the payload preset\'s)')
    ap.add_argument('--arm-preset', choices=sorted(ARM_PRESETS), help='arm parameters of both arms (default: the MJCF\'s)')
    ap.add_argument('--payload-preset', choices=sorted(PAYLOAD_PRESETS),
                    help='payload mass and arm preset of a two-robot payload preset (coupling: this scene\'s ropes)')
    ap.add_argument('--arm-params', metavar='JSON', help='{role: {stiffness, damping, springref, q0, ...}} overrides')
    ap.add_argument('--impulse', default='200,0,0', help='force on the payload [N], world frame')
    ap.add_argument('--impulse-start', type=float, default=5.0)
    ap.add_argument('--impulse-duration', type=float, default=0.5)
    ap.add_argument('--export', metavar='NPZ')
    args = ap.parse_args()

    arms = ROLES if args.arms == 'both' else (args.arms,)
    preset = PAYLOAD_PRESETS.get(args.payload_preset, {})
    arm_preset = args.arm_preset or preset.get('arm_preset')
    payload_mass = args.payload_mass if args.payload_mass is not None else preset.get('payload_mass')
    overrides = load_arm_params(args.arm_params) if args.arm_params else {}
    for role in arms:  # command-line values override the JSON file
        overrides.setdefault(role, {}).update({q: _vec(getattr(args, f'{q}_{role}')) for q in
                                               ('stiffness', 'damping', 'springref', 'q0') if getattr(args, f'{q}_{role}')})
    arm_params = resolve_arm_params(arm_preset, overrides) if arm_preset else overrides
    q0 = {role: arm_params[role]['q0'] for role in arms if (arm_params.get(role) or {}).get('q0') is not None}
    model = build_model(arms, arm_params, payload_mass)
    run = dict(arms=arms, q0=q0, duration=args.duration, impulse=_vec(args.impulse),
               impulse_start=args.impulse_start, impulse_duration=args.impulse_duration)
    if args.headless:
        log = simulate(model, **run)
    else:
        import mujoco.viewer
        data = mujoco.MjData(model)
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
            log = simulate(model, data=data, viewer=v, **run)
    effective = {role: compiled_arm_params(model, role) for role in arms}
    print(f"simulated {log['time'][-1]:.2f} s; arms {arms}; payload {model.body(PAYLOAD).mass[0]:.2f} kg; "
          f"arm parameters {json.dumps(effective)}")
    if args.export:
        meta = dict(scene=SCENE, arms=arms, arm_preset=arm_preset, payload_preset=args.payload_preset,
                    arm_params=effective, q0=q0, payload_mass=float(model.body(PAYLOAD).mass[0]),
                    impulse=run['impulse'], impulse_start=args.impulse_start, impulse_duration=args.impulse_duration,
                    timestep=model.opt.timestep, units='SI; quaternions wxyz; rope_force is the tendon-limit tension')
        np.savez_compressed(args.export, metadata=json.dumps(meta), **log)
        print("exported to", args.export)


if __name__ == '__main__':
    main()
