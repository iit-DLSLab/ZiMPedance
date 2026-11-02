"""MjSpec helpers shared by the PACC (passive arm) simulations.

The robot and arm geometry comes from the MJCF files in gym_quadruped/robot_model; these helpers only
change physical parameters on the spec before it is compiled, so the values reach MuJoCo's
physics directly (joint stiffness, damping and spring reference; body masses).
"""
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_MODELS = os.path.join(REPO, 'gym_quadruped', 'robot_model')
ARM_JOINTS = ('arm_link_1_joint', 'arm_link_2_joint', 'arm_link_3_joint')
ROLES = ('leader', 'follower')

# Passive-arm parameter set of one robot, the format both PACC simulations take (also as JSON, see
# load_arm_params): stiffness [N m/rad], damping [N m s/rad], springref [rad] (spring rest angle) and
# q0 [rad] (initial angle), one value per arm joint; link3_mass and eef_mass [kg]. Arm link and EEF
# masses are those of the original two-robot scene (aliengo_two_robots/new_urdf_export_*.xml).
ARM_PARAM_KEYS = ('stiffness', 'damping', 'springref', 'q0', 'link3_mass', 'eef_mass')
_MASSES = {'leader': dict(link3_mass=0.25, eef_mass=0.07), 'follower': dict(link3_mass=0.25, eef_mass=0.05)}


def _arms(stiffness, damping, springref=(0, 0, 0), q0=(0, 0, 0)):
    return {role: dict(stiffness=list(stiffness), damping=list(damping), springref=list(springref), q0=list(q0),
                       **_MASSES[role]) for role in ROLES}


ARM_PRESETS = {
    # the robot MJCFs' own impedance (aliengo_leader/ and aliengo_follower/new_urdf_export.xml), as in the
    # ZiMPedance paper's model; ZiMPedance hardware-identified k [5, 8, 5]
    'model': _arms([5, 8, 5], [0.5, 1.5, 0.5]),
    # the arms of the original elastic two-robot scene, which carried a 5 kg payload on the same two
    # 15 N/m tendons (earlier two-robot scene, aliengo_two_robots/
    # aliengo_arm_{leader,follower}_named.xml, Jan 2025)
    'original_elastic': _arms([10, 10, 10], [1, 1, 1]),
    # the last two-robot scene of the PACC sim (aliengo_two_robots/
    # new_urdf_export_{leader,follower}.xml, Sept 2025); tuned in simulation, not measured
    'original_dual': _arms([25, 25, 5], [3.7, 8.7, 3.7]),
}

# Payload presets of the two-robot carrying sim: payload coupling and mass, and the arm preset to use
# with it. 'scenarios' lists the closed-loop scenarios of collaborative_carrying_sim.py that pass for
# the preset (the regression runs them); 'experimental' presets fail others.
# None of these impedances was identified for its payload on hardware.
PAYLOAD_PRESETS = {
    # ZiMPedance payload and PACC paper Sec. IV-B (2 kg on ropes)
    'light_2kg': dict(payload='elastic_payload', payload_mass=2.0, arm_preset='model', status='validated',
                      scenarios=('stand', 'straight', 'stop', 'turn', 'turn_right', 'turn_stop')),
    # the original elastic scene's 5 kg on its two tendons; its own arms ('original_elastic') fall in
    # straight carrying with this controller, the stiff 'original_dual' arms carry it
    'medium_5kg': dict(payload='elastic_payload', payload_mass=5.0, arm_preset='original_dual', status='experimental',
                       scenarios=('stand', 'straight')),
    # PACC paper Sec. IV-A (7 kg on a rigid bar) with the original stiff two-robot arms
    'heavy_7kg': dict(payload='rigid_bar', payload_mass=7.0, arm_preset='original_dual', status='experimental',
                      scenarios=('stand', 'straight')),
}


def resolve_arm_params(arm_preset='model', overrides=None):
    """{role: parameter set}: ARM_PRESETS[arm_preset] with `overrides` ({role: {key: value}}) applied
    key by key; None values are ignored."""
    params = {}
    for role in ROLES:
        given = {k: v for k, v in ((overrides or {}).get(role) or {}).items() if v is not None}
        unknown = set(given) - set(ARM_PARAM_KEYS)
        if unknown:
            raise KeyError(f"unknown arm parameter(s) {sorted(unknown)} for {role}; expected {ARM_PARAM_KEYS}")
        params[role] = {**ARM_PRESETS[arm_preset][role], **given}
    return params


def load_arm_params(path):
    """Read {role: {key: value}} arm parameters from a JSON file (any subset of roles and keys)."""
    with open(path) as f:
        return json.load(f)


def estimator_rest_position(role, springref):
    """joint_pos0 for PassiveArmInterface so that its spring torque matches MuJoCo's -k (q - springref).
    The estimator works on (q1, |q2|, -q3) for the follower and on q for the leader, writes the joint-3
    spring as -k (q3 + joint_pos0[2]) and ignores joint_pos0[0] (joint-1 rest fixed at 0)."""
    s = np.asarray(springref, float)
    if abs(s[0]) > 1e-12:
        raise ValueError("PassiveArmInterface assumes a joint-1 spring rest of 0")
    return np.array([0.0, s[1], s[2] if role == 'follower' else -s[2]])


def arm_joint_names(role):
    """Passive-arm joint names of a robot role ('leader' or 'follower')."""
    return [f'{role}/{name}' for name in ARM_JOINTS]


def set_arm_params(spec, role, stiffness=None, damping=None, springref=None):
    """Set the passive-arm joint stiffness [N m/rad], damping [N m s/rad] and spring reference
    [rad] of `role` on an MjSpec. None leaves a quantity as the MJCF defines it."""
    for i, name in enumerate(arm_joint_names(role)):
        joint = spec.joint(name)
        if joint is None:
            raise KeyError(f"joint {name} not in the spec")
        if stiffness is not None:
            joint.stiffness = float(stiffness[i])
        if damping is not None:
            joint.damping = float(damping[i])
        if springref is not None:
            joint.springref = float(springref[i])


def set_body_mass(spec, body_name, mass):
    """Set a body's mass, scaling its explicit inertia by the same factor (shape kept)."""
    body = spec.body(body_name)
    if body is None:
        raise KeyError(f"body {body_name} not in the spec")
    if body.mass > 0:
        body.inertia = np.asarray(body.inertia) * (mass / body.mass)
    body.mass = float(mass)


def tendon(spec, name):
    return next(t for t in spec.tendons if t.name == name)


def arm_state(model, data, role):
    """Passive-arm joint positions and velocities of `role`."""
    ids = [model.joint(n).id for n in arm_joint_names(role)]
    return (np.array([data.qpos[model.jnt_qposadr[j]] for j in ids]),
            np.array([data.qvel[model.jnt_dofadr[j]] for j in ids]))


def set_arm_qpos(model, data, role, q):
    for name, value in zip(arm_joint_names(role), q):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value


def payload_setup(payload_preset=None, payload=None, payload_mass=None, arm_preset=None):
    """(payload, payload_mass, arm_preset) of a payload preset (default light_2kg), each replaced by the
    explicit value when one is given."""
    preset = PAYLOAD_PRESETS[payload_preset or 'light_2kg']
    return (payload or preset['payload'], preset['payload_mass'] if payload_mass is None else payload_mass,
            arm_preset or preset['arm_preset'])
