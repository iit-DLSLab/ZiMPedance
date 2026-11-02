"""Two-robot collaborative-carrying scene, composed with MjSpec from the single-robot models.

    model, info = build_collaborative_model(terrain='flat', payload='elastic_payload', payload_mass=2.0,
                                            arm_params={'leader': {...}, 'follower': {...}})

Composition (no robot definition is duplicated):
- the leader and the terrain: gym_quadruped/robot_model/aliengo_leader/scene_<terrain>.xml (rear-mounted
  arm; legs, motors and IMU through aliengo_leader_actuated.xml);
- the follower: aliengo_follower/aliengo_follower_actuated.xml (front-mounted arm), attached with
  MjSpec through a frame at its spawn pose;
- the payload coupling (PAYLOADS):
  - 'elastic_payload' (default): weight.xml hung between the two end effectors by two elastic
    spatial tendons, stiffness 15 N/m, limited to 0.15 m (as in the earlier
    two-robot scene);
  - 'rigid_bar': rigid_load.xml connected to both end effectors by two 'connect' equality
    constraints;
  - 'elastic': no payload body, one tendon from end effector to end effector, stiffness 10 N/m,
    limited to 0.4 m.
  Tendons act as elastic ropes: spring deadband [0, spawn length] (no force when shorter, k beyond)
  up to the length limit. MuJoCo's default rest length, the length at qpos0, is the bodies' file
  poses in a composed scene (1.1-1.2 m here against 0.12 m at spawn), which would push the payload
  away; the original scenes relied on that default as well.

Joint and actuator names keep their leader/ and follower/ prefixes, which the controllers use.
Names that both robot files define are renamed per role: the EEF sites become eef_leader and
eef_follower, the IMU sensors imu_leader / imu_follower (the names of the original two-robot
scene), and the leader's EEF body 'follower/eef' becomes 'leader/eef'. Model order is leader,
follower, payload, which QuadrupedEnvColl's indexing assumes (follower base at qpos[22:29]).
"""
import mujoco
import numpy as np

try:
    from simulation.pacc_models import (ROBOT_MODELS, ROLES, ARM_PRESETS, PAYLOAD_PRESETS, resolve_arm_params,
                                        set_arm_params, set_body_mass, set_arm_qpos)
except ImportError:
    from pacc_models import (ROBOT_MODELS, ROLES, ARM_PRESETS, PAYLOAD_PRESETS, resolve_arm_params,
                             set_arm_params, set_body_mass, set_arm_qpos)

LEG_JOINTS = {role: {f'{leg}{i}': [f'{role}/{leg}_{j}_joint' for j in ('hip', 'thigh', 'calf')]
                     for leg in ('FL', 'FR', 'RL', 'RR')}
              for i, role in ((1, 'leader'), (2, 'follower'))}
FEET_GEOMS = {'leader': dict(FL1='FL1', FR1='FR1', RL1='RL1', RR1='RR1'),
              'follower': dict(FL2='FL2', FR2='FR2', RL2='RL2', RR2='RR2')}
STAND_LEGS = np.array([0.0, 0.9, -1.8] * 4)
# Spawn height of both bases: feet clear of the ground (follower feet are 4 cm spheres, 0.311 m below
# the trunk when standing). QuadrupedEnvColl.reset lifts only the leader until no foot touches, so a
# touching follower foot would never be cleared.
BASE_HEIGHT = 0.355

# file: payload body (None: no body); tendons: (site1, site2) pairs; eef_gap: initial end-effector
# distance beyond the mount points (each side); drop: payload below the end effectors at spawn
PAYLOADS = {
    'elastic_payload': dict(file='aliengo_two_robots/weight.xml', original_mass=10.0, coupling='tendon',
                            tendons=[('eef_leader', 'mount_point_1'), ('eef_follower', 'mount_point_2')],
                            stiffness=15.0, max_length=0.15, eef_gap=0.06, drop=0.1),
    'rigid_bar': dict(file='aliengo_two_robots/rigid_load.xml', original_mass=0.001, coupling='connect',
                      eef_gap=0.0, drop=0.0),
    'elastic': dict(file=None, coupling='tendon', tendons=[('eef_follower', 'eef_leader')],
                    stiffness=10.0, max_length=0.4, eef_distance=0.3),
}
EEF_BODY = {'leader': 'leader/eef', 'follower': 'follower/eef'}


def _rename(items, old, new):
    for item in items:
        if item.name == old:
            item.name = new
            return
    raise KeyError(old)


def _robot_specs(terrain):
    arena = mujoco.MjSpec.from_file(f'{ROBOT_MODELS}/aliengo_leader/scene_{terrain}.xml')
    follower = mujoco.MjSpec.from_file(f'{ROBOT_MODELS}/aliengo_follower/aliengo_follower_actuated.xml')
    for spec in (arena, follower):
        for key in list(spec.keys):  # replaced by one two-robot keyframe
            spec.delete(key)
    _rename(arena.sites, 'eef', 'eef_leader')
    _rename(arena.bodies, 'follower/eef', 'leader/eef')
    _rename(arena.geoms, 'follower/eef', 'leader/eef')
    _rename(arena.sensors, 'imu', 'imu_leader')
    _rename(arena.sensors, 'imu_gyro', 'imu_gyro_leader')
    _rename(follower.sites, 'eef', 'eef_follower')
    _rename(follower.geoms, 'eef', 'follower/eef')
    _rename(follower.sensors, 'imu', 'imu_follower')
    _rename(follower.sensors, 'follower/imu_gyro', 'imu_gyro_follower')
    return arena, follower


def _box_inertia(mass, half_size):
    a, b, c = 2 * np.asarray(half_size)
    return mass / 12.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])


def build_collaborative_model(terrain='flat', payload='elastic_payload', payload_mass=2.0, arm_params=None,
                              arm_preset='model', follower_offset_y=0.0, follower_yaw=0.0):
    """Compile the two-robot scene. arm_params: {role: {stiffness, damping, springref, q0, link3_mass,
    eef_mass}}, overriding pacc_models.ARM_PRESETS[arm_preset] key by key (the robot MJCFs' leg and arm
    geometry is kept; the arm impedance and masses are always set from the parameter set);
    payload_mass None keeps the payload file's. Returns (model, info); info['arm_params'] holds the
    values compiled into the model."""
    params = resolve_arm_params(arm_preset, arm_params)

    def compose(follower_x, payload_pos, payload_quat):
        arena, follower = _robot_specs(terrain)
        quat = np.array([np.cos(follower_yaw / 2), 0, 0, np.sin(follower_yaw / 2)])
        frame = arena.worldbody.add_frame(pos=[follower_x, follower_offset_y, 0.0], quat=quat)
        frame.attach_body(follower.body('follower/trunk'), '', '')
        spec_p = PAYLOADS[payload]
        if spec_p['file']:
            load = mujoco.MjSpec.from_file(f"{ROBOT_MODELS}/{spec_p['file']}")
            for light in list(load.lights):
                load.delete(light)
            load_frame = arena.worldbody.add_frame(pos=list(payload_pos), quat=list(payload_quat))
            load_frame.attach_body(load.body('mountpoint'), '', '')
            for role in ROLES:
                # the payload hangs at the arm tip: contacts with the last link and EEF body there are
                # geometric overlap, not collisions (they fought the connection in the original scene)
                for body in (f'{role}/pa_link_2_ext', EEF_BODY[role]):
                    arena.add_exclude(bodyname1='mountpoint', bodyname2=body)
        if spec_p['coupling'] == 'connect':
            for mount, eef in (('mount_point_1', 'eef_leader'), ('mount_point_2', 'eef_follower')):
                eq = arena.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_SITE)
                eq.name1, eq.name2, eq.name = mount, eef, f'connect_{eef}'
        else:
            for site1, site2 in spec_p['tendons']:
                link = arena.add_tendon(name=f'link_{site1}', limited=True, range=[0, spec_p['max_length']],
                                        width=0.01, stiffness=spec_p['stiffness'])
                link.wrap_site(site1)
                link.wrap_site(site2)
        for role in ROLES:
            p = params[role]
            set_arm_params(arena, role, p['stiffness'], p['damping'], p['springref'])
            set_body_mass(arena, f'{role}/pa_link_2_ext', p['link3_mass'])
            set_body_mass(arena, EEF_BODY[role], p['eef_mass'])
        if not spec_p['file']:
            return arena, 0.0
        bar = arena.body('mountpoint')
        mass = spec_p['original_mass'] if payload_mass is None else payload_mass
        if payload_mass is not None:
            # the original inertial block is a placeholder copied from a hip link; use the bar's box
            bar.mass, bar.ipos, bar.iquat = float(mass), [0, 0, 0], [1, 0, 0, 0]
            bar.inertia = _box_inertia(mass, next(g for g in arena.geoms if g.parent.name == 'mountpoint').size)
        return arena, mass

    # Place the follower so that, at the initial arm angles, the end effectors are as far apart as the
    # payload's mount points (connect: no initial violation) plus the tendon spans, with the payload
    # between and below them; without a payload body, at the coupling's eef_distance.
    spec_p = PAYLOADS[payload]
    arena, _ = compose(-1.3, [0, 0, 1.0], [1, 0, 0, 0])
    model = arena.compile()
    data = mujoco.MjData(model)
    qpos = _stand_qpos(model, -1.3, follower_offset_y, follower_yaw, params, [0, 0, 1.0], [1, 0, 0, 0])
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    eef_l, eef_f = data.site('eef_leader').xpos.copy(), data.site('eef_follower').xpos.copy()
    if spec_p['file']:
        eef_dist = (np.linalg.norm(data.site('mount_point_1').xpos - data.site('mount_point_2').xpos)
                    + 2 * spec_p['eef_gap'])
    else:
        eef_dist = spec_p['eef_distance']
    dx = np.sqrt(max(eef_dist ** 2 - (eef_l[1] - eef_f[1]) ** 2 - (eef_l[2] - eef_f[2]) ** 2, 0.0))
    follower_x = -1.3 + (eef_l[0] - eef_f[0]) - dx
    eef_f = eef_f + np.array([follower_x + 1.3, 0, 0])
    mid = 0.5 * (eef_l + eef_f) - np.array([0, 0, spec_p.get('drop', 0.0)])
    yaw_bar = np.arctan2(eef_l[1] - eef_f[1], eef_l[0] - eef_f[0])
    pitch_bar = -np.arcsin((eef_l[2] - eef_f[2]) / np.linalg.norm(eef_l - eef_f))
    bar_quat = np.zeros(4)
    mujoco.mju_euler2Quat(bar_quat, [0, pitch_bar, yaw_bar], 'xyz')
    arena, mass = compose(follower_x, mid, bar_quat)
    model = arena.compile()
    key_qpos = _stand_qpos(model, follower_x, follower_offset_y, follower_yaw, params, mid, bar_quat)
    arena.add_key(name='home', qpos=key_qpos, ctrl=np.zeros(model.nu))
    model = arena.compile()
    if model.ntendon:
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_tendon(model, data)
        for tendon in arena.tendons:
            tendon.springlength = [0.0, float(data.ten_length[model.tendon(tendon.name).id])]
        model = arena.compile()
    for role in ROLES:  # report what MuJoCo integrates
        ids = [model.joint(f'{role}/{j}').id for j in ('arm_link_1_joint', 'arm_link_2_joint', 'arm_link_3_joint')]
        params[role].update(stiffness=model.jnt_stiffness[ids].tolist(), damping=model.dof_damping[model.jnt_dofadr[ids]].tolist(),
                            springref=model.qpos_spring[model.jnt_qposadr[ids]].tolist(),
                            link3_mass=float(model.body(f'{role}/pa_link_2_ext').mass[0]),
                            eef_mass=float(model.body(EEF_BODY[role]).mass[0]))
    info = dict(terrain=terrain, arm_preset=arm_preset, payload=payload, payload_mass=mass, follower_x=follower_x,
                follower_offset_y=follower_offset_y, follower_yaw=follower_yaw, arm_params=params,
                leg_joints=LEG_JOINTS, feet_geoms=FEET_GEOMS)
    return model, info


def _stand_qpos(model, follower_x, follower_y, follower_yaw, params, payload_pos, payload_quat):
    data = mujoco.MjData(model)
    q = data.qpos
    q[0:3], q[3:7] = [0, 0, BASE_HEIGHT], [1, 0, 0, 0]
    q[22:25], q[25:29] = [follower_x, follower_y, BASE_HEIGHT], [np.cos(follower_yaw / 2), 0, 0, np.sin(follower_yaw / 2)]
    for role, base in (('leader', 7), ('follower', 29)):
        q[base:base + 12] = STAND_LEGS
        set_arm_qpos(model, data, role, params[role]['q0'])
    if model.njnt and 'weight_joint' in [model.joint(j).name for j in range(model.njnt)]:
        load = model.joint('weight_joint').qposadr[0]
        q[load:load + 3], q[load + 3:load + 7] = payload_pos, payload_quat
    return q.copy()


def link_tensions(model, data):
    """Force carried by each robot's connection to the payload [N]: the connect-constraint force, or the
    tendon's elastic force k (L - L0) plus its length-limit force (one shared tendon for 'elastic')."""
    out = {}
    for role in ROLES:
        site = f'eef_{role}'
        if model.neq:
            eid = model.equality(f'connect_{site}').id
            rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == eid)
            out[role] = float(np.linalg.norm(data.efc_force[rows][:3]))
            continue
        names = [model.tendon(i).name for i in range(model.ntendon)]
        name = f'link_{site}' if f'link_{site}' in names else names[0]
        tid = model.tendon(name).id
        spring = model.tendon_stiffness[tid] * max(data.ten_length[tid] - model.tendon_lengthspring[tid][1], 0.0)
        rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON) & (data.efc_id == tid)
        out[role] = float(spring + np.abs(data.efc_force[rows]).sum())
    return out
