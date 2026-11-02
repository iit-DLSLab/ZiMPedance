"""Compare the Pinocchio model built from an MJCF against MuJoCo on the same file.

    python simulation/validation/check_pinocchio_vs_mujoco.py <xml> [--n 200] [--meshdir DIR] [--out res.json]

The Pinocchio model is built the way the 'zimpedance' controller builds it. Pinocchio >= 4 uses
buildModelFromMJCF(xml, FreeFlyer, "root_joint"). Pinocchio 3.x does not have that call, so
buildModelFromMJCF(xml) is used (root from the XML's <freejoint/>), and the nq of the 2-argument
FreeFlyer form is recorded.

For random configurations (base pose; leg and arm joints within limits; the first sample is the
zero configuration) it reports the maximum error of:
  eef            eef site, world frame
  arm_bodies     arm body origins, world frame
  all_bodies     every body origin, world frame
  eef_in_trunk   eef site, trunk frame (independent of how the root is placed)
  bodies_in_trunk  every body origin, trunk frame
  M_act          actuated-joint block of the mass matrix (includes armature)
  g_act          actuated-joint gravity torques
The actuated blocks do not depend on the free-joint velocity convention, which differs between
MuJoCo (world linear velocity) and Pinocchio (local).

--meshdir overrides the compiler meshdir, for XMLs whose meshes are not next to them
(MuJoCo needs the meshes to load the file; Pinocchio does not).
"""
import argparse
import json
import os
import re
import tempfile

import mujoco
import numpy as np
import pinocchio as pin

ARM_JOINTS = ["leader/arm_link_1_joint", "leader/arm_link_2_joint", "leader/arm_link_3_joint"]
ARM_BODIES = ["leader/pa_link_0_ext", "leader/pa_link_1_ext", "leader/pa_link_2_ext"]
TRUNK = "leader/trunk"


def load_mujoco(xml, meshdir):
    if meshdir is None:
        return mujoco.MjModel.from_xml_path(xml)
    text = re.sub(r'meshdir="[^"]*"', f'meshdir="{os.path.abspath(meshdir)}/"', open(xml).read())
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=os.path.dirname(os.path.abspath(xml))) as f:
        f.write(text)
        f.flush()
        return mujoco.MjModel.from_xml_path(f.name)


def build_pinocchio(xml, res):
    if int(pin.__version__.split(".")[0]) >= 4:
        r = pin.buildModelFromMJCF(xml, pin.JointModelFreeFlyer(), "root_joint")
        res["build"] = "buildModelFromMJCF(xml, FreeFlyer, 'root_joint')"
    else:
        try:
            r2 = pin.buildModelFromMJCF(xml, pin.JointModelFreeFlyer())
            res["two_arg_freeflyer_nq"] = int((r2[0] if isinstance(r2, tuple) else r2).nq)
        except Exception as e:
            res["two_arg_freeflyer_nq"] = f"error: {e}"
        r = pin.buildModelFromMJCF(xml)
        res["build"] = "buildModelFromMJCF(xml)  (root from <freejoint/>)"
    return r[0] if isinstance(r, tuple) else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--meshdir")
    ap.add_argument("--out")
    ap.add_argument("--role", default="leader", help="name prefix of the robot: leader or follower")
    args = ap.parse_args()
    global ARM_JOINTS, ARM_BODIES, TRUNK
    ARM_JOINTS = [n.replace("leader/", f"{args.role}/") for n in ARM_JOINTS]
    ARM_BODIES = [n.replace("leader/", f"{args.role}/") for n in ARM_BODIES]
    TRUNK = TRUNK.replace("leader/", f"{args.role}/")

    res = {"pinocchio": pin.__version__, "mujoco": mujoco.__version__, "xml": os.path.abspath(args.xml)}
    m = load_mujoco(args.xml, args.meshdir)
    d = mujoco.MjData(m)
    model = build_pinocchio(args.xml, res)
    data = model.createData()

    res["nq_nv"] = [int(model.nq), int(model.nv)]
    res["mujoco_nq_nv"] = [int(m.nq), int(m.nv)]
    res["joints"] = [model.names[i] for i in range(1, model.njoints)]
    res["total_mass_pin"] = float(sum(model.inertias[i].mass for i in range(model.njoints)))
    res["total_mass_mj"] = float(m.body_mass.sum())
    fid = model.getFrameId("eef")
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "eef")
    res["eef_parent_joint_pin"] = model.names[model.frames[fid].parentJoint]
    res["eef_parent_body_mj"] = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[sid])

    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge]
    res["comparable"] = all(model.existJointName(n) for n in names) and model.nq == m.nq
    if not res["comparable"]:
        res["missing_in_pin"] = [n for n in names if not model.existJointName(n)]
        print(json.dumps(res, indent=1))
        return
    pin_q = [model.idx_qs[model.getJointId(n)] for n in names]
    pin_v = [model.idx_vs[model.getJointId(n)] for n in names]
    mj_q = [m.jnt_qposadr[j] for j in hinge]
    mj_v = [m.jnt_dofadr[j] for j in hinge]
    res["armature_pin"] = sorted({float(model.armature[i]) for i in pin_v})
    res["armature_mj"] = sorted({float(m.dof_armature[i]) for i in mj_v})

    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, m.nbody)]
    bodies = [b for b in bodies if model.existFrame(b)]
    res["bodies_compared"] = len(bodies)
    bid = {b: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies}
    tid_pin = model.getFrameId(TRUNK)

    rng = np.random.default_rng(0)
    keys = ["eef", "arm_bodies", "all_bodies", "eef_in_trunk", "bodies_in_trunk", "M_act", "g_act"]
    err = {k: [] for k in keys}
    samples = []
    for k in range(args.n):
        qj = np.zeros(len(hinge))
        pos, quat = np.array([0.0, 0.0, 0.4]), np.array([1.0, 0.0, 0.0, 0.0])
        if k:
            for i, j in enumerate(hinge):
                lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi)
                qj[i] = rng.uniform(lo, hi)
            pos = pos + rng.normal(0, 0.2, 3)
            quat = rng.normal(size=4)
            quat /= np.linalg.norm(quat)

        d.qpos[:] = 0
        d.qpos[:3], d.qpos[3:7] = pos, quat
        d.qpos[mj_q] = qj
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        M_mj = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, M_mj, d.qM)

        q = pin.neutral(model)
        q[:3] = pos
        q[3:7] = [quat[1], quat[2], quat[3], quat[0]]  # Pinocchio stores xyzw
        q[pin_q] = qj
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        M_pin = pin.crba(model, data, q)
        M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T
        g_pin = pin.computeGeneralizedGravity(model, data, q)

        T = data.oMf[tid_pin]
        R_mj, p_mj = d.xmat[bid[TRUNK]].reshape(3, 3), d.xpos[bid[TRUNK]]
        in_trunk_pin = lambda x: T.rotation.T @ (x - T.translation)
        in_trunk_mj = lambda x: R_mj.T @ (x - p_mj)
        p_body = lambda b: data.oMf[model.getFrameId(b)].translation

        err["eef"].append(np.linalg.norm(data.oMf[fid].translation - d.site_xpos[sid]))
        err["arm_bodies"].append(max(np.linalg.norm(p_body(b) - d.xpos[bid[b]]) for b in ARM_BODIES))
        err["all_bodies"].append(max(np.linalg.norm(p_body(b) - d.xpos[bid[b]]) for b in bodies))
        err["eef_in_trunk"].append(np.linalg.norm(in_trunk_pin(data.oMf[fid].translation)
                                                  - in_trunk_mj(d.site_xpos[sid])))
        err["bodies_in_trunk"].append(max(np.linalg.norm(in_trunk_pin(p_body(b)) - in_trunk_mj(d.xpos[bid[b]]))
                                          for b in bodies))
        err["M_act"].append(np.abs(M_pin[np.ix_(pin_v, pin_v)] - M_mj[np.ix_(mj_v, mj_v)]).max())
        err["g_act"].append(np.abs(g_pin[pin_v] - d.qfrc_bias[mj_v]).max())
        if k < 3:
            samples.append({"q_arm": qj[[names.index(n) for n in ARM_JOINTS]].round(3).tolist(),
                            "eef_in_trunk_pin": in_trunk_pin(data.oMf[fid].translation).round(4).tolist(),
                            "eef_in_trunk_mj": in_trunk_mj(d.site_xpos[sid]).round(4).tolist()})

    res["n_configs"] = args.n
    res["max_error"] = {k: float(max(v)) for k, v in err.items()}
    res["samples"] = samples
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "joints"}, indent=1))


if __name__ == "__main__":
    main()
