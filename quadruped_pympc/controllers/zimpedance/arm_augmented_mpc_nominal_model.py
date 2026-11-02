# Description: This file contains the class Centroidal_Model that defines the
# prediction model used by the MPC

# Authors: Giulio Turrisi -

import time
import unittest
import casadi as cs

import numpy as np
from acados_template import AcadosModel

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(dir_path)
sys.path.append(dir_path + '/../../')

from liecasadi import SO3

from quadruped_pympc import config

use_adam = False
use_fixed_inertia = True
use_centroidal_model = True
# import adam

# from adam.casadi import KinDynComputations
## Import of Pinocchio 
import pinocchio as pin
import pinocchio.casadi as cpin

def read_arm_impedance(xml_path, joint_names):
    """Joint stiffness, damping and spring reference of the named MJCF joints.

    Pinocchio parses damping but not stiffness or springref, so the attributes are read from the
    XML directly. Values missing on a joint fall back to the top-level <default><joint>, then to
    MuJoCo's default of 0.
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(xml_path).getroot()
    default = root.find('default/joint')
    joints = {j.get('name'): j for j in root.iter('joint')}

    def attr(joint, key):
        value = joint.get(key)
        if value is None and default is not None:
            value = default.get(key)
        return float(value) if value is not None else 0.0

    missing = [n for n in joint_names if n not in joints]
    if missing:
        raise ValueError(f"arm joints {missing} not found in {xml_path}")
    return tuple(np.array([attr(joints[n], key) for n in joint_names])
                 for key in ('stiffness', 'damping', 'springref'))


# Class that defines the prediction model of the NMPC
class Arm_Augmented_Centroidal_Model:
    def __init__(self,) -> None: 




        self.joints_name_list_ = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
               'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 
               'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
               'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
               'passive_arm_joint_0','passive_arm_joint_1','passive_arm_joint_2'
               ]

        model_path_ = config.mpc_params.get('armpc_model_xml') or dir_path + '/new_urdf_export.xml'
        self.model_path = model_path_

        model_result = pin.buildModelFromMJCF(
            model_path_, pin.JointModelFreeFlyer(), "root_joint"
        )
        # Pinocchio versions with MJCF constraint support return
        # (model, constraint_models); older versions return only model.
        model = model_result[0] if isinstance(model_result, tuple) else model_result
        data  = model.createData()

        # Physical parameters of the prediction model, read from the same MJCF.
        # The passive arm is the last three joints.
        self.total_mass = float(pin.computeTotalMass(model))
        self.arm_joint_names = list(model.names[-3:])
        self.arm_stiffness, self.arm_damping, self.arm_spring_rest = read_arm_impedance(
            model_path_, self.arm_joint_names)
        cmodel = cpin.Model(model)
        cdata  = cmodel.createData()
        q = cs.SX.sym("q", model.nq, 1)   # config (7 + n_joints)
        v = cs.SX.sym("v", model.nv, 1)   # velocity (6 + n_joints)
        M = cpin.crba(cmodel, cdata, q)
        h = cpin.rnea(cmodel, cdata, q, v, cs.SX.zeros(model.nv, 1))
        # Wrap as CasADi functions
        self.M_fun = cs.Function("M_fun", [q], [M])
        self.h_fun = cs.Function("h_fun", [q, v], [h])

        frame_id = model.getFrameId("eef")   # link/frame name from URDF
        J = cpin.computeFrameJacobian(cmodel, cdata, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.jac_arm_fun = cs.Function("jac_arm_fun", [q], [J])

        # Compute end-effector position
        # end-effector position in BASE frame
        frame_id_eef = model.getFrameId("eef")
        self.fwd_kin=cpin.forwardKinematics(cmodel, cdata, q)
        self.frame_update=cpin.updateFramePlacements(cmodel, cdata)

        p_ee_sym = cdata.oMf[frame_id_eef].translation  # 3×1 SX
        # ## create a function that computes the forward kinematics of the arm end effector
        self.fk_arm_fun = cs.Function("fk_arm_fun", [q], [p_ee_sym])

        # frame_update=cpin.updateFramePlacements(cmodel, cdata)
        # p_ee_W = cdata.oMf[frame_id_eef].translation

        # Define state and its casadi variables
        com_position_x = cs.SX.sym("com_position_x")
        com_position_y = cs.SX.sym("com_position_y")
        com_position_z = cs.SX.sym("com_position_z")

        com_velocity_x = cs.SX.sym("com_velocity_x")
        com_velocity_y = cs.SX.sym("com_velocity_y")
        com_velocity_z = cs.SX.sym("com_velocity_z")
        
        roll = cs.SX.sym("roll", 1, 1)
        pitch = cs.SX.sym("pitch", 1, 1)
        yaw = cs.SX.sym("yaw", 1, 1)
        omega_x = cs.SX.sym("omega_x", 1, 1)
        omega_y = cs.SX.sym("omega_y", 1, 1)
        omega_z = cs.SX.sym("omega_z", 1, 1)
        
        foot_position_fl = cs.SX.sym("foot_position_fl", 3, 1)
        foot_position_fr = cs.SX.sym("foot_position_fr", 3, 1)
        foot_position_rl = cs.SX.sym("foot_position_rl", 3, 1)
        foot_position_rr = cs.SX.sym("foot_position_rr", 3, 1)


        com_position_z_integral = cs.SX.sym("com_position_z_integral")
        com_velocity_x_integral = cs.SX.sym("com_velocity_x_integral")
        com_velocity_y_integral = cs.SX.sym("com_velocity_y_integral")
        com_velocity_z_integral = cs.SX.sym("com_velocity_z_integral")
        roll_integral = cs.SX.sym("roll_integral")
        pitch_integral = cs.SX.sym("pitch_integral")

        ### ARM AUGMENTATION
        q_arm_1 = cs.SX.sym("q_arm_1")
        q_arm_2 = cs.SX.sym("q_arm_2")
        q_arm_3 = cs.SX.sym("q_arm_3")



        q_dot_arm_1 = cs.SX.sym("q_dot_arm_1")
        q_dot_arm_2 = cs.SX.sym("q_dot_arm_2")
        q_dot_arm_3 = cs.SX.sym("q_dot_arm_3")


        self.states = cs.vertcat(com_position_x,
                            com_position_y,
                            com_position_z,
                            com_velocity_x,
                            com_velocity_y,
                            com_velocity_z,
                            roll,
                            pitch,
                            yaw,
                            omega_x,
                            omega_y,
                            omega_z,
                            foot_position_fl,
                            foot_position_fr,
                            foot_position_rl,
                            foot_position_rr,
                            com_position_z_integral,
                            com_velocity_x_integral,
                            com_velocity_y_integral,
                            com_velocity_z_integral,
                            roll_integral,
                            pitch_integral,
                            q_arm_1,
                            q_arm_2,
                            q_arm_3,
                            q_dot_arm_1,
                            q_dot_arm_2,
                            q_dot_arm_3
                            )
        


        # Define state dot 
        ### ARM AUGMENTATION
        self.states_dot = cs.vertcat(cs.SX.sym("linear_com_vel", 3, 1), 
                                     cs.SX.sym("linear_com_acc", 3, 1), 
                                     cs.SX.sym("euler_rates_base", 3, 1), 
                                     cs.SX.sym("angular_acc_base", 3, 1),
                                     cs.SX.sym("linear_vel_foot_FL", 3, 1),
                                     cs.SX.sym("linear_vel_foot_FR", 3, 1),
                                     cs.SX.sym("linear_vel_foot_RL", 3, 1),
                                     cs.SX.sym("linear_vel_foot_RR", 3, 1),
                                     cs.SX.sym("linear_com_vel_z_integral", 1, 1),
                                     cs.SX.sym("linear_com_acc_integral", 3, 1),
                                     cs.SX.sym("euler_rates_roll_integral", 1, 1),
                                     cs.SX.sym("euler_rates_pitch_integral", 1, 1),
                                     cs.SX.sym("q_dot_arm", 3, 1),
                                     cs.SX.sym("q_ddot_arm", 3, 1)
                                     )
        


        # Define input and its casadi variables
        foot_velocity_fl = cs.SX.sym("foot_velocity_fl", 3, 1)
        foot_velocity_fr = cs.SX.sym("foot_velocity_fr", 3, 1)
        foot_velocity_rl = cs.SX.sym("foot_velocity_rl", 3, 1)
        foot_velocity_rr = cs.SX.sym("foot_velocity_rr", 3, 1)

        foot_force_fl = cs.SX.sym("foot_force_fl", 3, 1)
        foot_force_fr = cs.SX.sym("foot_force_fr", 3, 1)
        foot_force_rl = cs.SX.sym("foot_force_rl", 3, 1)
        foot_force_rr = cs.SX.sym("foot_force_rr", 3, 1)

        self.inputs = cs.vertcat(foot_velocity_fl, 
                            foot_velocity_fr, 
                            foot_velocity_rl, 
                            foot_velocity_rr, 
                            foot_force_fl, 
                            foot_force_fr, 
                            foot_force_rl, 
                            foot_force_rr)
        

        # Usefull for debug what things goes where in y_ref in the compute_control function,
        # because there are a lot of variables
        self.y_ref = cs.vertcat(self.states, self.inputs)
        

        # Define acados parameters that can be changed at runtine
        self.stanceFL = cs.SX.sym("stanceFL", 1, 1)
        self.stanceFR = cs.SX.sym("stanceFR", 1, 1)
        self.stanceRL = cs.SX.sym("stanceRL", 1, 1)
        self.stanceRR = cs.SX.sym("stanceRR", 1, 1)
        self.stance_param = cs.vertcat(self.stanceFL , self.stanceFR , self.stanceRL , self.stanceRR)


        self.mu_friction = cs.SX.sym("mu_friction", 1, 1)
        self.stance_proximity = cs.SX.sym("stanceProximity", 4, 1)
        self.base_position = cs.SX.sym("base_position", 3, 1)
        self.base_yaw = cs.SX.sym("base_yaw", 1, 1)

        self.external_wrench = cs.SX.sym("external_wrench", 6, 1)

        self.mass = cs.SX.sym("mass", 1, 1)

        self.eef_position = cs.SX.sym("eef_position", 3, 1)
        # Measured EEF minus fk_arm_fun at the current state: the FK root is the MPC position
        # state (CoM in x, y), not the trunk origin, so the EEF cost needs this alignment.
        self.eef_fk_offset = cs.SX.sym("eef_fk_offset", 3, 1)

        ### ARM AUGMENTATION
        self.k = cs.SX.sym("k",3,1)
        self.d = cs.SX.sym("D",3,1)

        self.q_arm_rest_1 = cs.SX.sym("q_arm_rest_1")
        self.q_arm_rest_2 = cs.SX.sym("q_arm_rest_2")
        self.q_arm_rest_3 = cs.SX.sym("q_arm_rest_3")

        self.joint_position = np.array([
            0.0, 0.9, -1.8, 
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8, 
            0.0, 0.9, -1.8,
            0.0, 0.0, 0.0 #for now Ill fix somw random pos 
        ])

        self.joint_vel= np.zeros(len(self.joints_name_list_ ))

    def compute_eef_position_world(self, com_position, roll, pitch, yaw, q_arm, base_position_offset):
        joint_position_update = cs.vertcat(
            self.joint_position[0:12],
            q_arm
        )

        cy = cs.cos(yaw * 0.5)
        sy = cs.sin(yaw * 0.5)
        cp = cs.cos(pitch * 0.5)
        sp = cs.sin(pitch * 0.5)
        cr = cs.cos(roll * 0.5)
        sr = cs.sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        base_quat = cs.vertcat(qx, qy, qz, qw)

        full_joint_pos_update = cs.vertcat(
            com_position + base_position_offset,
            base_quat,
            joint_position_update
        )

        return self.fk_arm_fun(full_joint_pos_update)




    def compute_b_R_w(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        #Z Y X rotations!
        #world to base
        Rx = cs.SX.eye(3)
        Rx[0,0] = 1   
        Rx[0,1] = 0
        Rx[0,2] = 0
        Rx[1,0] = 0
        Rx[1,1] = cs.cos(roll)
        Rx[1,2] = cs.sin(roll)
        Rx[2,0] = 0
        Rx[2,1] = -cs.sin(roll)
        Rx[2,2] = cs.cos(roll)
                
        Ry = cs.SX.eye(3)
        Ry[0,0] = cs.cos(pitch)
        Ry[0,1] = 0
        Ry[0,2] = -cs.sin(pitch)
        Ry[1,0] = 0
        Ry[1,1] = 1
        Ry[1,2] = 0
        Ry[2,0] = cs.sin(pitch)
        Ry[2,1] = 0
        Ry[2,2] = cs.cos(pitch)

        Rz = cs.SX.eye(3)
        Rz[0,0] = cs.cos(yaw)
        Rz[0,1] = cs.sin(yaw)
        Rz[0,2] = 0
        Rz[1,0] = -cs.sin(yaw)
        Rz[1,1] = cs.cos(yaw)
        Rz[1,2] = 0
        Rz[2,0] = 0
        Rz[2,1] = 0
        Rz[2,2] = 1
        
        b_R_w = Rx@Ry@Rz
        return b_R_w


    def forward_dynamics(self, states: np.ndarray, inputs: np.ndarray, param: np.ndarray) -> cs.SX:
        """
        This method computes the symbolic forward dynamics of the robot. It is used inside
        Acados to compute the prediction model. It fill the same variables as the one in
        self.states_dot.

        Args:
            states: A numpy array of shape (29,) representing the current state of the robot.
            inputs: A numpy array of shape (29,) representing the inputs to the robot.
            param: A numpy array of shape (4,) representing the parameters (contact status) of the robot.

        Returns:
            A CasADi SX object of shape (29,) representing the predicted state of the robot.
        """
        
        # Saving for clarity a bunch of variables
        foot_velocity_fl = inputs[0:3]
        foot_velocity_fr = inputs[3:6]
        foot_velocity_rl = inputs[6:9]
        foot_velocity_rr = inputs[9:12]
        foot_force_fl = inputs[12:15]
        foot_force_fr = inputs[15:18]
        foot_force_rl = inputs[18:21]
        foot_force_rr = inputs[21:24]

        com_position = states[0:3]
        com_velocity = states[3:6]
        foot_position_fl = states[12:15]
        foot_position_fr = states[15:18]
        foot_position_rl = states[18:21]
        foot_position_rr = states[21:24]

        q_arm= states[30:33] #arm joint position
        q_dot_arm = states[33:36] #arm joint velocity
        
        stanceFL = param[0]
        stanceFR = param[1]
        stanceRL = param[2]
        stanceRR = param[3]
        stance_proximity_FL = param[5]
        stance_proximity_FR = param[6]
        stance_proximity_RL = param[7]
        stance_proximity_RR = param[8]

        k = param[19:22] #spring constant
        d = param[22:25] #damping constant
        roll = states[6]
        pitch = states[7]
        yaw = states[8]
        
        b_R_w = self.compute_b_R_w(roll, pitch, yaw)

        omega = states[9:12]  # angular velocity in the base frame

        # roll = states[6]
        # pitch = states[7]
        # yaw = states[8]
        
        # b_R_w = self.compute_b_R_w(roll, pitch, yaw)
        # omega_rotated = b_R_w @ omega



        ##=============================== JOINTS POSITION AND VELOCITY ===========================
        #update the joint position before passing the value to adam
        joint_position_update = cs.vertcat(
            self.joint_position[0:12], #legs joint position
            q_arm #arm joint position
        )

        self.joint_position_update = joint_position_update
        joint_vel = cs.vertcat(
            self.joint_vel[0:12], #legs joint velocity
            q_dot_arm #arm joint velocity
        )


        # pinocchio takes as input the base pos + the joint position
        # convert rpy to quaternion
        base_quat = cs.SX.zeros(4,1) # x y z w
        cy = cs.cos(yaw * 0.5)
        sy = cs.sin(yaw * 0.5)
        cp = cs.cos(pitch * 0.5)
        sp = cs.sin(pitch * 0.5)
        cr = cs.cos(roll * 0.5)
        sr = cs.sin(roll * 0.5)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy          
        # Pinocchio wants [x,y,z,w]
        base_quat = cs.vertcat(qx, qy, qz, qw)

        self.full_joint_pos_update = cs.vertcat(
            com_position, base_quat, joint_position_update
        )

        # Full 9-DOF (6 base + 3 arm) coupled dynamics, solved exactly via
        # a Schur-complement reduction instead of a dense 9x9 factorization:
        # one 3x3 solve against the arm block M_s (reused for both the
        # coupling columns and the rhs), then one 6x6 solve against the
        # Schur-reduced base block. This keeps the full bidirectional
        # coupling (M_bs: arm inertia reacts on the base; M_sb: base
        # acceleration excites the arm) at roughly the cost of the old
        # one-way approximation, which solved M_s alone (implicitly
        # assuming zero base acceleration) and only injected M_bs @ q_ddot_arm
        # into the base equation, dropping the M_sb @ a_base term entirely.
        #
        # M_arm/M_base_arm/B_arm/inertia below come from Pinocchio's
        # CRBA/RNEA on full_joint_pos_update/full_vel, whose floating-base
        # convention expresses both linear and angular base velocity in the
        # BASE/local frame (see base_linear_velocity below). The coupled
        # solve is therefore carried out entirely in base frame; only the
        # final linear acceleration is rotated back to world frame to match
        # the (world-frame) com_velocity/com_position state convention.
        # The previous version summed the base-frame tau_base_arm directly
        # into the world-frame contact_force_world (no rotation), and for
        # the angular part let it be rotated a second time by the
        # `b_R_w @ contact_moment_world` line below -- both were frame
        # bugs, fixed here by keeping everything in base frame until the
        # final rotation-back of the linear term.
        base_linear_velocity = b_R_w @ com_velocity
        self.full_vel = cs.vertcat(base_linear_velocity, omega, joint_vel)
        mass_matrix = self.M_fun(self.full_joint_pos_update)
        bias = self.h_fun(self.full_joint_pos_update, self.full_vel)
        M_arm = mass_matrix[18:21, 18:21]      # M_s   (3x3)
        M_base_arm = mass_matrix[0:6, 18:21]   # M_bs  (6x3); M_sb = M_base_arm.T (M symmetric)
        B_arm = bias[18:21]                    # h_s   (3x1)
        inertia = mass_matrix[3:6, 3:6]        # base angular block (3x3)
        self.mass = mass_matrix[0, 0]          # total mass (scalar)

        # p_eef = self.fk_arm_fun(self.full_joint_pos_update)  # 3×1 SX
        # self.eef_position_world = p_eef
        #instead the eef position is passed as parameter from the wb_interface




        # FINAL linear_com_vel STATE (1)
        linear_com_vel = states[3:6]


        contact_force_world = foot_force_fl @ stanceFL
        contact_force_world += foot_force_fr @ stanceFR
        contact_force_world += foot_force_rl @ stanceRL
        contact_force_world += foot_force_rr @ stanceRR
        gravity = cs.SX([0, 0, -9.81])

        # Start to write the component of euler_rates_base and angular_acc_base STATES
        roll = states[6]
        pitch = states[7]
        yaw = states[8]

        conj_euler_rates = cs.SX.eye(3)
        conj_euler_rates[1, 1] = cs.cos(roll)
        conj_euler_rates[2, 2] = cs.cos(pitch)*cs.cos(roll)
        conj_euler_rates[2, 1] = -cs.sin(roll)
        conj_euler_rates[0, 2] = -cs.sin(pitch)
        conj_euler_rates[1, 2] = cs.cos(pitch)*cs.sin(roll)


        contact_moment_world = (
            cs.skew(foot_position_fl - com_position)
            @ foot_force_fl @ stanceFL
        )
        contact_moment_world += (
            cs.skew(foot_position_fr - com_position)
            @ foot_force_fr @ stanceFR
        )
        contact_moment_world += (
            cs.skew(foot_position_rl - com_position)
            @ foot_force_rl @ stanceRL
        )
        contact_moment_world += (
            cs.skew(foot_position_rr - com_position)
            @ foot_force_rr @ stanceRR
        )





        # FINAL angular_acc_base STATE (4)



        ##################################################


        q_arm_rest = param[25:28]
        rs = (
            -B_arm
            - cs.diag(k) @ (q_arm - q_arm_rest)
            - cs.diag(d) @ q_dot_arm
        )
        # NOTE on the arm end-effector external-force term (the paper's
        # "-J_s^T f_e" in eq. 12 of the augmented-dynamics derivation):
        # intentionally NOT included here yet. It would need a raw 3D
        # end-effector force expressed in the same LOCAL_WORLD_ALIGNED
        # frame as jac_arm_fun's Jacobian, which is not currently plumbed
        # as an acados parameter -- the existing `external_wrench` param is
        # a 6D wrench already resolved at the base/CoM by
        # passive_arm_interface.py (skew(p_ee) @ f_ee), not compatible with
        # a direct J_s^T multiplication in the arm's own generalized
        # coordinates. Also flagging: if/when this term is added, the
        # correct sign looks to be +J_s^T f_e, matching PACC eq. 5/6's
        # convention, under which the existing force estimator is
        # self-consistently derived (J_ee^T f_hat_ee = -(tau_g+tau_s+tau_d)
        # at quasi-static equilibrium) -- not "-J_s^T f_e" as literally
        # written in eq. 12, which appears to be an inconsistent sign
        # relative to both the PACC estimator and this paper's own
        # ZMP section (f_ext enters f = m_tot(g - a_com) + f_ext with a "+").

        # Base "own" operator (M_b), block-diagonal in mass/inertia as in
        # the original reduced-order model, expressed in BASE frame to
        # match M_arm/M_base_arm's Pinocchio convention.
        Mb = cs.diagcat(self.mass * cs.SX.eye(3), inertia)

        F_grf_base = b_R_w @ contact_force_world
        M_grf_base = b_R_w @ contact_moment_world
        rb = cs.vertcat(
            F_grf_base + self.mass * (b_R_w @ gravity),
            M_grf_base - cs.skew(omega) @ inertia @ omega,
        )

        # Schur complement: one 3x3 factorization of M_arm, reused (via
        # multi-RHS solve) for both the M_sb coupling columns and rs.
        rhs_arm = cs.horzcat(M_base_arm.T, rs)      # [M_sb | rs], (3x7)
        sol_arm = cs.solve(M_arm, rhs_arm)          # M_s^-1 [M_sb | rs]
        Ms_inv_Msb = sol_arm[:, 0:6]                # M_s^-1 M_sb, (3x6)
        Ms_inv_rs = sol_arm[:, 6:7]                 # M_s^-1 rs,   (3x1)

        S = Mb - M_base_arm @ Ms_inv_Msb            # Schur complement (6x6)
        rhs_base = rb - M_base_arm @ Ms_inv_rs
        base_acc = cs.solve(S, rhs_base)            # [a_lin; alpha], BASE frame

        q_ddot_arm = Ms_inv_rs - Ms_inv_Msb @ base_acc

        linear_com_acc = b_R_w.T @ base_acc[0:3]    # rotate back to world frame
        angular_acc_base = base_acc[3:6]            # already base frame, matches `omega`






        # FINAL euler_rates_base STATE (3)
        euler_rates_base = cs.solve(conj_euler_rates, omega)

        # FINAL linear_foot_vel STATES (5,6,7,8)
        if(not config.mpc_params["use_foothold_optimization"]):
            foot_velocity_fl = foot_velocity_fl@0.0
            foot_velocity_fr = foot_velocity_fr@0.0
            foot_velocity_rl = foot_velocity_rl@0.0
            foot_velocity_rr = foot_velocity_rr@0.0
        linear_foot_vel_FL = foot_velocity_fl@(1-stanceFL)@(1-stance_proximity_FL)
        linear_foot_vel_FR = foot_velocity_fr@(1-stanceFR)@(1-stance_proximity_FR)
        linear_foot_vel_RL = foot_velocity_rl@(1-stanceRL)@(1-stance_proximity_RL)
        linear_foot_vel_RR = foot_velocity_rr@(1-stanceRR)@(1-stance_proximity_RR)

        # Integral states
        integral_states = states[24:30]
        integral_states[0] += states[2]
        integral_states[1] += states[3]
        integral_states[2] += states[4]
        integral_states[3] += states[5]
        integral_states[4] += roll
        integral_states[5] += pitch


 

        
        # The order of the return should be equal to the order of the states_dot
        return cs.vertcat(linear_com_vel,     # 3
                          linear_com_acc,     # 3
                          euler_rates_base,   # 3
                          angular_acc_base,   # 3 x 1
                          linear_foot_vel_FL, # 3 x 1 
                          linear_foot_vel_FR, # 3 x 1 
                          linear_foot_vel_RL, # 3 x 1 
                          linear_foot_vel_RR, # 3 x 1  ]
                          integral_states, # 6 x 1
                          q_dot_arm, # 3 x 1
                          q_ddot_arm # 3 x 1
                          )
    

        
    def export_robot_model(self,) -> AcadosModel:
        """
        This method set some general properties of the NMPC, such as the params,
        prediction mode, etc...! It will be called in arm_augmented_centroidal_nmpc.py
        """
 
        # dynamics
        self.param = cs.vertcat(self.stance_param, self.mu_friction, self.stance_proximity, self.base_position,
                                self.base_yaw, self.external_wrench,self.k,self.d,self.q_arm_rest_1,self.q_arm_rest_2,self.q_arm_rest_3,self.eef_position,
                                self.eef_fk_offset)
        f_expl = self.forward_dynamics(self.states, self.inputs, self.param)
        f_impl = self.states_dot - f_expl

        acados_model = AcadosModel()
        acados_model.f_impl_expr = f_impl
        acados_model.f_expl_expr = f_expl
        acados_model.x = self.states
        acados_model.xdot = self.states_dot
        acados_model.u = self.inputs
        acados_model.p = self.param
        acados_model.name = "arm_augmented_centroidal_model"
        acados_model.cost_y_expr = cs.vertcat(
            self.y_ref,
            self.compute_eef_position_world(
                self.states[0:3],
                self.states[6],
                self.states[7],
                self.states[8],
                self.states[30:33],
                self.base_position + self.eef_fk_offset,
            )
        )
        acados_model.cost_y_expr_e = cs.vertcat(
            self.states,
            self.compute_eef_position_world(
                self.states[0:3],
                self.states[6],
                self.states[7],
                self.states[8],
                self.states[30:33],
                self.base_position + self.eef_fk_offset,
            )
        )





        return acados_model
    
