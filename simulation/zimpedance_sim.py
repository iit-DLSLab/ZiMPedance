# Description: This script is used to simulate the full model of the robot in mujoco

# Authors:
# Giulio Turrisi, Daniel Ordonez

import time
import numpy as np
from tqdm import tqdm

# Gym and Simulation related imports
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr

# Config imports
from quadruped_pympc import config as cfg

# Helper functions for plotting
from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco
from gym_quadruped.utils.mujoco.visual import render_vector
from gym_quadruped.utils.mujoco.visual import render_sphere
import mujoco

# PyMPC controller imports
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper

from quadruped_pympc.interfaces.pacc_interface import PassiveArmInterface
from quadruped_pympc.helpers.terrain_estimator import TerrainEstimator


# HeightMap import
if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
    from gym_quadruped.sensors.heightmap import HeightMap

from quadruped_pympc.helpers.zmp_utils import plot_current_foot_prediction, compute_zmp,plot_zmp_vis,zmp_color,compute_center_of_pressure,compute_zmp_margin,compute_zmp_mpc,compute_com_acc,compute_capture_point

# from quadruped_pympc.helpers.zmp_utils import plot_current_foot_prediction, check_zmp_constraint_satisfaction_nominal,plot_zmp_vis,zmp_check_and_save,compute_zmp_margin_gpt,plot_zmp_trajectory
## ZMP Study Stuff
import pickle as pkl
import os

import copy

from gym_quadruped.utils.live_plotter import MujocoPlotter

from scipy.spatial.transform import Rotation as R

try:
    from simulation.zimpedance_tasks import (TASKS, apply_task, make_task_runtime, export_npz, run_metadata,
                                             set_arm_impedance, add_markers)
except ImportError:  # repo root not on sys.path
    from zimpedance_tasks import (TASKS, apply_task, make_task_runtime, export_npz, run_metadata,
                                  set_arm_impedance, add_markers)


def _smoothstep(x: float) -> float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _lerp_vec(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * np.asarray(start, dtype=float) + alpha * np.asarray(end, dtype=float)


def _make_render_transition_state(env, transition_cfg: dict) -> dict:
    transition_groups = set()
    transition_groups.update(transition_cfg.get("visible_geom_groups", []))
    transition_groups.update(transition_cfg.get("fade_out_geom_groups", []))
    transition_groups.update(transition_cfg.get("fade_in_geom_groups", []))
    transition_groups.update([2, 4, 5, 6])

    group_geom_ids = {}
    group_base_alpha = {}
    for group_id in sorted(transition_groups):
        geom_ids = np.where(env.mjModel.geom_group == group_id)[0]
        group_geom_ids[group_id] = geom_ids
        group_base_alpha[group_id] = env.mjModel.geom_rgba[geom_ids, 3].copy()
    return {
        "group_geom_ids": group_geom_ids,
        "group_base_alpha": group_base_alpha,
        "cfg": transition_cfg,
    }


def _update_render_transition(env, state: dict, base_pos: np.ndarray, eef_pos: np.ndarray, sim_time: float) -> dict:
    cfg = state["cfg"]
    close_duration = cfg.get("arm_only_duration", 2.0)
    transition_start_time = cfg.get("transition_start_time", close_duration)
    transition_duration = cfg.get("transition_duration", 3.0)
    reveal_start = cfg.get("robot_reveal_start", transition_start_time)
    reveal_duration = cfg.get("robot_reveal_duration", transition_duration)
    zmp_start = cfg.get("zmp_reveal_start", close_duration + 0.75 * transition_duration)
    zmp_duration = cfg.get("zmp_reveal_duration", 0.5)
    camera_bias = cfg.get("camera_bias", 1.0)

    close_pose = cfg["close_camera"]
    wide_pose = cfg["wide_camera"]

    if "lookat_world" in close_pose:
        close_lookat = np.asarray(close_pose["lookat_world"], dtype=float)
    else:
        close_lookat = eef_pos + np.asarray(close_pose.get("lookat_offset", np.zeros(3)), dtype=float)

    if "lookat_world" in wide_pose:
        wide_lookat = np.asarray(wide_pose["lookat_world"], dtype=float)
    else:
        wide_lookat = base_pos + np.asarray(wide_pose.get("lookat_offset", np.zeros(3)), dtype=float)

    cam_progress = np.clip((sim_time - transition_start_time) / max(transition_duration, 1e-9), 0.0, 1.0)
    cam_alpha = _smoothstep(np.power(cam_progress, camera_bias))
    reveal_alpha = _smoothstep((sim_time - reveal_start) / max(reveal_duration, 1e-9))
    zmp_alpha = _smoothstep((sim_time - zmp_start) / max(zmp_duration, 1e-9))

    env.viewer.cam.lookat[:] = _lerp_vec(close_lookat, wide_lookat, cam_alpha)
    env.viewer.cam.distance = float((1.0 - cam_alpha) * close_pose["distance"] + cam_alpha * wide_pose["distance"])
    env.viewer.cam.azimuth = float((1.0 - cam_alpha) * close_pose["azimuth"] + cam_alpha * wide_pose["azimuth"])
    env.viewer.cam.elevation = float((1.0 - cam_alpha) * close_pose["elevation"] + cam_alpha * wide_pose["elevation"])

    visible_groups = cfg.get("visible_geom_groups")
    if visible_groups is not None:
        env.viewer.opt.geomgroup[:] = 0
        for group_id in visible_groups:
            if 0 <= group_id < len(env.viewer.opt.geomgroup):
                env.viewer.opt.geomgroup[group_id] = 1

    for group_id, geom_ids in state["group_geom_ids"].items():
        base_alpha = state["group_base_alpha"][group_id]
        if group_id in cfg.get("fade_out_geom_groups", []):
            env.mjModel.geom_rgba[geom_ids, 3] = base_alpha * (1.0 - reveal_alpha)
        elif group_id in cfg.get("fade_in_geom_groups", []):
            env.mjModel.geom_rgba[geom_ids, 3] = base_alpha * reveal_alpha
        else:
            env.mjModel.geom_rgba[geom_ids, 3] = base_alpha

    return {"robot_alpha": reveal_alpha, "zmp_alpha": zmp_alpha, "camera_alpha": cam_alpha}

# Function to update the end effector position based on the joint angles and base position

def update_eef_position(q,base_pos,base_ori_euler_xyz):
    #arm forward kinematics is all wrong the original one at least
    Pee_Barm= np.zeros((3, 1))  # End effector position in the base frame

    p0 = np.array([0.1183, 0.0, 0.074])
    p1 = np.array([0.0498, 0.0, 0.034])
    p2 = np.array([0.1817, 0.0, 0.1845])
    p3 = np.array([0.0388, 0.0, -0.25])

    Pee_Barm[0] = (p0[0] + p1[0]*np.cos(q[0]) - p1[1]*np.sin(q[0]) - p2[1]*np.sin(q[0]) - p3[1]*np.sin(q[0]) +
                  p2[0]*np.cos(q[0])*np.cos(q[1]) + p2[2]*np.cos(q[0])*np.sin(q[1]) +
                  p3[0]*np.cos(q[0])*np.cos(q[1])*np.cos(q[2]) + p3[2]*np.cos(q[0])*np.cos(q[1])*np.sin(q[2]) +
                  p3[2]*np.cos(q[0])*np.cos(q[2])*np.sin(q[1]) - p3[0]*np.cos(q[0])*np.sin(q[1])*np.sin(q[2])) +0.15
    #changed the signb here to match arm movement along y
    Pee_Barm[1] = (p0[1] + p1[1]*np.cos(q[0]) + p2[1]*np.cos(q[0]) + p3[1]*np.cos(q[0]) +
                    
                  p1[0]*np.sin(q[0]) 
                  
                  + p2[0]*np.cos(q[1])*np.sin(q[0]) 
                  
                  + p2[2]*np.sin(q[0])*np.sin(q[1]) +
                  p3[0]*np.cos(q[1])*np.cos(q[2])*np.sin(q[0]) 
                  
                  + p3[2]*np.cos(q[1])*np.sin(q[0])*np.sin(q[2]) +
                  p3[2]*np.cos(q[2])*np.sin(q[0])*np.sin(q[1]) 
                  - p3[0]*np.sin(q[0])*np.sin(q[1])*np.sin(q[2])) 

    Pee_Barm[2] = (p0[2] + p1[2] + p3[2]*np.cos(q[1] + q[2]) - p3[0]*np.sin(q[1] + q[2]) +
                  p2[2]*np.cos(q[1]) - p2[0]*np.sin(q[1])) 
    
    Pee_Barm = Pee_Barm.flatten() #this is in the base frame
    
    # compute the end effector position in the world frame
    # rotatio nmatrix base to world frame
    rot_B_to_W = R.from_euler('xyz', base_ori_euler_xyz).as_matrix()
    Pee_world = rot_B_to_W @ Pee_Barm + base_pos   # Transform the end effector position to the world frame
    # print("Pee_world", Pee_world)
    eef_pos=np.array([Pee_world[0], Pee_world[1], Pee_world[2]])



    return eef_pos


VELOCITY_TRIM_MAX = 0.15  # m/s, bound of the velocity_trim_gain integrator


def run_simulation(process=0, num_episodes=500, return_dict=None, seed_number=0, base_vel_command_type="human",
                   render=True, render_transition_cfg=None, env_overrides=None,
                   task=None, export_path=None):
    """task: a preset from simulation/zimpedance_tasks.py, already applied to the config with
    apply_task(); it adds a base-velocity schedule and an EEF reference. export_path writes the
    trajectory log to an NPZ. The log is also returned, and pickled to collaborative_task_params['output_folder'] unless that is None.
    """
    
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(seed_number) 

    robot_name = cfg.robot
    hip_height = cfg.hip_height
    robot_leg_joints = cfg.robot_leg_joints
    robot_feet_geom_names = cfg.robot_feet_geom_names
    scene_name = cfg.simulation_params['scene']
    simulation_dt = cfg.simulation_params['dt']

    plotting = cfg.simulation_params['plotting']

    state_observables_names = ('base_pos', 'base_lin_vel', 'base_ori_euler_xyz', 'base_ori_quat_wxyz', 'base_ang_vel',
                               'qpos_js', 'qvel_js', 'tau_ctrl_setpoint',
                               'feet_pos_base', 'feet_vel_base', 'contact_state', 'contact_forces_base',)
    
    arm_joint_names = cfg.arm_joint_names

    amplitude =cfg.collaborative_task_params['amplitude']  # amplitude of the sinusoidal yaw motion (rad)
    frequency = cfg.collaborative_task_params['frequency']  # frequency of the sinusoidal yaw motion (Hz)
    omega = 2 * np.pi * frequency  # angular frequency (rad/s)
    apply_impulse = cfg.collaborative_task_params['apply_impulse']  # whether to apply impulse or not]
    apply_sin = cfg.collaborative_task_params['apply_sin']  # whether to apply sinusoidal yaw motion or not
    release_from_position = cfg.collaborative_task_params['release_from_position']

    sin_traj_tracking = cfg.collaborative_task_params['sin_traj_tracking']  # whether to track the sinusoidal trajectory with the reference velocity or not
    
    ##Live plotting stuff
    # x_lims = [(0, 500), (0, 500)]
    if plotting:
        ##Live plotting stuff
        y_lims = [(-0.4, 0.4), 
                  (-0.4, 0.4),
                  (-0.4, 0.4),
                  (-0.1, 0.2),
                  (-1, 1),
                  (-1, 1),
                  (-100,100),
                  (-100,100),
                  (-100,100)]  # Different Y ranges for demonstration
    
        plotter = MujocoPlotter(enable=True)
        plotter.create(figure_name="LivePlot", subplot_titles=["v-x","v-y","v-z",
                                                               "margin",
                                                               "ori-x","ori-z",
                                                               "force-x","force-y","force-z"], rows=3, cols=3, 
                       y_limits=y_lims, window_size=1000,tracking_plot=False)
        plotter.start()

  # Different Y ranges for demonstration

    # create empty lists for the data
    base_pos_x,base_pos_y,base_pos_z = [],[],[]
    base_ori_x,base_ori_y,base_ori_z = [],[],[]
    base_lin_vel_x,base_lin_vel_y,base_lin_vel_z = [],[],[]
    base_ang_vel_x,base_ang_vel_y,base_ang_vel_z = [],[],[]
    error_list = []

    # breakpoint()
    # Create the quadruped robot environment -----------------------------------------------------------
    env_kwargs = dict(
        robot=robot_name,
        hip_height=hip_height,
        legs_joint_names=robot_leg_joints,  # Joint names of the legs DoF
        feet_geom_name=robot_feet_geom_names,  # Geom/Frame id of feet
        scene=scene_name,
        sim_dt=simulation_dt,
        ref_base_lin_vel=(0, 0.6),  # pass a float for a fixed value
        ref_base_ang_vel=(-0.2, 0.2),  # pass a float for a fixed value
        ground_friction_coeff=1.5,  # pass a float for a fixed value
        base_vel_command_type=base_vel_command_type,  # "forward", "random", "forward+rotate", "human"
        state_obs_names=state_observables_names,  # Desired quantities in the 'state' vec
        frame_external_force=cfg.simulation_params['frame_external_force'],
    )
    if env_overrides is not None:
        env_kwargs.update(env_overrides)

    env = QuadrupedEnv(**env_kwargs)
    velocity_schedule, eef_task = (None, None) if task is None else make_task_runtime(
        task, cfg.mpc_params['horizon'], cfg.mpc_params['dt'])
    
    # breakpoint()
    # Passive arm interface

    passive_arm_interface = PassiveArmInterface(spring_gains= cfg.collaborative_task_params['spring_gains'],
                                                robot=        robot_name,
                                                joint_pos0=   np.array([0.0, 0.0, 0.0]),
                                                mass_matrix=  np.array([0.5, 0.5, 0.25]),
                                                damping_gains=cfg.collaborative_task_params['damping_gains'],
                                                com_offset= np.array([0,0,0]),
    )

    Jeef_arm =np.zeros((3, 3))  # Jacobian of the end effector in the base frame

    # Some robots require a change in the zero joint-space configuration. If provided apply it
    if cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], cfg.qpos0_js))

    env.reset(random=False)
    if render:
        env.render()  # Pass in the first render call any mujoco.viewer.KeyCallbackType
    render_transition_state = None
    # the camera transition needs its camera poses; a cfg with only timing keys skips it
    if render and render_transition_cfg is not None and "close_camera" in render_transition_cfg:
        render_transition_state = _make_render_transition_state(env, render_transition_cfg)

    motion_start_time = None if render_transition_cfg is None else render_transition_cfg.get("motion_start_time")
    max_sim_time = None if render_transition_cfg is None else render_transition_cfg.get("max_sim_time")


    # Initialization of variables used in the main control loop --------------------------------

    # Jacobian matrices
    jac_feet_prev = LegsAttr(*[np.zeros((3, env.mjModel.nv)) for _ in range(4)])
    jac_feet_dot = LegsAttr(*[np.zeros((3, env.mjModel.nv)) for _ in range(4)])
    # Torque vector
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    # Torque limits
    tau_soft_limits_scalar = 0.9
    # tau_limits = LegsAttr(
    #     FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL]*tau_soft_limits_scalar,
    #     FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR]*tau_soft_limits_scalar,
    #     RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL]*tau_soft_limits_scalar,
    #     RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR]*tau_soft_limits_scalar)

    
    
    # Feet positions and Legs order
    feet_pos = None
    feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ["FL", "FR", "RL", "RR"]

    terrain_computation = TerrainEstimator()
    # Create HeightMap -----------------------------------------------------------------------
    if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
        resolution_vfa = 0.04
        dimension_vfa = 7
        heightmaps = LegsAttr(FL=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mjModel=env.mjModel, mjData=env.mjData),
                        FR=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mjModel=env.mjModel, mjData=env.mjData),
                        RL=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mjModel=env.mjModel, mjData=env.mjData),
                        RR=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mjModel=env.mjModel, mjData=env.mjData))
    else:
        heightmaps = None
    ###

    update_arm=False


    Pee_Barm = np.zeros((3, 1))

    # Quadruped PyMPC controller initialization -------------------------------------------------------------
    mpc_frequency = cfg.mpc_params['horizon']


    quadrupedpympc_observables_names = ('ref_base_height', 'ref_base_angles', 'ref_feet_pos',
                                        'nmpc_GRFs', 'nmpc_footholds', 
                                        'swing_time', 'phase_signal', 'lift_off_positions')
    
    quadrupedpympc_wrapper = QuadrupedPyMPC_Wrapper(initial_feet_pos=env.feet_pos, legs_order=legs_order,
                                                    quadrupedpympc_observables_names = quadrupedpympc_observables_names)
    if task is not None:
        set_arm_impedance(env.mjModel, quadrupedpympc_wrapper, task['arm_stiffness'], task['arm_damping'])
    horizon = cfg.mpc_params['horizon']
    lift_off_pos= None
    legs_contact_pos = []
    geom_id_feet_pred=[-1,-1,-1,-1]
    geom_id_zmp=[-1,-1,-1,-1,-1]
    eef_pos_id=[-1] * horizon
    pred_base_pos_id=[-1] * horizon
    eef_pos_id_mjc=[-1,-1]
    cop_id=-1
    sin_traj_id = [-1] * horizon
    cached_predicted_eef_positions = [None] * horizon
    task_marker_ids = []  # viewer slots of the EEF task's path and target spheres
    velocity_trim_gain = cfg.simulation_params.get('velocity_trim_gain', 0.0)
    velocity_trim = np.zeros(2)  # heading-frame (x, y) added to the linear velocity command
    eef_task_reference = None

    def task_markers():
        if eef_task is None:
            return []
        eef = env.mjData.site('eef').xpos
        return eef_task.markers(eef) + eef_task.target_marker(eef_task_reference, eef)

    
    eef_adam=[-1,-1]
    
    base_pos_list,feet_pos_list,base_ori_list,contact_state_list,feet_GRF_list,end_effector_position_list,lift_off_pos_list,sim_time_histroy=[],[],[],[],[],[],[],[]
    base_ang_vel_list=[]
    base_lin_vel_list = []
    optimal_state_list = []  # Store the optimal state from the controller for ZMP study
    joint_pos_arm_list = []
    joint_vel_arm_list = []
    wrenche_est_list = []
    ref_linear_velocity_list = []
    ref_angular_velocity_list = []
    ref_orientation_list = []
    ref_position_list = []
    zmp_list,base_acc_list = [],[]
    margin_list = []
    eef_ref_list, mpc_status_list, qpos_list = [], [], []
    # # --------------------------------------------------------------
    render_cfg = render_transition_cfg if render_transition_cfg is not None else {}
    RENDER_FREQ = int(render_cfg.get("render_freq", 24))  # Hz
    N_EPISODES = num_episodes
    N_STEPS_PER_EPISODE = 2000 if env.base_vel_command_type != "human" else 4000
    if max_sim_time is not None:  # an explicit duration overrides the default episode length
        N_STEPS_PER_EPISODE = int(np.ceil(float(max_sim_time) / simulation_dt)) + 1
    last_render_time = time.time()

    state_obs_history, ctrl_state_history = [], []
    optimal_next_state = np.zeros(12)  # Initialize the optimal next state for the MPC controller
    mpc_update_period_steps = max(1, round(1 / (quadrupedpympc_wrapper.mpc_frequency * simulation_dt)))

    yaw_ref = 0.0
    sway_ref=0
    # generate a sinusoidal trajectory to track
    sway_ref_traj = amplitude * np.sin(omega * np.arange(8000) * 0.02)
    sway_dot_ref_traj = amplitude * omega * np.cos(omega * np.arange(N_STEPS_PER_EPISODE) * simulation_dt)
    ref_base_lin_vel_traj = np.zeros((N_STEPS_PER_EPISODE,3))

    # ratio = (0.02 * simulation_dt)  # ratio between the mpc frequency and the simulation frequency

    ratio = int(float(0.02) / float(simulation_dt))
    for episode_num in tqdm(range(N_EPISODES), desc="Episodes"):

        ep_state_obs_history, ep_ctrl_state_history = [], []
        step_start = time.time()
        time_start = time.time()
        robot_root_body_id = 1
        base_pos = env.base_pos
        robot_com = env.mjData.subtree_com[robot_root_body_id]
        robot_com_2 = env.mjData.subtree_com[robot_root_body_id]
        com_shift = base_pos - robot_com
        ref_joint_pos_arm = np.array([0.0, 0.0, 0.0])

        n_impulses = 8
        F_mag = amplitude
        start_after = 3.0        # s
        impulse_dt = 0.2        # s
        rest_dt = 2            # s
        angles = np.linspace(-np.pi/4, np.pi/4, n_impulses, endpoint=False)
        # schedule
        starts = start_after + np.arange(n_impulses) * (impulse_dt + rest_dt)
        ends   = starts + impulse_dt
        current_impulse = 0
        
        in_impulse = False
        disp_mag = np.zeros(n_impulses)
        impulse_geom_id = -1
        sin_geom_id = -1
        release_geom_id = -1
        release_force_active = False
        release_force_phase = 0
        release_sequence_done = False
        impulse_list=[]

        impulse_list=[]
        ref_base_ang_vel = np.array([0.0, 0.0, 0.0])  # Reference angular velocity in the world frame
        # ref_base_pos = np.zeros((25,3))
        # ref_base_lin_vel = np.zeros((25,3))  # Reference linear velocity in the world frame

        ref_sin_traj = np.zeros((horizon,3))
        ref_sin_dot_traj = np.zeros((horizon,3))

        ref_base_pos_traj=np.zeros((horizon,3))
        # ref_base_pos_traj[:, 2] = 0.35  # Set all z coordinates to 0.35
        ref_base_lin_vel=np.zeros((horizon,3))
        for _ in range(N_STEPS_PER_EPISODE):
            


            # Update value from SE or Simulator ----------------------
            feet_pos = env.feet_pos(frame='world')
            hip_pos = env.hip_positions(frame='world')
            base_lin_vel = env.base_lin_vel(frame='world')
            base_ang_vel = env.base_ang_vel(frame='world') #only this is wrong world is actually base!!! leave it like this the robot can rotate up to 90deg
            base_ori_euler_xyz = env.base_ori_euler_xyz
            base_pos = env.base_pos

            base_acc = copy.deepcopy(env.mjData.sensor('imu').data)

            robot_com = env.base_pos - com_shift

            
            # Get the reference base velocity in the world frame
            ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()
            if velocity_schedule is not None:  # task preset: command in the heading frame
                ref_lin_vel_heading, ref_yaw_rate = velocity_schedule(
                    env.mjData.time, eef_pos=env.mjData.site('eef').xpos, heading_R=env.heading_orientation_SO3)
                ref_base_lin_vel = env.heading_orientation_SO3 @ ref_lin_vel_heading
                ref_base_ang_vel = np.array([0.0, 0.0, ref_yaw_rate])
            current_sim_time = env.mjData.time
            if motion_start_time is not None and current_sim_time < float(motion_start_time):
                ref_base_lin_vel = np.zeros_like(ref_base_lin_vel)
                ref_base_ang_vel = np.zeros_like(ref_base_ang_vel)
            if velocity_trim_gain > 0:
                # integral trim on the heading-frame linear velocity; reset whenever the command is zero
                R_h = env.heading_orientation_SO3
                cmd_h = (R_h.T @ ref_base_lin_vel)[:2]
                if np.linalg.norm(cmd_h) < 1e-9:
                    velocity_trim[:] = 0.0
                else:
                    meas_h = (R_h.T @ base_lin_vel)[:2]
                    velocity_trim = np.clip(velocity_trim + velocity_trim_gain * (cmd_h - meas_h) * simulation_dt,
                                            -VELOCITY_TRIM_MAX, VELOCITY_TRIM_MAX)
                    ref_base_lin_vel = ref_base_lin_vel + R_h @ np.r_[velocity_trim, 0.0]
            
            # Get the inertia matrix
            if(cfg.simulation_params['use_inertia_recomputation']):
                inertia = env.get_base_inertia().flatten()  # Reflected inertia of base at qpos, in world frame
            else:
                inertia = cfg.inertia.flatten()

            # Get the qpos and qvel
            qpos, qvel = env.mjData.qpos, env.mjData.qvel
            joints_pos = LegsAttr(FL=qpos[7:10], FR=qpos[10:13],
                                  RL=qpos[13:16], RR=qpos[16:19])
            
            
            # Get Centrifugal, Coriolis, Gravity for the swing controller
            legs_mass_matrix = env.legs_mass_matrix
            legs_qfrc_bias = env.legs_qfrc_bias
            legs_qfrc_passive = env.legs_qfrc_passive

            # Compute feet jacobian
            feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
            
            # Compute jacobian derivatives of the contact points
            jac_feet_dot = (feet_jac - jac_feet_prev) / simulation_dt  # Finite difference approximation
            jac_feet_prev = feet_jac  # Update previous Jacobians
            
            # Compute feet velocities
            feet_vel = LegsAttr(**{leg_name: feet_jac[leg_name] @ env.mjData.qvel for leg_name in legs_order})

            # Idx of the leg
            legs_qvel_idx = env.legs_qvel_idx
            legs_qpos_idx = env.legs_qpos_idx

            # arm joint postion
            # # arm joint position update
            # arm_joint_pos =np.array([env.mjData.sensor(arm_joint_names[0]).data,
            #                          env.mjData.sensor(arm_joint_names[1]).data,
            #                          env.mjData.sensor(arm_joint_names[2]).data])
            
            # arm_joint_vel =np.array([env.mjData.sensor(arm_joint_names[0]).data,
            #                         env.mjData.sensor(arm_joint_names[1]).data,
            #                         env.mjData.sensor(arm_joint_names[2]).data])
            
            arm_joint_pos = np.array ([qpos[-3], qpos[-2], qpos[-1]])
            arm_joint_vel = np.array ([qvel[-3], qvel[-2], qvel[-1]])
            
            # arm_joint_pos[2]=0.0
            # arm_joint_vel[2]=0.0


            ### get inertias from arm 
            # list_arm_body_names=np.array(["follower/pa_link_0_ext", "follower/pa_link_1_ext", "follower/pa_link_2_ext"])
            # for i,f in enumerate(list_arm_body_names):
            #     body_id=mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_BODY.value, "follower/pa_link_0_ext")
            #     # print("site id",f,mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_SITE.value, f))
            #     body_in=env.mjModel.body_inertia[body_id]
            #     print("body inertia",f,body_in)

            # breakpoint()

            
            


            
            # Compute zmp -----------------------------------------------------------------------
            legs_contact_pos = []
            contact_state, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)

            wrench_estimate,Jeef_arm,end_effector_position=passive_arm_interface.calculate_force_estimates_damping(Jeef_arm,
                                                            np.array ([qpos[-3], qpos[-2], qpos[-1]]),
                                                            arm_joint_vel,
                                                            base_ori_euler_xyz, #replace this with actual rot matrix?
                                                            )

            step=env.step_num

            current_t=step*simulation_dt    
            current_step= env.step_num


            # #set a sinusoidal reference for the base
            if update_arm==True and current_t > 1.0 and sin_traj_tracking==True :
                # print("ratio", ratio)

                base_idx = int(current_step / ratio)

                ## set up a trajectory fro the base to follow on sway I need to feed the MPC the trajectory from current step up to the horizon (N=25)
                
                for k in range(horizon):
                    # if current_step + k < N_STEPS_PER_EPISODE:
                    #     print(current_step + k)
                    #     ref_sin_traj[k] = np.array([0, sway_ref_traj[current_step + k], 0.35])
                    #     ref_sin_dot_traj[k] = np.array([0, sway_dot_ref_traj[current_step + k], 0])
                    # else:
                    #     ref_sin_traj[-1] = np.array([0, sway_ref_traj[-1], 0.35])
                    #     ref_sin_dot_traj[-1] = np.array([0, sway_dot_ref_traj[-1], 0])
                    idx = base_idx + k  # each node = one MPC step ahead (0.02 s)
                    # sim_idx = idx * ratio  # convert back to simulation index
                    sim_idx = int((base_idx + k) * ratio)
                    # if sim_idx < len(sway_ref_traj):
                    if sim_idx < N_STEPS_PER_EPISODE:
                        # ref_sin_traj[k] = np.array([0, sway_ref_traj[sim_idx], 0.35])
                        # ref_sin_dot_traj[k] = np.array([0, sway_dot_ref_traj[sim_idx], 0])
                        ref_sin_traj[k] = np.array([0, 0,sway_ref_traj[sim_idx]])
                        ref_sin_dot_traj[k] = np.array([0, 0,sway_dot_ref_traj[sim_idx]])
                    else:
                        # ref_sin_traj[k] = ref_sin_traj[-1]
                        # ref_sin_dot_traj[k] = ref_sin_dot_traj[-1]
                        ref_sin_traj[k] = np.array([0, 0,sway_ref_traj[-1]])
                        ref_sin_dot_traj[k] = np.array([0, 0,sway_dot_ref_traj[-1]])

                ref_base_lin_vel_traj = ref_sin_dot_traj
                ref_base_pos_traj = ref_sin_traj

                # print("ref base ",ref_base_pos_traj)

                
            #     time_sin = step * simulation_dt


            #     yaw_ref = amplitude * np.sin(omega* time_sin)
            #     yaw_ref_vel = amplitude * omega * np.cos(omega * time_sin) 

            #     # ref_base_ang_vel = np.array([0.0, 0.0, yaw_ref_vel])  # Reference angular velocity in the world frame
            #     sway_ref = yaw_ref
            #     sway_ref_vel = yaw_ref_vel
            #     ref_base_lin_vel = np.array([0.0, yaw_ref_vel, 0.0])  # Reference linear velocity in the world frame


            
            end_effector_position_save=copy.deepcopy(env.mjData.site('eef').xpos)
            # robot_com=env.com

            # MPC position: the current whole-body CoM in x, y (the arm pose moves it; a constant
            # offset taken at reset biased the pitch), and the base height in z
            robot_com = env.mjData.subtree_com[robot_root_body_id].copy()
            robot_com[2] = base_pos[2]
            if cfg.simulation_params.get('com_centred_footholds', False):
                com_offset = np.r_[robot_com[:2] - base_pos[:2], 0.0]
                hip_pos = LegsAttr(**{leg: hip_pos[leg] + com_offset for leg in legs_order})
            

            eef_task_reference = None
            if eef_task is not None and update_arm:  # starts after the arm calibration at 1 s
                eef_task_reference = eef_task.reference(env.mjData.time, end_effector_position_save, base_pos)

            tau,optimal_next_state = quadrupedpympc_wrapper.compute_actions(robot_com, base_lin_vel, base_ori_euler_xyz, base_ang_vel,
                                                    feet_pos, hip_pos, joints_pos,
                                                    heightmaps,
                                                    legs_order, simulation_dt, ref_base_lin_vel, ref_base_ang_vel,
                                                    env.step_num, qpos, qvel, feet_jac, jac_feet_dot, feet_vel, legs_qfrc_passive,
                                                    legs_qfrc_bias,
                                                    legs_mass_matrix, legs_qpos_idx, legs_qvel_idx, tau, inertia,
                                                    arm_joint_pos, arm_joint_vel,wrench_estimate,ref_joint_pos_arm, end_effector_position_save,
                                                    ref_base_pos_traj,ref_base_lin_vel_traj,
                                                    eef_task_reference=eef_task_reference)
            if render_transition_cfg is not None and render_transition_cfg.get("render_predicted_eef", False):
                if env.step_num % mpc_update_period_steps == 0 and isinstance(optimal_next_state, (list, tuple)):
                    prediction_stride = max(1, int(render_transition_cfg.get("predicted_eef_stride", 2)))
                    max_points = int(render_transition_cfg.get("predicted_eef_max_points", horizon))
                    prediction_len = min(horizon, len(optimal_next_state), max_points)
                    copy_data = copy.deepcopy(env.mjData)
                    cached_predicted_eef_positions = [None] * horizon
                    for i in range(0, prediction_len, prediction_stride):
                        pred_state = np.asarray(optimal_next_state[i], dtype=float)
                        pred_base_pos = pred_state[0:3].copy()
                        pred_base_ori_rpy = pred_state[6:9].copy()
                        pred_base_quat = R.from_euler('zyx', pred_base_ori_rpy).as_quat()
                        copy_data.qpos[0:3] = pred_base_pos
                        copy_data.qpos[3:7] = pred_base_quat[::-1]
                        mujoco.mj_fwdPosition(env.mjModel, copy_data)
                        cached_predicted_eef_positions[i] = copy_data.site('eef').xpos.copy()
            # print("base orientation", base_ori_euler_xyz)
            # print("Base angular velocity", base_ang_vel)
            
            # one step prediction error


            # breakpoint()
            # Limit tau between tau_limits
            for leg in ["FL", "FR", "RL", "RR"]:
                # tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
                tau[leg] = np.clip(tau[leg], -1000, 1000)

            quadrupedpympc_observables = quadrupedpympc_wrapper.get_obs()

            

            lift_off_pos = quadrupedpympc_observables["lift_off_positions"]
            # Set control and mujoco step ----------------------------------------------------------------------
            action = np.zeros(env.mjModel.nu)
            
            action[env.legs_tau_idx.FL] = tau.FL
            action[env.legs_tau_idx.FR] = tau.FR
            action[env.legs_tau_idx.RL] = tau.RL
            action[env.legs_tau_idx.RR] = tau.RR
            
        ####================================================================== ZMP STUFF ========================
            ## temporary fix adding terrain estimation here to adjust zmpplot in real time
            terrain_roll, \
            terrain_pitch, \
            terrain_height = terrain_computation.compute_terrain_estimation(
            base_position=base_pos,
            yaw=base_ori_euler_xyz[2],
            feet_pos=quadrupedpympc_observables["lift_off_positions"],
            current_contact=None)

            nmpc_footholds=quadrupedpympc_observables["nmpc_footholds"]
            # ref_feet_pos=quadrupedpympc_observables["ref_feet_pos"]
            # lift_off_pos=quadrupedpympc_observables["lift_off_positions"]
            contact_state, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)
            # eef_pos_mjco=np.array ([qpos[-3], qpos[-2], qpos[-1]])
            eef_pos_mjco=env.mjData.site('eef').xpos
            
            base_pos_controller = optimal_next_state[1][0:3]
            zmp=compute_zmp(np.array([robot_com[0], robot_com[1], robot_com[2] - terrain_height]),
                            base_acc,
                            base_ori_euler_xyz,
                            wrench_estimate,
                            eef_pos_mjco,
                            )
            # zmp=-1 * zmp
            zmp_margin=compute_zmp_margin(zmp,nmpc_footholds,contact_state)
            cop=compute_center_of_pressure(np.array([robot_com[0],robot_com[1],0]),base_ori_euler_xyz,feet_pos,feet_GRF)
            # cop_list.append(cop)
            # zmp_list.append(zmp)
              

            # action_noise = np.random.normal(0, 2, size=env.mjModel.nu)
            # action += action_noise

            
            # # lock base in place
            # env.mjData.qpos[0:3] = np.array([0,0,0.35])  # lock base position
            # env.mjData.qpos[3:7] = np.array([1,0,0,0])  # lock base orientation
            # #lock legs
            # env.mjData.qpos[7:19] = np.array([
            #             0.0, 0.9, -1.8, 
            #             0.0, 0.9, -1.8,
            #             0.0, 0.9, -1.8, 
            #             0.0, 0.9, -1.8,
            # ])  # lock legs positio
            # #set qvel at zero for base and leg
            # env.mjData.qvel[0:6] = np.zeros(6)  # lock base velocity
            # env.mjData.qvel[6:18] = np.zeros(12)  # lock legs velocity
            # action = np.zeros(env.mjModel.nu)

            # # # do somethign after 1 second
            # if env.mjData.time > 1 and env.mjData.time < 2:
            #     #set arm position 
            #     # env.mjData.qpos[19:21] = np.array([0., 0.])
            #     env.mjData.qpos[-1] = -0.8  
            


            state, reward, is_terminated, is_truncated, info = env.step(action=action)
            if release_sequence_done:
                is_truncated = True
            
            # Store the history of observations and control -------------------------------------------------------
            ep_state_obs_history.append(state)
            base_lin_vel_err = ref_base_lin_vel - base_lin_vel
            base_ang_vel_err = ref_base_ang_vel - base_ang_vel
            base_poz_z_err = quadrupedpympc_observables["ref_base_height"] - base_pos[2]
            ctrl_state = np.concatenate((base_lin_vel_err, base_ang_vel_err, [base_poz_z_err], quadrupedpympc_observables["phase_signal"]))
            ep_ctrl_state_history.append(ctrl_state)
            #checl if adam matrix is equal to the mujoco matrix
            #mujoco 
            # lol,lol2,mass_mat_adam,bias_mat= compute_forward_kinematics_arm(qpos,qvel)
            # dst=np.eye(21)
            

            # mujoco.mj_fullM(env.mjModel,dst,mujoco_mass_matrix)
            # bias_mat_array = np.array(bias_mat).flatten()
            # mujoco_bias_mat = env.mjData.qfrc_bias

            # n = 20
            # M = np.zeros((n, n))
            # k = 0
            # for i in range(n):
            #     for j in range(i+1):
            #         M[i, j] = mujoco_mass_matrix[k]      # fill lower triangle
            #         M[j, i] = mujoco_mass_matrix[k]      # mirror to upper triangle
            #         k += 1
            
            # print(M)
            # breakpoint()

            #lets reduce the matrices to the base + arm srbd model
            # mat_base_arm_mjc = np.zeros((9,9))z
            # M_base_arm = dst[18:21,0:6] 
            # mat_base_arm_mjc[0:6,0:6] = dst[0:6,0:6]
            # mat_base_arm_mjc[0:6, 6:9] = M_base_arm.T
            # mat_base_arm_mjc[6:9, 0:6] = M_base_arm
            # mat_base_arm_mjc[6:9,6:9] = dst[18:21, 18:21] 
            # # same with adam matrix
            # mat_base_arm_adam = np.zeros((9,9))
            # M_base_arm_adam = mass_mat_adam[18:21,0:6]
            # mat_base_arm_adam[0:6,0:6] = mass_mat_adam[0:6,0:6]
            # mat_base_arm_adam[0:6, 6:9] = M_base_arm_adam.T
            # mat_base_arm_adam[6:9, 0:6] = M_base_arm_adam
            # mat_base_arm_adam[6:9,6:9] = mass_mat_adam[18:21, 18:21]
            # print("Mass matrix mismatch!")
            # print("Mujoco mass matrix:","\n", dst)
            # print("Adam mass matrix:","\n", mass_mat_adam)
            # print("Bias matrix mujoco", mujoco_bias_mat)
            # print("Bias matrix adam", bias_mat_array)
            # print("qvel", env.mjData.qvel)


            # #check if the matrices are the same
            # if not np.allclose(mat_base_arm_mjc, mat_base_arm_adam,rtol=0.005):
            #     print("Mass matrix mismatch!")
            #     print("Mujoco mass matrix:","\n", mat_base_arm_mjc)
            #     print("Adam mass matrix:","\n", mat_base_arm_adam)
            #     # #lets check block by bock
            #     # mass_adam_base = mass_mat_adam[0:6, 0:6]
            #     # mass_mujoco_base = dst[0:6, 0:6]
            #     # print("base matrices","\n", np.c_[mass_adam_base, mass_mujoco_base])
            #     breakpoint()
            


            # Store the data for zmp study -----------------------------------------------------------------------
            sim_time_histroy.append(env.mjData.time)
            base_pos_list.append(copy.deepcopy(env.base_pos))
            base_ang_vel_list.append(copy.deepcopy(base_ang_vel))
            feet_pos_list.append(feet_pos)
            base_ori_list.append(copy.deepcopy(base_ori_euler_xyz))
            # contact_state_list.append(optimal_next_state[9:12])
            feet_GRF_list.append(feet_GRF)
            end_effector_position_list.append(end_effector_position_save)
            lift_off_pos_list.append(lift_off_pos)
            base_lin_vel_list.append(copy.deepcopy(base_lin_vel))
            optimal_state_list.append(optimal_next_state[0])
            ref_angular_velocity_list.append(copy.deepcopy(ref_base_ang_vel))
            ref_linear_velocity_list.append( copy.deepcopy(ref_base_lin_vel))
            ref_orientation_list.append(quadrupedpympc_observables["ref_base_angles"])
            ref_position_list.append(np.array([0, sway_ref, cfg.hip_height]))
            joint_pos_arm_list.append(copy.deepcopy(arm_joint_pos))
            joint_vel_arm_list.append(arm_joint_vel)
            wrenche_est_list.append(wrench_estimate)
            # print("wrench estimate", wrench_estimate    )
            zmp_list.append(zmp)
            margin_list.append(zmp_margin)

            contact_state_list.append(contact_state)
            base_acc_list.append(base_acc)
            eef_ref_list.append([np.asarray(eef_task_reference[k])[0]
                                 if eef_task_reference is not None and k in eef_task_reference else np.nan
                                 for k in ('ref_eef_x', 'ref_eef_y', 'ref_eef_z')])
            solver = getattr(quadrupedpympc_wrapper.srbd_controller_interface.controller, 'acados_ocp_solver', None)
            mpc_status_list.append(solver.status if solver is not None else np.nan)
            qpos_list.append(qpos.copy())
            if  update_arm==False and step*simulation_dt > 1.0:
                arm_joint_pos0= np.array([qpos[-3], qpos[-2], qpos[-1]])
                # passive_arm_interface.update_rest_position(arm_joint_pos,0)
                ref_joint_pos_arm = arm_joint_pos
                # env.mjModel.body_mass[-1] += load_carried
                update_arm=True
                print("Arm is now calibrated to position", arm_joint_pos0)
                # breakpoint()

            # ### apply a an impulse on the eef after 5 seconds
            # if update_arm==True and step*simulation_dt > 5.0 and step*simulation_dt < 5.2:
            #     body_force_id = mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_BODY, 'follower/eef')
            #     # env.mjData.xfrc_applied[body_force_id][0:3]=np.array([50,0,0])
    
            #     print("Impulse applied on the arm")

            #     for theta in angles:
            #         # reset simulation or reload initial state before each impulse
            #         # env.set_state(initial_state)
            #         env.mjData.xfrc_applied[body_force_id][0:3] = np.array([F_mag*np.cos(theta), F_mag*np.sin(theta), 0])
            #         for step in range(2):
            #             mujoco.mj_step(env.mjModel, env.mjData)
            #         # final_pos = env.mjData.xpos[body_force_id].copy()
            #         # eef_displacements.append(np.linalg.norm(final_pos - initial_pos))
            

            # while current_impulse < n_impulses:
            #     t = step * simulation_dt

            #     # after 5 s start impulses
            #     if t > 5.0:
            #         theta = angles[current_impulse]
            #         t_in_impulse = t - (5.0 + current_impulse * (impulse_duration + rest_duration))

            #         if 0.0 < t_in_impulse < impulse_duration:
            #             env.mjData.xfrc_applied[body_force_id][0:3] = np.array(
            #                 [F_mag * np.cos(theta), F_mag * np.sin(theta), 0]
            #             )
            #         else:
            #             env.mjData.xfrc_applied[body_force_id][0:3] = np.zeros(3)

            #         # once rest time finished move to next impulse
            #         if t_in_impulse > (impulse_duration + rest_duration):
            #             current_impulse += 1
            # default: no force
            force = np.zeros(3)
            body_force_id = mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_BODY, 'follower/eef')
            t = step * simulation_dt
            # Keep sinusoidal disturbance independent from impulse disturbance.
            sin_force = np.zeros(3)
            impulse_force = np.zeros(3)
            release_force = np.zeros(3)

            if update_arm and apply_sin and t > 1.0:
                # sin_force = np.array([0.0, 0.0, F_mag * np.sin(omega * t)])
                sin_force = np.array([ F_mag * np.sin(omega * t),0.0, 0.0])

            if update_arm and release_from_position and 5.0 <= t < 6.5:
                if not release_force_active or release_force_phase != 1:
                    release_force_active = True
                    release_force_phase = 1
                    print(f"Release force phase 1 applied at t={t:.3f}s: [30.0, 40.0, -20.0]")
                release_force = np.array([15.0, 10.0, -5.0])
            elif update_arm and release_from_position and 10.0 <= t < 12.0:
                if not release_force_active or release_force_phase != 2:
                    release_force_active = True
                    release_force_phase = 2
                    print(f"Release force phase 2 applied at t={t:.3f}s: [30.0, -40.0, -20.0]")
                release_force = np.array([15.0, -10.0, -5.0])
            elif update_arm and release_from_position and 14.0 <= t < 15.5:
                if not release_force_active or release_force_phase != 3:
                    release_force_active = True
                    release_force_phase = 3
                    print(f"Release force phase 3 applied at t={t:.3f}s: [30.0, 40.0, -20.0]")
                release_force = np.array([15.0, 10.0, -5.0])
            elif update_arm and release_from_position and 17.0 <= t < 19.0:
                if not release_force_active or release_force_phase != 4:
                    release_force_active = True
                    release_force_phase = 4
                    print(f"Release force phase 4 applied at t={t:.3f}s: [30.0, -40.0, -20.0]")
                release_force = np.array([15.0, -10.0, -5.0])
            elif release_force_active:
                release_force_active = False
                print(f"Release force removed at t={t:.3f}s")

            if update_arm and release_from_position and t >= 19.5 and not release_sequence_done:
                release_sequence_done = True
                print(f"Release force sequence completed at t={t:.3f}s. Closing simulation and saving PKL.")

            # if update_arm and  apply_impulse and step*simulation_dt > 3.0 and step*simulation_dt < 3.5:

            #     print("Impulse applied on the arm")

            #     force = np.array([80, 0, -10])


            if update_arm and current_impulse < n_impulses and apply_impulse:
                t0 = starts[current_impulse]
                t1 = ends[current_impulse]

                if t0 < t < t1:
                    # entering impulse window
                    if not in_impulse:
                        in_impulse = True
                        eef_start = env.mjData.xpos[body_force_id].copy()


                    if current_impulse ==8 :
                        #apply a pure x force
                        impulse_force = np.array([F_mag, 0, 0])
                    if current_impulse ==9 :
                        #apply a pure y force
                        impulse_force = np.array([0, F_mag, 0])
                        print("no more impulse this episode")
                    elif current_impulse <8:
                        theta = angles[current_impulse]
                        impulse_force = np.array([F_mag*np.cos(theta), F_mag*np.sin(theta), -10])


                elif t >= t1 and in_impulse:
                    # impulse just finished -> record displacement, advance
                    eef_end = env.mjData.xpos[body_force_id].copy()
                    disp_mag[current_impulse] = np.linalg.norm(eef_end - eef_start)
                    in_impulse = False
                    current_impulse += 1
                    print("no more impulse this episode")

            force = sin_force + impulse_force + release_force

            # apply force for this step (or zeros outside windows)
            env.mjData.xfrc_applied[body_force_id][0:3] = force
            impulse_list.append(force.copy())

            # apply force for this step (or zeros outside windows)
            env.mjData.xfrc_applied[body_force_id][0:3] = force
            
            # if step*simulation_dt > 5.0 :
            #     print("5 seconds passed, stopping simulation")
            #     # breakpoint()
            #     env.mjModel.body_mass[-1] = 0.25
            
            # Render only at a certain frequency -----------------------------------------------------------------
            if render and (time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1):
                _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)
                zmp_alpha = 1.0
                add_markers(env.viewer.user_scn, task_markers(), task_marker_ids)
                if render_transition_state is not None:
                    transition_vis = _update_render_transition(
                        env=env,
                        state=render_transition_state,
                        base_pos=base_pos,
                        eef_pos=end_effector_position_save,
                        sim_time=env.mjData.time,
                    )
                    zmp_alpha = transition_vis["zmp_alpha"]

                # print("grf z", feet_GRF.FL[2], feet_GRF.FR[2], feet_GRF.RL[2], feet_GRF.RR[2])
                # print("mass adam",mass_mat_adam[0,0])
                # print("mujoco mass",mujoco_mass_matrix[0,0])
                # counter= counter + 1
                # if counter % 15 == 0:

                # Plot the swing trajectory
                # feet_traj_geom_ids = plot_swing_mujoco(viewer=env.viewer,
                #                                        swing_traj_controller=quadrupedpympc_wrapper.wb_interface.stc,
                #                                        swing_period=quadrupedpympc_wrapper.wb_interface.stc.swing_period,
                #                                        swing_time=LegsAttr(FL=quadrupedpympc_observables["swing_time"][0],
                #                                                            FR=quadrupedpympc_observables["swing_time"][1],
                #                                                            RL=quadrupedpympc_observables["swing_time"][2],
                #                                                            RR=quadrupedpympc_observables["swing_time"][3]),
                #                                        lift_off_positions=quadrupedpympc_observables["lift_off_positions"],
                #                                        nmpc_footholds=quadrupedpympc_observables["nmpc_footholds"],
                #                                        ref_feet_pos=quadrupedpympc_observables["ref_feet_pos"],
                #                                        geom_ids=feet_traj_geom_ids)
                # use mujcoo for inv kin
                ## Compute inv kinematics with mujoco
                

                # breakpoint()

                ##========================== PREDICTION PLOTTING EEF ==================================================================


                transition_cfg = render_transition_state["cfg"] if render_transition_state is not None else {}
                if transition_cfg.get("render_predicted_eef", False):
                    active_indices = [i for i, pos in enumerate(cached_predicted_eef_positions) if pos is not None]
                    last_active_idx = active_indices[-1] if active_indices else -1
                    for i, pred_eef_pos in enumerate(cached_predicted_eef_positions):
                        if pred_eef_pos is None:
                            continue

                        if i == 0:
                            colour = np.array([0.0, 0.85, 0.95, 0.95])
                            diameter = 0.018
                        elif i == last_active_idx:
                            colour = np.array([0.0, 0.0, 0.0, 0.95])
                            diameter = 0.018
                        else:
                            colour = np.array(transition_cfg.get("predicted_eef_color", [0.0, 0.85, 0.95, 0.35]), dtype=float)
                            diameter = float(transition_cfg.get("predicted_eef_diameter", 0.010))

                        eef_pos_id[i] = render_sphere(
                            viewer=env.viewer,
                            position=np.array([pred_eef_pos[0], pred_eef_pos[1], pred_eef_pos[2]]),
                            diameter=diameter,
                            color=colour,
                            geom_id=eef_pos_id[i],
                        )
                    
                # for i in range(horizon):
                #     #draw the trajextory
                #     color=[1, 0, 1, .3]
                #     sin_traj_id[i]= render_sphere(viewer=env.viewer,
                #                     position=([ref_base_pos_traj[i][0],ref_base_pos_traj[i][1],ref_base_pos_traj[i][2]]),
                #                     diameter=d,
                #                     color=colour,
                #                     geom_id=sin_traj_id[i]
                #                     )
                #     # print("optimal next state", optimal_next_state[i][12:15])

                # # # plot adam
                # # # # breakpoint()
                # # lol=np.array(lol).flatten()
                # # # lol2=np.array(lol2).flatten()
                # eef_adam[0]= render_sphere(viewer=env.viewer,
                #                 position=(robot_com),
                #                 diameter=0.05,
                #                 color=[1, 1, 0, 1],
                #                 geom_id=eef_adam[0]
                #                 )

                # eef_adam[1]=render_vector(env.viewer,
                #                                                 vector=lol2,
                #                                                 pos=([lol[0],lol[1],lol[2]]),
                #                                                 scale=np.linalg.norm(lol2) * 0.1,
                #                                                 color=np.array([1, 1, 0, 0.1]),
                #                                                 geom_id=eef_adam[1])
                # Update and Plot the heightmap
                # if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
                #     #if(stc.check_apex_condition(current_contact, interval=0.01)):
                #     for leg_id, leg_name in enumerate(legs_order):
                #         data = heightmaps[leg_name].data#.update_height_map(ref_feet_pos[leg_name], yaw=env.base_ori_euler_xyz[2])
                #         if(data is not None):
                #             for i in range(data.shape[0]):
                #                 for j in range(data.shape[1]):
                #                         heightmaps[leg_name].geom_ids[i, j] = render_sphere(viewer=env.viewer,
                #                                                                             position=([data[i][j][0][0],data[i][j][0][1],data[i][j][0][2]]),
                #                                                                             diameter=0.01,
                #                                                                             color=[0, 1, 0, .5],
                #                                                                             geom_id=heightmaps[leg_name].geom_ids[i, j]
                #                                                                             )
                            
                if transition_cfg.get("render_mujoco_grfs", False):
                    grf_color = np.array(transition_cfg.get("grf_overlay_color", [0.173, 0.627, 0.173, 0.95]), dtype=float)
                    grf_scale = float(transition_cfg.get("grf_overlay_scale", 0.003))
                    for leg_name in legs_order:
                        feet_GRF_geom_ids[leg_name] = render_vector(
                            env.viewer,
                            vector=feet_GRF[leg_name],
                            pos=feet_pos[leg_name],
                            scale=np.linalg.norm(feet_GRF[leg_name]) * grf_scale,
                            color=grf_color,
                            geom_id=feet_GRF_geom_ids[leg_name],
                        )
                #Plot zmp
                for leg_id in legs_order :
                    if contact_state[leg_id]:
                        # # # # # # # print("contact")
                        legs_contact_pos.append(feet_pos[leg_id])


                if cfg.simulation_params['gait'] == 'crawl' or cfg.simulation_params['gait'] == 'full_stance':

                    colour_leg=np.array([0.173, 0.627, 0.173, 0.5 * zmp_alpha])
                    # if cfg.simulation_params['gait'] == 'crawl':
                    #     zmp[2]=legs_contact_pos[1][2]
                    geom_id_zmp=plot_zmp_vis(legs_contact_pos,zmp,env,geom_id_zmp,colour_leg)

                if update_arm and sin_traj_tracking:
                    # Visualize yaw-tracking reference as the EEF path generated by
                    # rotating the current base->eef vector around the world z axis.
                    eef_center = env.mjData.site('eef').xpos.copy()
                    base_to_eef = eef_center - base_pos
                    for i in range(horizon):
                        yaw_delta = ref_base_pos_traj[i][2]
                        c = np.cos(yaw_delta)
                        s = np.sin(yaw_delta)
                        rot_z = np.array([[c, -s, 0.0],
                                          [s,  c, 0.0],
                                          [0.0, 0.0, 1.0]])
                        dot_pos = base_pos + rot_z @ base_to_eef
                        sin_traj_id[i] = render_sphere(
                            viewer=env.viewer,
                            position=np.array([dot_pos[0], dot_pos[1], dot_pos[2]]),
                            diameter=0.008,
                            color=np.array([1, 0, 1, 0.35]),
                            geom_id=sin_traj_id[i]
                        )

                ##render impulse direction
                if in_impulse:
                    impulse_geom_id = render_vector(env.viewer,
                                                vector=impulse_force*0.02,
                                                pos=env.mjData.site('eef').xpos,
                                                scale=np.linalg.norm(impulse_force) * 0.02,
                                                color=np.array([1, 0, 0, 0.8]),
                                                geom_id=impulse_geom_id)
                if update_arm and apply_sin and np.linalg.norm(sin_force) > 1e-9:
                    sin_geom_id = render_vector(env.viewer,
                                                vector=sin_force*0.02,
                                                pos=env.mjData.site('eef').xpos,
                                                scale=np.linalg.norm(sin_force) * 0.02,
                                                color=np.array([0, 1, 1, 0.8]),
                                                geom_id=sin_geom_id)
                if update_arm and release_from_position and np.linalg.norm(release_force) > 1e-9:
                    release_geom_id = render_vector(env.viewer,
                                                vector=release_force*0.02,
                                                pos=env.mjData.site('eef').xpos,
                                                scale=np.linalg.norm(release_force) * 0.02,
                                                color=np.array([1, 0.5, 0, 0.9]),
                                                geom_id=release_geom_id)

                # # PLot initial CoM position
                # cop_id=render_sphere(viewer=env.viewer,
                #                           position=np.array([cop[0],cop[1],zmp[2]]),
                #                           diameter=0.008,
                #                           color=[0, 0.5, 0.5, 0.5], #cyan
                #                           geom_id=cop_id
                #                           )
                # Plot foo contact
                # geom_id_feet_pred=plot_current_foot_prediction(env,geom_id_foot=geom_id_feet_pred,
                #                                                contact_state=contact_state,
                #                                                foot_pos_curr=lift_off_pos)
                
                env.render()
                last_render_time = time.time()
                # Update the sliding windows
                if plotting:
                    plotter.plots["LivePlot"].send_data([base_lin_vel_list[-1][0],base_lin_vel_list[-1][1],base_lin_vel_list[-1][2],
                                                         margin_list[-1],
                                                         base_ori_list[-1][0],base_ori_list[-1][2],
                                                         wrenche_est_list[-1][0],wrenche_est_list[-1][1],wrenche_est_list[-1][2]])


                    # # roll = rpy[0]
                    # # pitch = rpy[1]
                    # # yaw = rpy[2]
                stop_time = time.time()
                wall_clock_time = stop_time - step_start
                total_simulated_time = env.step_num * cfg.simulation_params['dt']
                rtf = total_simulated_time / wall_clock_time
                # print(f"Real Time Facto r: {rtf:.2f}","\n","MPC_frequncy",mpc_frequency)
            # # # # # # Reset the environment if the episode is terminated ------------------------------------------------
            reached_time_limit = max_sim_time is not None and env.mjData.time >= float(max_sim_time)
            if env.step_num >= N_STEPS_PER_EPISODE or reached_time_limit:


                # breakpoint()
                time_stop = time.time()
                # Save data in the pkl -------------------------------------------------------
                data_zmp_study={'time':sim_time_histroy,
                                'base_pos':base_pos_list,
                                'base_lin_vel':base_lin_vel_list,
                                'base_ang_vel':base_ang_vel_list,
                                'base_ori':base_ori_list,
                                'feet_pos':feet_pos_list,
                                'feet_GRF':feet_GRF_list,
                                'end_eff_pos':end_effector_position_list,
                                'lift_off_pos':lift_off_pos_list,
                                'joint_pos_arm':joint_pos_arm_list,
                                'joint_vel_arm':joint_vel_arm_list,
                                'wrench_est':wrenche_est_list,
                                'zmp':zmp_list,
                                'ref_base_lin_vel':ref_linear_velocity_list,
                                'ref_base_ang_vel':ref_angular_velocity_list,
                                'ref_base_ori':ref_orientation_list,
                                'ref_base_pos':ref_position_list,
                                'contact_state':contact_state_list,
                                'base_acc':base_acc_list,
                                'margin_sim':margin_list,
                                'angles_impulse':angles,
                                'impulse':impulse_list,
                                'optimal_state':optimal_state_list,
                                'eef_ref':eef_ref_list,
                                'mpc_status':mpc_status_list,
                                'qpos':qpos_list,
                                }
                

                if cfg.collaborative_task_params.get('output_folder'):
                    output_file = os.path.join(cfg.collaborative_task_params['output_folder'], cfg.pkl_name+'_armpc.pkl')
                    with open(output_file, 'wb') as f:
                        pkl.dump(data_zmp_study, f)
                        print("data saved")
                if export_path is not None:
                    export_npz(export_path, data_zmp_study, run_metadata(cfg, task, quadrupedpympc_wrapper))
                    print("trajectory exported to", export_path)
                env.close()
                return data_zmp_study



                if is_terminated:
                    print("Environment terminated")

                else:
                    state_obs_history.append(ep_state_obs_history)
                    ctrl_state_history.append(ep_ctrl_state_history)


                sim_time_histroy,base_pos_list,feet_pos_list,base_ori_list,contact_state_list,feet_GRF_list,end_effector_position_list,lift_off_pos_list=[],[],[],[],[],[],[],[]
                joint_pos_arm_list,wrenche_est_list = [],[]

                env.reset(random=False)     
                quadrupedpympc_wrapper.reset(initial_feet_pos = env.feet_pos(frame='world')) 

                if(return_dict is not None):
                    return_dict['process'+str(process)+'_ctrl_state_history_ep'+str(episode_num)] = np.array(ctrl_state_history).reshape(-1, len(ctrl_state))
                    if(is_terminated or is_truncated):
                        return_dict['process'+str(process)+'_success_rate_ep'+str(episode_num)] = 0
                    else:
                        return_dict['process'+str(process)+'_success_rate_ep'+str(episode_num)] = 1
                
                break
    
    plotter.stop()

    env.close()
    return return_dict




if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Single-robot MuJoCo simulation. Without --task: the "
                                             "interactive run configured in quadruped_pympc/config.py.")
    ap.add_argument('--task', choices=sorted(TASKS), help="ZiMPedance task preset (simulation/zimpedance_tasks.py)")
    ap.add_argument('--duration', type=float, help="simulated seconds (default: the preset's)")
    ap.add_argument('--scene', help="terrain scene override, e.g. ramp, stairs (robot_model/<robot>/scene_<name>.xml)")
    ap.add_argument('--headless', action='store_true', help="no viewer")
    ap.add_argument('--export', metavar='NPZ', help="write the trajectory log to this .npz")
    args = ap.parse_args()
    if args.task is None:
        run_simulation()
    else:
        task = apply_task(cfg, args.task)
        if args.scene:
            cfg.simulation_params['scene'] = args.scene
        cfg.collaborative_task_params['output_folder'] = None  # no pickle; use --export
        print(f"task {args.task}: {task['description']}")
        run_simulation(num_episodes=1, render=not args.headless, task=dict(task, name=args.task),
                       render_transition_cfg={"max_sim_time": args.duration or task['duration']},
                       export_path=args.export)
    #run_simulation(num_episodes=1, render=False)
