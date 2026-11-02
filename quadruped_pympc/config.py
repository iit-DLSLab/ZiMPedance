"""This file includes all of the configuration parameters for the MPC controllers
and of the internal simulations that can be launch from the folder /simulation.
"""
import os

import numpy as np
from quadruped_pympc.helpers.quadruped_utils import GaitType

ROBOT_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gym_quadruped', 'robot_model')

# These are used both for a real experiment and a simulation -----------
# These are the only attributes needed per quadruped, the rest can be computed automatically ----------------------
robot = 'aliengo_leader'  # 'aliengo_leader' (passive arm at the rear) or 'aliengo_follower' (arm at the front)
robot_leg_joints = dict(FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',],  # TODO: Make configs per robot.
                        FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',])
robot_feet_geom_names = dict(FL='FL', FR='FR', RL='RL', RR='RR')
qpos0_js = None  # Zero joint-space configuration. If None it will be extracted from the URDF.
two_robot_mode = True
# ----------------------------------------------------------------------------------------------------------------
def _aliengo_arm_robot(name):
    """Per-robot attributes of the passive-arm Aliengos: 'aliengo_leader' (arm at the rear) and
    'aliengo_follower' (arm at the front). select_robot() at the end of this file switches between them."""
    role = name.split('_')[1]
    feet = {'leader': '1', 'follower': '2'}[role]
    return dict(
        mass={'leader': 24.637, 'follower': 26}[role],
        inertia=np.array([[0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
                          [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
                          [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808]]),
        robot_leg_joints={leg: [f'{role}/{leg}_{j}_joint' for j in ('hip', 'thigh', 'calf')]
                          for leg in ('FL', 'FR', 'RL', 'RR')},
        robot_feet_geom_names={leg: leg + feet for leg in ('FL', 'FR', 'RL', 'RR')},
        arm_joint_names=[f'{role}/arm_link_{i}_joint_pos' for i in (1, 2, 3)]
                        + [{'leader': 'leader/eef_leader', 'follower': 'follower/eef_contact'}[role]],
        legorder=[leg + feet for leg in ('FL', 'FR', 'RL', 'RR')],
        urdf_filename="aliengo_passive_arm.urdf",
        hip_height=0.3,
    )


globals().update(_aliengo_arm_robot(robot))


mpc_params = {
    # 'zimpedance' is the ZiMPedance arm-aware MPC (controllers/zimpedance, single robot)
    # 'collaborative' is the PACC MPC with the passive arm wrench in its model (controllers/collaborative,
    # used by simulation/collaborative_carrying_sim.py, which sets it)
    'type':                                    'zimpedance',
    # horizon is the number of timesteps in the future that the mpc will optimize
    # dt is the discretization time used in the mpc
    #dt*horizon is the time in the future that the mpc will optimize
    'horizon':                                 25,
    'dt':                                      0.02,

    # GRF limits for each single leg
    "grf_max":                                 mass * 9.81,
    "grf_min":                                 0,
    'mu':                                      0.5,

    # this is used to have a smaller dt near the start of the horizon
    'use_nonuniform_discretization':           False,
    'horizon_fine_grained':                    2,
    'dt_fine_grained':                         0.01,

    # if this is true, we optimize the step frequency as well
    # for the sampling controller, this is done in the rollout
    # for the gradient-based controller, this is done with a batched version of the ocp
    'optimize_step_freq':                      False,
    'step_freq_available':                     [1.4, 2.0, 2.4],  # not supported in this repository

    # ----- START properties only for the gradient-based mpc -----

    # this is used if you want to manually warm start the mpc
    'use_warm_start':                          False,

    # this enables integrators for height, linear velocities, roll and pitch
    'use_integrators':                         False,
    'alpha_integrator':                        0.1,
    'integrator_cap':                          [0.5, 0.2, 0.2, 0.0, 0.0, 1.0],

    # if this is off, the mpc will not optimize the footholds and will
    # use only the ones provided in the reference
    'use_foothold_optimization':               True,

    # this is set to false automatically is use_foothold_optimization is false
    # because in that case we cannot chose the footholds and foothold
    # constraints do not any make sense
    'use_foothold_constraints':                False,

    # works with all the mpc types except 'sampling'. In sim does not do much for now,
    # but in real it minizimes the delay between the mpc control and the state
    'use_RTI':                                 False,
    # If RTI is used, we can set the advance RTI-step! (Standard is the simpler RTI)
    # See https://arxiv.org/pdf/2403.07101.pdf
    'as_rti_type':                             "AS-RTI-A",  # "AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D", "Standard"
    'as_rti_iter':                             1,  # > 0, the higher the better, but slower computation!

    # This will force to use DDP instead of SQP, based on https://arxiv.org/abs/2403.10115.
    # Note that RTI is not compatible with DDP, and no state costraints for now are considered
    'use_DDP':                                 False,

    # this is used only in the case 'use_RTI' is false in a single mpc feedback loop.
    # More is better, but slower computation!
    'num_qp_iterations':                       1,

    # this is used to speeding up or robustify acados' solver (hpipm).
    'solver_mode':                             'balance',  # balance, robust, speed, crazy_speed


    # ONLY ONE CAN BE TRUE AT A TIME (only gradient)
    'use_static_stability':                    False,
    'use_zmp_stability':                       True,
    'trot_stability_margin':                   0.0,   # for diagonal support, track the line instead of a tube
    'pace_stability_margin':                   0.03,
    'crawl_stability_margin':                  0.04,  # in general, 0.02 is a good value

    # this is used to compensate for the external wrenches
    # you should provide explicitly this value in compute_control
    'external_wrenches_compensation':          True,
    'external_wrenches_compensation_num_step': 25,

    # this is used only in the case of collaborative mpc, to
    # compensate for the external wrench in the prediction (only collaborative)
    'passive_arm_compensation':                True,

    # Only for 'zimpedance' (controllers/zimpedance): cost weights are
    # taken from armpc_weight_profiles below, selected by name.
    'armpc_weight_profile':                    'paper_table3',
    # Pinocchio MJCF used inside the 'zimpedance' prediction model. The default is the
    # simulated robot's own MJCF (every scene_*.xml of `robot` includes it; aliengo_leader:
    # arm mounted at the rear). None uses the controller's new_urdf_export.xml, the hardware
    # robot with the arm at the front. The acados solver rebuilds when this file changes.
    'armpc_model_xml':                         os.path.join(ROBOT_MODELS, robot, 'new_urdf_export.xml'),
    # Source of the 'zimpedance' passive-arm parameters. 'model': joint stiffness, damping and
    # springref of armpc_model_xml, reference GRF from its total mass (sim: matches the plant).
    # 'hw_as_run': what the hardware ran, i.e. k and d from collaborative_task_params, spring
    # rest = the arm pose calibrated after 1 s, reference mass 25.523 kg.
    'armpc_arm_params':                        'model',


    # ----- END properties for the gradient-based mpc -----

    }

# Cost-weight profiles for 'zimpedance'. Vectors are (x, y, z) or
# (roll, pitch, yaw); arm vectors are per passive joint; R_foot_force is per foot.
_armpc_hw_as_run = {
    # The weights the hardware experiments ran with (paper Tables IV and V).
    'Q_position':          [0, 0, 1500],
    'Q_velocity':          [200, 200, 200],
    'Q_base_angle':        [200, 300, 0],
    'Q_base_angle_rates':  [20, 20, 50],
    'Q_foot_pos':          [300, 300, 300],
    'Q_q_arm':             [0.5, 0.5, 0.5],
    'Q_q_dot_arm':         [0.5, 0.5, 0.5],
    'Q_eef':               [5, 20000, 5],
    'R_foot_vel':          [0.0001, 0.0001, 0.00001],
    'R_foot_force':        [0.008, 0.008, 0.015],
}
armpc_weight_profiles = {
    # Paper Table III (differs from hw_as_run in Q_base_angle, Q_q_arm, Q_q_dot_arm, R_foot_force).
    'paper_table3':       {**_armpc_hw_as_run,
                           'Q_base_angle': [200, 200, 0],
                           'Q_q_arm':      [5, 5, 5],
                           'Q_q_dot_arm':  [10, 10, 10],
                           'R_foot_force': [0.001, 0.001, 0.001]},
    'hw_as_run':          _armpc_hw_as_run,
    # The weights of the hardware y tracking (paper Fig. 6).
    'hw_eef_y_tracking':  {**_armpc_hw_as_run,
                           'Q_position':         [50, 1200, 1000],
                           'Q_velocity':         [20, 200, 80],
                           'Q_base_angle':       [80, 150, 80],
                           'Q_base_angle_rates': [15, 100, 50],
                           'Q_q_arm':            [1, 1, 1],
                           'Q_q_dot_arm':        [15, 15, 15]},
    # Paper Sec. V-D: Table III with Q_position, Q_velocity, Q_base_angle replaced.
    'paper_eef_tracking': {**_armpc_hw_as_run,
                           'Q_position':   [50, 800, 1000],
                           'Q_velocity':   [20, 80, 80],
                           'Q_base_angle': [60, 140, 50],
                           'Q_q_arm':      [5, 5, 5],
                           'Q_q_dot_arm':  [10, 10, 10],
                           'R_foot_force': [0.001, 0.001, 0.001]},
    # Simulation y-z EEF tasks (circle of paper Fig. 6, right), EEF residual on (y, z) only.
    'sim_eef_yz_tracking': {**_armpc_hw_as_run,
                            'Q_position':         [0, 0, 1500],
                            'Q_velocity':         [50, 50, 50],
                            'Q_base_angle':       [20, 20, 100],
                            'Q_base_angle_rates': [20, 20, 50],
                            'Q_q_arm':            [5, 5, 5],
                            'Q_q_dot_arm':        [10, 10, 10],
                            'Q_eef':              [0, 1500, 1500],
                            'R_foot_force':       [0.001, 0.001, 0.001]},
    # Simulation figure eight: the base is held (Q_position y, roll and yaw raised) and the arm
    # joints are free (Q_q_arm = Q_q_dot_arm = 0), so the MPC excites the soft arm instead of
    # carrying the EEF with the base: 0.24 m of EEF sway for 13 mm of base sway.
    'sim_eef_eight_tracking': {**_armpc_hw_as_run,
                               'Q_position':         [0, 600, 1500],
                               'Q_velocity':         [50, 50, 50],
                               'Q_base_angle':       [600, 20, 300],
                               'Q_base_angle_rates': [20, 20, 50],
                               'Q_q_arm':            [0, 0, 0],
                               'Q_q_dot_arm':        [0, 0, 0],
                               'Q_eef':              [0, 1500, 1500],
                               'R_foot_force':       [0.001, 0.001, 0.001]},
    # Simulation x-y reaching while walking: sim_eef_yz_tracking with the EEF cost on (x, y); the
    # task commands the base velocity, so the base position is free in x and y.
    'sim_eef_xy_reaching': {**_armpc_hw_as_run,
                            'Q_position':         [0, 0, 1500],
                            'Q_velocity':         [50, 50, 50],
                            'Q_base_angle':       [20, 20, 100],
                            'Q_base_angle_rates': [20, 20, 50],
                            'Q_q_arm':            [5, 5, 5],
                            'Q_q_dot_arm':        [10, 10, 10],
                            'Q_eef':              [1500, 1500, 0],
                            'R_foot_force':       [0.001, 0.001, 0.001]},
}
# -----------------------------------------------------------------------

# Automatic simulation dt selection based on the chosen controller type.
# Update this map to keep controller-specific integration settings in one place.
sim_dt_by_controller = {
    'zimpedance': 0.002,
    'collaborative': 0.002,
}
default_sim_dt = 0.006
auto_sim_dt = sim_dt_by_controller.get(mpc_params['type'], default_sim_dt)

simulation_params = {
    'swing_generator':             'scipy',  # 'scipy', 'explicit', 'ndcurves'
    'swing_position_gain_fb':      500,
    'swing_velocity_gain_fb':      10,
    'swing_integral_gain_fb':      0,
    'step_height':                 0.3 * hip_height,  # 0.05 go2

    # Visual Foothold adapatation
    "visual_foothold_adaptation":  'blind',  # only 'blind' here (no heightmap sensor)

    # this is the integration time used in the simulator
    'dt':                          auto_sim_dt,  # auto-selected from mpc_params['type']


    'gait':                        'crawl',  # 'trot', 'pace', 'crawl', 'bound', 'full_stance'
    'gait_params':                 {'trot': {'step_freq': 1.4, 'duty_factor': 0.65, 'type': GaitType.TROT.value},
                                    # 'crawl': {'step_freq': 0.5, 'duty_factor': 0.7, 'type': GaitType.CIRCULARCRAWL.value},#dynamic crawl!! with step freq 0.9
                                    'crawl': {'step_freq': 0.5, 'duty_factor': 0.8, 'type': GaitType.BACKDIAGONALCRAWL.value},
                                    #'crawl': {'step_freq': 0.5, 'duty_factor': 0.8, 'type': GaitType.CIRCULARCRAWL.value},
                                    'front_diagcrawl': {'step_freq': 0.5, 'duty_factor': 0.8, 'type': GaitType.FRONTDIAGONALCRAWL.value},
                                    # 'crawl': {'step_freq': 0.5, 'duty_factor': 0.8, 'type': GaitType.BFDIAGONALCRAWL.value},
                                    
                                    'pace': {'step_freq': 2, 'duty_factor': 0.7, 'type': GaitType.PACE.value},
                                    'bound': {'step_freq': 4, 'duty_factor': 0.65, 'type': GaitType.BOUNDING.value},
                                    'full_stance': {'step_freq': 2, 'duty_factor': 0.65, 'type': GaitType.FULL_STANCE.value},
                                   },

    
                                    # 'crawl': {'step_freq': 0.6, 'duty_factor': 0.76, 'type': GaitType.BACKDIAGONALCRAWL.value},
                                    #'crawl': {'step_freq': 0.9, 'duty_factor': 0.76, 'type': GaitType.BACKDIAGONALCRAWL.value},


    # velocity mode: human will give you the possibility to use the keyboard, the other ares
    # forward only random linear-velocity, random will give you random linear-velocity and yaw-velocity
    'mode':                        'human',  # 'human', 'forward', 'random'
    'ref_z':                       hip_height,


    # the MPC will be called every 1/(mpc_frequency*dt) timesteps
    # this helps to evaluate more realistically the performance of the controller
    'mpc_frequency':               100,

    'use_inertia_recomputation':   False,

    # Add the legs' bias torques (gravity, Coriolis) to the stance torque -J^T f, so that the
    # realized GRFs match the MPC's. False keeps the upstream stance law.
    'stance_leg_bias_compensation': False,

    # Place the reference footholds around the whole-body CoM instead of the hips (the hip
    # positions are shifted by the horizontal CoM-to-base offset). With a passive arm the CoM sits
    # 6-8 cm off the trunk centre, and hip-centred footholds then bias the crawl speed by about
    # offset / (stance time / 2): ~0.1 m/s backward with the front arm (aliengo_follower), forward
    # with the rear arm (aliengo_leader). False keeps the upstream hip-centred footholds.
    'com_centred_footholds':       False,

    # Integral trim of the linear velocity command [1/s] (simulation/zimpedance_sim.py): adds
    # gain * integral(commanded - measured heading-frame velocity) to the command, bounded at
    # 0.15 m/s, reset whenever the command is zero. It removes the steady speed error of the crawl
    # (the ZMP margin slows it forward and speeds it backward by ~0.03 m/s). 0 disables it.
    'velocity_trim_gain':          0.0,

    'scene':                       'flat',  # scene_<name>.xml of the robot: flat, rough, stairs, ramp; leader also slope, pyramid, suspend_stairs

    'frame_external_force':         'world', # 'leader/trunk', leader/arm_link_2 'follower/eef_follower' 'load_center
    'plotting':                     False,  # if true, the simulation will plot the results
    }

collaborative_task_params = {

        # ARM PARAMETERS

        'mass_matrix':              np.array([0.554,
                                              0.554,
                                              0.254 ]),
        'damping_gains':            np.array([0.5,
                                              0.5   ,
                                              0.5 ]),
        'spring_gains' :            np.array([5,
                                              5,
                                              5 ]),
        'eef_rest_height':          0.4799749300066574,

        'load_carried':             0,

        'two_robots_mode':          True,
        'output_folder':            None,  # a directory: simulation/zimpedance_sim.py pickles its log there

        'amplitude' :10, # Amplitude
        'frequency' :0.45,  # Frequency in Hz
        'sin_axis':                np.array([1.0, 0.2, 0.0]),
        'zmp_marker_diameter':     0.014,
        'zmp_support_line_width':  0.0035,
        'zmp_visual_x_scale':      1.0,
        'zmp_visual_x_offset':     0.0,
        'apply_sin': False,
        'apply_impulse': False,
        'release_from_position': False,
        'sin_traj_tracking': False,


}

gait=simulation_params['gait']
pkl_name='zimpedance_run'
# -----------------------------------------------------------------------


def select_robot(name):
    """Switch the simulated robot after import between 'aliengo_leader' and 'aliengo_follower'
    (the per-robot attributes above are set once, from `robot`, when this module is imported)."""
    global robot
    robot = name
    globals().update(_aliengo_arm_robot(name))
    mpc_params['armpc_model_xml'] = os.path.join(ROBOT_MODELS, name, 'new_urdf_export.xml')
    simulation_params['step_height'] = 0.3 * hip_height
    simulation_params['ref_z'] = hip_height
