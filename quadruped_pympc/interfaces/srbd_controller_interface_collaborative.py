import numpy as np

from gym_quadruped.utils.quadruped_utils import LegsAttr

from quadruped_pympc import config as cfg

class SRBDControllerInterface:
    """This is an interface for a controller that uses the SRBD method to optimize the gait"""


    def __init__(self, ):
        """ Constructor for the SRBD controller interface """
        
        self.type = cfg.mpc_params['type']
        self.mpc_dt = cfg.mpc_params['dt']
        self.horizon = cfg.mpc_params['horizon']
        self.optimize_step_freq = cfg.mpc_params['optimize_step_freq']
        self.step_freq_available = cfg.mpc_params['step_freq_available']


        self.previous_contact_mpc = np.array([1, 1, 1, 1])
        

        if self.type == 'collaborative':
            from quadruped_pympc.controllers.collaborative.centroidal_nmpc_collaborative import Acados_NMPC_Collaborative
            self.controller = Acados_NMPC_Collaborative()
        else:
            raise ValueError(f"mpc_params['type'] = {self.type!r}: the collaborative wrapper runs the "
                             "'collaborative' controller")


    def compute_control(self, 
                        state_current: dict,
                        ref_state: dict,
                        contact_sequence: np.ndarray,
                        inertia: np.ndarray,
                        pgg_phase_signal: np.ndarray,
                        pgg_step_freq: float,
                        optimize_swing: int,
                        external_wrenches: np.ndarray,
                        Jeef_arm: np.ndarray,
                        end_effector_position:np.ndarray) -> [LegsAttr, LegsAttr, LegsAttr, LegsAttr, LegsAttr, float]:
        """Compute the control using the SRBD method

        Args:
            state_current (dict): The current state of the robot
            ref_state (dict): The reference state of the robot
            contact_sequence (np.ndarray): The contact sequence of the robot
            inertia (np.ndarray): The inertia of the robot
            pgg_phase_signal (np.ndarray): The periodic gait generator phase signal of the legs (from 0 to 1)
            pgg_step_freq (float): The step frequency of the periodic gait generator
            optimize_swing (int): The flag to optimize the swing
            external_wrenches (np.ndarray): The external wrench applied to the robot to compensate

        Returns:
            tuple: The GRFs and the feet positions in world frame, 
                   and the best sample frequency (only if the controller is sampling)
        """
    

        current_contact = np.array([contact_sequence[0][0],
                                    contact_sequence[1][0],
                                    contact_sequence[2][0],
                                    contact_sequence[3][0]])
       
        spring_gain = cfg.collaborative_task_params['spring_gains']
        state_current['passive_arm_force'] = external_wrenches
        nmpc_joints_pos = None
        nmpc_joints_vel = None
        nmpc_joints_acc = None

        nmpc_GRFs, \
        nmpc_footholds, \
        nmpc_predicted_state, \
        status = self.controller.compute_control(state_current,
                                                 ref_state,
                                                 contact_sequence,
                                                 external_wrenches=external_wrenches,
                                                 inertia=inertia,
                                                 end_effector_position=end_effector_position,
                                                 end_effector_jacobian=Jeef_arm,
                                                 end_effector_gain=spring_gain)

        nmpc_footholds = LegsAttr(FL=nmpc_footholds[0],
                                  FR=nmpc_footholds[1],
                                  RL=nmpc_footholds[2],
                                  RR=nmpc_footholds[3])

        best_sample_freq = pgg_step_freq

        nmpc_GRFs = LegsAttr(FL=nmpc_GRFs[0:3] * current_contact[0],
                                FR=nmpc_GRFs[3:6] * current_contact[1],
                                RL=nmpc_GRFs[6:9] * current_contact[2],
                                RR=nmpc_GRFs[9:12] * current_contact[3])
            

        
        return nmpc_GRFs, nmpc_footholds, nmpc_joints_pos, nmpc_joints_vel, nmpc_joints_acc, best_sample_freq, nmpc_predicted_state
        
