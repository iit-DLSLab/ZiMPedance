import os
import numpy as np

class SwingTrajectoryController:
    def __init__(self,
                 step_height: float,
                 swing_period: float,
                 position_gain_fb: np.ndarray,
                 velocity_gain_fb: np.ndarray,
                 generator: str) -> None:

        self.generator = generator

        if (self.generator == "ndcurves"):
            from .swing_generators.ndcurves_swing_trajectory_generator import SwingTrajectoryGenerator
            self.swing_generator = SwingTrajectoryGenerator(swing_period=swing_period, step_height=step_height)
        elif (self.generator == "scipy"):
            from .swing_generators.scipy_swing_trajectory_generator import SwingTrajectoryGenerator
            self.swing_generator = SwingTrajectoryGenerator(swing_period=swing_period, step_height=step_height)
        else:
            from .swing_generators.explicit_swing_trajectory_generator import SwingTrajectoryGenerator
            self.swing_generator = SwingTrajectoryGenerator(swing_period=swing_period, step_height=step_height)

        self.position_gain_fb = position_gain_fb
        self.velocity_gain_fb = velocity_gain_fb
        self.swing_period = swing_period
        self.swing_time = [0, 0, 0, 0]

        self.use_feedback_linearization = True
        self.use_friction_compensation = True

    def regenerate_swing_trajectory_generator(self, step_height: float, swing_period: float) -> None:
        if (self.generator == "ndcurves"):
            from .swing_generators.ndcurves_swing_trajectory_generator import SwingTrajectoryGenerator
            self.swing_generator = SwingTrajectoryGenerator(swing_period=swing_period, step_height=step_height)
        elif (self.generator == "scipy"):
            from .swing_generators.scipy_swing_trajectory_generator import SwingTrajectoryGenerator
            self.swing_generator = SwingTrajectoryGenerator(swing_period=swing_period, step_height=step_height)
        else:
            from .swing_generators.explicit_swing_trajectory_generator import SwingTrajectoryGenerator
            self.swing_generator = SwingTrajectoryGenerator(swing_period=swing_period, step_height=step_height)
        self.swing_period = swing_period

    def compute_swing_control_cartesian_space(self,
                              leg_id,
                              q_dot,
                              J,
                              J_dot,
                              lift_off,
                              touch_down,
                              foot_pos,
                              foot_vel,
                              passive_force,
                              h,
                              mass_matrix):
        """TODO: Docstring.

        Args:
        ----
            model:
            q:
            q_dot:
            J:
            J_dot:
            lift_off:
            touch_down:
            swing_time:
            foot_pos:
            foot_vel:
            h:
            mass_matrix:

        Returns:
        -------

        """
        # Compute trajectory references
        des_foot_pos, des_foot_vel, des_foot_acc = self.swing_generator.compute_trajectory_references(
            self.swing_time[leg_id],
            lift_off,
            touch_down
            )

        err_pos = des_foot_pos - foot_pos
        err_pos = err_pos.reshape((3,))

        err_vel = des_foot_vel - foot_vel
        err_vel = err_vel.reshape((3,))

        accelleration = des_foot_acc + \
                        self.position_gain_fb * (err_pos) + \
                        self.velocity_gain_fb * (err_vel)

        accelleration = accelleration.reshape((3,))

        # Compute inertia matrix in task space.
        # Mass Matrix and centrifugal missing
        tau_swing = J.T @ (self.position_gain_fb * (err_pos) + self.velocity_gain_fb * (err_vel))
        if(self.use_feedback_linearization):
            tau_swing += mass_matrix @ np.linalg.pinv(J) @ (accelleration - J_dot @ q_dot) + h

        if(self.use_friction_compensation):
            tau_swing -= passive_force

        return tau_swing, des_foot_pos, des_foot_vel
    


    def compute_swing_control_joint_space(self, 
                                          nmpc_joints_pos, 
                                          nmpc_joints_vel, 
                                          nmpc_joints_acc, 
                                          qpos, qvel,
                                          legs_mass_matrix,
                                          legs_qfrc_bias):
        error_position = nmpc_joints_pos - qpos
        error_position = error_position.reshape((3, ))

        error_velocity = nmpc_joints_vel - qvel
        error_velocity = error_velocity.reshape((3, ))

        accelleration = nmpc_joints_acc
        accelleration = accelleration.reshape((3,))
        
        tau_swing = self.position_gain_fb*error_position + self.velocity_gain_fb*error_velocity
        # Feedback linearization
        if(self.use_feedback_linearization):
            tau_swing += legs_mass_matrix@(accelleration + self.position_gain_fb*error_position + self.velocity_gain_fb*error_velocity) + legs_qfrc_bias
        
        return tau_swing, None, None



    def update_swing_time(self, current_contact, legs_order, dt):
        for leg_id, leg_name in enumerate(legs_order):
            # Swing time reset
            if current_contact[leg_id] == 0:
                if self.swing_time[leg_id] < self.swing_period:
                    self.swing_time[leg_id] = self.swing_time[leg_id] + dt
            else:
                self.swing_time[leg_id] = 0



    def check_apex_condition(self, current_contact, interval=0.02):
        optimize_swing = 0
        for leg_id in range(4):
            # Swing time check
            if (current_contact[leg_id] == 0):
                if ((self.swing_time[leg_id] > (self.swing_period / 2.) - interval) and \
                        (self.swing_time[leg_id] < (self.swing_period / 2.) + interval)):
                    optimize_swing = 1
        return optimize_swing



    def check_full_stance_condition(self, current_contact):
        stance = 1
        # If one leg is not in stance, the robot is not in full stance
        for leg_id in range(4):
            if (current_contact[leg_id] == 0):
                stance = 0
        return stance
