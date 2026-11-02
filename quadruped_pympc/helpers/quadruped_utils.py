from __future__ import annotations

import copy
from collections import namedtuple
from enum import Enum

import numpy as np
from mujoco.viewer import Handle

from quadruped_pympc.helpers.swing_trajectory_controller import SwingTrajectoryController


from scipy.spatial.transform import Rotation as R



class GaitType(Enum):
    """Enumeration class to represent the different gaits that a quadruped robot can perform."""

    TROT = 0
    PACE = 1
    BOUNDING = 2
    CIRCULARCRAWL = 3
    BFDIAGONALCRAWL = 4
    BACKDIAGONALCRAWL = 5
    FRONTDIAGONALCRAWL = 6
    FULL_STANCE = 7


def plot_swing_mujoco(viewer: Handle,
                      swing_traj_controller: SwingTrajectoryController,
                      swing_period,
                      swing_time: namedtuple,
                      lift_off_positions: namedtuple,
                      nmpc_footholds: namedtuple,
                      ref_feet_pos: namedtuple,
                      geom_ids: namedtuple = None,
                      ):
    """Function to plot the desired foot swing trajectory in Mujoco.

    Args:
    ----
    viewer (Handle): The Mujoco viewer.
    swing_traj_controller (SwingTrajectoryController): The swing trajectory controller.
    swing_period: The swing period.
    swing_time (LegsAttr): The swing time for each leg.
    lift_off_positions (LegsAttr): The lift-off positions for each leg.
    nmpc_footholds (LegsAttr): The footholds for each leg.
    ref_feet_pos (LegsAttr): The reference feet positions for each leg.
    geom_ids (LegsAttr, optional): The geometry ids for each leg. Defaults to None.

    Returns:
    -------
    LegsAttr: The geometry ids for each leg trajectory
    """
    from gym_quadruped.utils.mujoco.visual import render_line, render_sphere
    from gym_quadruped.utils.quadruped_utils import LegsAttr

    # Plot desired foot swing trajectory in Mujoco.
    NUM_TRAJ_POINTS = 6

    if geom_ids is None:
        geom_ids = LegsAttr(FL=[], FR=[], RL=[], RR=[])
        # Instantiate a new geometry
        for leg_id, leg_name in enumerate(['FL', 'FR', 'RL', 'RR']):
            viewer.user_scn.ngeom += NUM_TRAJ_POINTS
            geom_ids[leg_name] = list(range(viewer.user_scn.ngeom - NUM_TRAJ_POINTS - 1, viewer.user_scn.ngeom - 1))

    # viewer.user_scn.ngeom = 1
    # We first draw the trajectory of the feet
    des_foot_traj = LegsAttr(FL=[], FR=[], RL=[], RR=[])
    for leg_id, leg_name in enumerate(['FL', 'FR', 'RL', 'RR']):
        # TODO: This function should be vectorized rather than queried sequentially
        if swing_time[leg_name] == 0.0:
            continue
        for point_idx, foot_swing_time in enumerate(np.linspace(swing_time[leg_name], swing_period, NUM_TRAJ_POINTS)):
            ref_foot_pos, _, _ = swing_traj_controller.swing_generator.compute_trajectory_references(
                foot_swing_time,
                lift_off_positions[leg_name],
                nmpc_footholds[leg_name])
            des_foot_traj[leg_name].append(ref_foot_pos.squeeze())

        for point_idx in range(NUM_TRAJ_POINTS - 1):
            render_line(viewer=viewer,
                        initial_point=des_foot_traj[leg_name][point_idx],
                        target_point=des_foot_traj[leg_name][point_idx + 1],
                        width=.005,
                        color=np.array([1, 0, 0, 1]),
                        geom_id=geom_ids[leg_name][point_idx]
                        )

        # Add a sphere at the the ref_feet_pos
        render_sphere(viewer=viewer,
                      position=ref_feet_pos[leg_name],
                      diameter=0.04,
                      color=np.array([0, 1, 0, .5]),
                      geom_id=geom_ids[leg_name][-1]
                      )
    return geom_ids
def plot_swing_mujoco_two_robots(viewer: Handle,
                      swing_traj_controller: SwingTrajectoryController,
                      swing_period,
                      swing_time: namedtuple,
                      lift_off_positions: namedtuple,
                      nmpc_footholds: namedtuple,
                      ref_feet_pos: namedtuple,
                      geom_ids: namedtuple = None,
                      ):
    """Function to plot the desired foot swing trajectory in Mujoco.

    Args:
    ----
    viewer (Handle): The Mujoco viewer.
    swing_traj_controller (SwingTrajectoryController): The swing trajectory controller.
    swing_period: The swing period.
    swing_time (LegsAttr): The swing time for each leg.
    lift_off_positions (LegsAttr): The lift-off positions for each leg.
    nmpc_footholds (LegsAttr): The footholds for each leg.
    ref_feet_pos (LegsAttr): The reference feet positions for each leg.
    geom_ids (LegsAttr, optional): The geometry ids for each leg. Defaults to None.

    Returns:
    -------
    LegsAttr: The geometry ids for each leg trajectory
    """
    from gym_quadruped.utils.mujoco.visual import render_line, render_sphere
    from gym_quadruped.utils.quadruped_utils import LegsAttr
    from gym_quadruped.utils.quadruped_utils_collab import LegsAttrTwoRbots


    # Plot desired foot swing trajectory in Mujoco.
    NUM_TRAJ_POINTS = 4

    if geom_ids is None:
        geom_ids = LegsAttrTwoRbots(FL1=[], FR1=[], RL1=[], RR1=[], FL2=[], FR2=[], RL2=[], RR2=[])
        # Instantiate a new geometry
        for leg_id, leg_name in enumerate(['FL1', 'FR1', 'RL1', 'RR1', 'FL2', 'FR2', 'RL2', 'RR2']):
            viewer.user_scn.ngeom += NUM_TRAJ_POINTS
            geom_ids[leg_name] = list(range(viewer.user_scn.ngeom - NUM_TRAJ_POINTS - 1, viewer.user_scn.ngeom - 1))

    # viewer.user_scn.ngeom = 1
    # We first draw the trajectory of the feet
    des_foot_traj = LegsAttrTwoRbots(FL1=[], FR1=[], RL1=[], RR1=[], FL2=[], FR2=[], RL2=[], RR2=[])
    for leg_id, leg_name in enumerate(['FL1', 'FR1', 'RL1', 'RR1', 'FL2', 'FR2', 'RL2', 'RR2']):
        # TODO: This function should be vectorized rather than queried sequentially
        if swing_time[leg_name] == 0.0:
            continue
        for point_idx, foot_swing_time in enumerate(np.linspace(swing_time[leg_name], swing_period, NUM_TRAJ_POINTS)):
            ref_foot_pos, _, _ = swing_traj_controller.swing_generator.compute_trajectory_references(
                foot_swing_time,
                lift_off_positions[leg_name],
                nmpc_footholds[leg_name])
            des_foot_traj[leg_name].append(ref_foot_pos.squeeze())

        for point_idx in range(NUM_TRAJ_POINTS - 1):
            render_line(viewer=viewer,
                        initial_point=des_foot_traj[leg_name][point_idx],
                        target_point=des_foot_traj[leg_name][point_idx + 1],
                        width=.005,
                        color=np.array([1, 0, 0, 1]),
                        geom_id=geom_ids[leg_name][point_idx]
                        )

        # Add a sphere at the the ref_feet_pos
        render_sphere(viewer=viewer,
                      position=ref_feet_pos[leg_name],
                      diameter=0.04,
                      color=np.array([0, 1, 0, .5]),
                      geom_id=geom_ids[leg_name][-1]
                      )
    return geom_ids



