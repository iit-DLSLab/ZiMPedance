
from gym_quadruped.utils.mujoco.visual import render_line,render_sphere
import numpy as np
import copy
import matplotlib.pyplot as plt
from quadruped_pympc import config

from quadruped_pympc.helpers.terrain_estimator import TerrainEstimator

from scipy.spatial.transform import Rotation as R
import math

def zmp_color(contact_state,legs_order):

    key1=str(legs_order[0])
    key2=str(legs_order[1])
    key3=str(legs_order[2])
    key4=str(legs_order[3])

    # blue works this is FL FR RL
    if contact_state[key1] and contact_state[key4] and contact_state[key2]:
        if contact_state[key3]:
            colour_leg=[1,1,1,0.4]
            # continue
        else:

            colour_leg= [0,0,1, 1]

    #cyan works this is FR RR RL
    elif contact_state[key2] and contact_state[key4] and contact_state[key3]:
        if contact_state[key1]:
            colour_leg=[1,1,0,0.4]

            # continue
        else:
            colour_leg= [1, 0, 1, 0.4]

    #triangle 3 magenta
    elif contact_state[key2] and contact_state[key1] and contact_state[key3]:
        if contact_state[key4]:
            colour_leg=[1,1,1,0.4]
            breakpoint()

        else:

            colour_leg= [0, 1,1, 0.4]
    #triangle 4 giallo 
    elif contact_state[key4] and contact_state[key1] and contact_state[key3]:
        if contact_state[key2]:
            colour_leg=[1,1,1,0.4]
        else:
            colour_leg= [1, 1, 0, 0.4]

    else:
        colour_leg=[1,1,1,1]

    return colour_leg
def plot_zmp_vis(legs_contact_pos,zmpl_rot,env,geom_id_zmp,colour_leg):
    width = config.collaborative_task_params.get('zmp_support_line_width', 0.002)
    d = config.collaborative_task_params.get('zmp_marker_diameter', 0.008)
    if len(legs_contact_pos) == 3:
        
        geom_id_zmp[0] = render_line(env.viewer,
                                                    target_point=legs_contact_pos[0],
                                                    initial_point=legs_contact_pos[1],
                                                    width=width,
                                                    color=colour_leg,
                                                    geom_id=geom_id_zmp[0])
        geom_id_zmp[1] = render_line(env.viewer,
                                    target_point=legs_contact_pos[1],
                                    initial_point=legs_contact_pos[2],
                                    width=width,
                                    color=colour_leg,
                                    geom_id=geom_id_zmp[1])
        geom_id_zmp[2] = render_line(env.viewer,
                                    target_point=legs_contact_pos[2],
                                    initial_point=legs_contact_pos[0],
                                    width=width,
                                    color=colour_leg,
                                    geom_id=geom_id_zmp[2])
        geom_id_zmp[3] = render_line(env.viewer,
                                    target_point=np.array([0,0,0]),
                                    initial_point=np.array([1,0,0]),
                                    width=width,
                                    color=[0, 0, 0, 0],
                                    geom_id=geom_id_zmp[3])
        
        geom_id_zmp[4]=render_sphere(viewer=env.viewer,
                                position=zmpl_rot,
                                diameter=d,
                                color=colour_leg,
                                geom_id=geom_id_zmp[4]
                                )
        
    
    elif len(legs_contact_pos) == 4:
        geom_id_zmp[0] = render_line(env.viewer,
                                                    target_point=legs_contact_pos[0],
                                                    initial_point=legs_contact_pos[1],
                                                    width=width,
                                                    color=colour_leg,
                                                    geom_id=geom_id_zmp[0])
        geom_id_zmp[1] = render_line(env.viewer,
                                    target_point=legs_contact_pos[0],
                                    initial_point=legs_contact_pos[2],
                                    width=width,
                                    color=colour_leg,
                                    geom_id=geom_id_zmp[1])
        geom_id_zmp[2] = render_line(env.viewer,
                                    target_point=legs_contact_pos[2],
                                    initial_point=legs_contact_pos[3],
                                    width=width,
                                    color=colour_leg,
                                    geom_id=geom_id_zmp[2])
        geom_id_zmp[3] = render_line(env.viewer,
                                    target_point=legs_contact_pos[3],
                                    initial_point=legs_contact_pos[1],
                                    width=width,
                                    color=colour_leg,
                                    geom_id=geom_id_zmp[3])
        
        geom_id_zmp[4]=render_sphere(viewer=env.viewer,
                                position=zmpl_rot,
                                diameter=d,
                                color=colour_leg,
                                geom_id=geom_id_zmp[4]
                                )
        # leg_pos_full_stance = np.array([legs_contact_pos[0],
        # legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]])
        # vector_four_points.append(leg_pos_full_stance)
    elif len(legs_contact_pos) == 2:
        geom_id_zmp[0] = render_line(env.viewer,
                                                    target_point=legs_contact_pos[0],
                                                    initial_point=legs_contact_pos[1],
                                                    width=width,
                                                    color=colour_leg,
                                                    geom_id=geom_id_zmp[0])
        geom_id_zmp[1] = render_line(env.viewer,
                                    target_point=legs_contact_pos[1],
                                    initial_point=legs_contact_pos[0],
                                    width=width,
                                    color=colour_leg,
                                    geom_id=geom_id_zmp[1])
        geom_id_zmp[2] = render_line(env.viewer,
                                    target_point=np.array([0,0,0]),
                                    initial_point=np.array([1,0,0]),
                                    width=width,
                                    color=[0, 0, 0, 0],
                                    geom_id=geom_id_zmp[2])
        geom_id_zmp[3] = render_line(env.viewer, 
                                    target_point=np.array([0,0,0]),
                                    initial_point=np.array([1,0,0]),
                                    width=width,
                                    color=[0, 0, 0, 0],
                                    geom_id=geom_id_zmp[3])
        geom_id_zmp[4]=render_sphere(viewer=env.viewer,
                                position=zmpl_rot,
                                diameter=d,
                                color=colour_leg,
                                geom_id=geom_id_zmp[4]
                                )
        
    else:
        geom_id_zmp[4]=render_sphere(viewer=env.viewer,
                                position=zmpl_rot,
                                diameter=d,
                                color=colour_leg,
                                geom_id=geom_id_zmp[4]
                                )
    return geom_id_zmp



def compute_zmp(base_position,linear_acc,base_or,ext_wrenches,Pee):
    # TODO: This import should go
    base_w = base_position

    gravity = np.array([0, 0, -9.81])
    linear_com_acc = np.asarray(linear_acc)


    if (config.mpc_params['use_zmp_stability']):
        # Keep the simulation/logged ZMP in world XY because the current
        # stability-margin utilities compare it directly against world-frame
        # foothold positions.
        robot_height = base_w[2]
        zmp_xy = base_w[0:2] - linear_com_acc[0:2] * (robot_height / (-gravity[2]))
        zmp = np.array([zmp_xy[0], zmp_xy[1], 0.0])


    else:
        x = base_position[0]
        y = base_position[1]
        zmp = np.array([x,y,0])

    return zmp
def compute_zmp_mpc(base_position,linear_com_acc,external_forces_linear,end_effector_position,base_ori):
        base_w = base_position
        gravity = np.array([0, 0, -9.81])
        mass=24
        zmp_x = mass*-gravity[2]*base_w[0] - base_w[2]*mass*linear_com_acc[0]
        zmp_x = zmp_x + end_effector_position[0]*external_forces_linear[2] - end_effector_position[2]*external_forces_linear[0]
        zmp_x = zmp_x/(mass*-gravity[2] +external_forces_linear[2])
        
        zmp_y = mass*-gravity[2]*base_w[1] - base_w[2]*mass*linear_com_acc[1]
        zmp_y = zmp_y + end_effector_position[1]*external_forces_linear[2] - end_effector_position[2]*external_forces_linear[1]
        zmp_y = zmp_y/(mass*-gravity[2] +external_forces_linear[2])
        zmp = np.array([zmp_x, zmp_y,0])
        h_R_w=R.from_euler('xyz', base_ori).as_matrix()
        # zmp = h_R_w@zmp  
        return zmp

def compute_com_acc(base_position,base_or, forces):

    base_w = base_position

    yaw = base_or[2]
    h_R_w = np.zeros((2, 2))
    h_R_w[0, 0] = np.cos(yaw)
    h_R_w[0, 1] = np.sin(yaw)
    h_R_w[1, 0] = -np.sin(yaw)
    h_R_w[1, 1] = np.cos(yaw)


    h_R_w=R.from_euler('xyz', base_or).as_matrix()

    # print('rotation matrix',h_R_w)
    # h_R_w=R.from_euler('z', base_or[2]).as_matrix()
    # print('rotation matrix',h_R_w)

    foot_force_fl = forces['FL']  #@param[0]
    foot_force_fr = forces['FR']  #@param[1]
    foot_force_rl = forces['RL']  #@param[2]
    foot_force_rr = forces['RR']  #@param[3]
    temp = foot_force_fl + foot_force_fr + foot_force_rl + foot_force_rr
    gravity = np.array([0, 0, -9.81])
    mass=24  #robot mass only the total mass is given by the external force along x!
    linear_com_acc = (1 / mass) * temp + gravity
    # ext_wrenches=np.zeros(6)

    
    return linear_com_acc

def compute_center_of_pressure(base_position,base_or,feet_pos,forces):
    foot_force_fl = forces['FL'][2]
    foot_force_fr = forces['FR'][2]
    foot_force_rl = forces['RL'][2]
    foot_force_rr = forces['RR'][2]
    base_w = copy.deepcopy(base_position)

    FL = copy.deepcopy(feet_pos['FL'])
    FR = copy.deepcopy(feet_pos['FR'])
    RL = copy.deepcopy(feet_pos['RL'])
    RR = copy.deepcopy(feet_pos['RR'])
    yaw = copy.deepcopy(base_or[2])

    yaw = base_or[2]
    h_R_w = np.zeros((2, 2))
    h_R_w[0, 0] = np.cos(yaw)
    h_R_w[0, 1] = np.sin(yaw)
    h_R_w[1, 0] = -np.sin(yaw)
    h_R_w[1, 1] = np.cos(yaw)
    # breakpoint()
    FL[0:2] = h_R_w @ (FL[0:2] - base_w[0:2])
    FR[0:2] = h_R_w @ (FR[0:2] - base_w[0:2])
    RL[0:2] = h_R_w @ (RL[0:2] - base_w[0:2])
    RR[0:2] = h_R_w @ (RR[0:2] - base_w[0:2])

    temp=foot_force_fl + foot_force_fr + foot_force_rl + foot_force_rr
    numerator=(FL*foot_force_fl+FR*foot_force_fr+RL*foot_force_rl+RR*foot_force_rr)


    # print('temp',temp)

    

    if np.all(temp) ==0:
        cop=[0,0,0]
    else:
        cop=numerator/temp
    cop=[cop[0],cop[1]]

    cop=cop+base_w[0:2]
    return cop


def compute_cop_margin_back_leg(base_position,base_or,feet_pos,forces,contact_pos_back1,contact_pos_back2):
    
    foot_force_fl = forces['FL']  #@param[0]
    foot_force_fr = forces['FR']  #@param[1]
    foot_force_rl = forces['RL']  #@param[2]
    foot_force_rr = forces['RR']  #@param[3]
    base_w = copy.deepcopy(base_position)
    FL = copy.deepcopy(feet_pos['FL'])
    FR = copy.deepcopy(feet_pos['FR'])
    RL = copy.deepcopy(feet_pos['RL'])
    RR = copy.deepcopy(feet_pos['RR'])
    yaw = copy.deepcopy(base_or[2])

    yaw = base_or[2]
    h_R_w = np.zeros((2, 2))
    h_R_w[0, 0] = np.cos(yaw)
    h_R_w[0, 1] = np.sin(yaw)
    h_R_w[1, 0] = -np.sin(yaw)
    h_R_w[1, 1] = np.cos(yaw)
    # breakpoint()
    FL[0:2] = h_R_w @ (FL[0:2] - base_w[0:2])
    FR[0:2] = h_R_w @ (FR[0:2] - base_w[0:2])
    RL[0:2] = h_R_w @ (RL[0:2] - base_w[0:2])
    RR[0:2] = h_R_w @ (RR[0:2] - base_w[0:2])

    temp=foot_force_fl + foot_force_fr + foot_force_rl + foot_force_rr

    numerator=(FL*foot_force_fl+FR*foot_force_fr+RL*foot_force_rl+RR*foot_force_rr)

    if np.all(temp) ==0:
        cop=[0,0,0]
    else:
        cop=numerator/temp[2]
    cop=[cop[0],cop[1]]
    cop=cop+base_w[0:2]
    # return cop
    p1 = contact_pos_back1
    p2 = contact_pos_back2
    x1, y1 = p1
    x2, y2 = p2
    zmp_x=cop[0]
    zmp_y=cop[1]
    numerator = abs((x2 - x1) * (y1 - zmp_y) - (y2 - y1) * (x1 - zmp_x))
    denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


    if denominator == 0:
        margin=-1
    else:
        margin = numerator / denominator
    # if margin>10000:
    #     breakpoint()
    return margin

def compute_zmp_margin_back_line(zmp, contact_pos_back1,contact_pos_back2):

    p1 = contact_pos_back1
    p2 = contact_pos_back2
    x1, y1 = p1
    x2, y2 = p2
    zmp_x, zmp_y = zmp
    numerator = (x2 - x1) * -( zmp_y - y1) - (y2 - y1) * -( zmp_x - x1)
    denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if denominator == 0:
        zmp_margin=10000
    else:
        zmp_margin = numerator / denominator
    return zmp_margin

def plot_current_foot_prediction(env,geom_id_foot,contact_state,foot_pos_curr):

            i=0
            # breakpoint()
            feet_ids=['FL','FR','RL','RR']

            
            for feet_id in feet_ids:

                if contact_state[feet_id]:
                    colour_feet=[1,0,0,0.6]
                else:
                    colour_feet=[0,0,1,0.6]

                geom_id_foot[i]=render_sphere(viewer=env.viewer,
                                        position=foot_pos_curr[feet_id],
                                        diameter=0.05,
                                        color=colour_feet,
                                        geom_id=geom_id_foot[i]
                                        )  
                i=i+1
            return geom_id_foot

def signed_distance(point,p1,p2):
    """
    Compute the signed distance from a point to the line defined by points p1 and p2.
    
    The sign of the distance indicates which side of the line the point lies on.
    """
    x0, y0 = point[0:2]
    x1, y1 = p1[0:2]
    x2, y2 = p2[0:2]
    numerator =(x2 - x1) * (y1-y0) - (y2 - y1) * (x1-x0)
    denominator = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if denominator == 0:
        raise ValueError("p1 and p2 cannot be the same point")
    margin=numerator / denominator
    return margin
    # if margin < 0.07:
    #     margin=0
    

def compute_zmp_margin(zmp,feet_pos,contact_state):
        # zmp_giulio=compute_zmp_giulio(base_pos[i],linear_com_acc,wrench_estimate[i],end_effector_pos[i],h_R_w)
        # zmp=zmp+base_pos[i][0:2] NO!

        # zmp=[zmp_exp[i][0],zmp_exp[i][1]]
        # print('zmp',zmp)
        # print('base pos',base_pos[i][0:2])
        # then I compute the zmp margin with respect to each side of the polygon
        list_margins=[]
        # zmp=zmp_exp[i][0:2]
        temp_marg_1=signed_distance(zmp,feet_pos['FL'],feet_pos['FR'])#*contact_state[i]['FL']*contact_state[i]['FR']
        temp_marg_2=signed_distance(zmp,feet_pos['FR'],feet_pos['RR'])#*contact_state[i]['FR']*contact_state[i]['RR']
        temp_marg_3=signed_distance(zmp,feet_pos['RR'],feet_pos['RL'])#*contact_state[i]['RR']*contact_state[i]['RL']
        temp_marg_4=signed_distance(zmp,feet_pos['RL'],feet_pos['FL'])#*contact_state[i]['RL']*contact_state[i]['FL']
        temp_marg_5_1=signed_distance(zmp,feet_pos['FL'],feet_pos['RR'])
        temp_marg_5_2=-temp_marg_5_1
        temp_marg_6_1=signed_distance(zmp,feet_pos['FR'],feet_pos['RL'])
        temp_marg_6_2=-temp_marg_6_1
        if contact_state['FL']*contact_state['FR']*contact_state['RL']*contact_state['RR']:
            list_margins=(temp_marg_1,temp_marg_2,temp_marg_3,temp_marg_4)
        elif contact_state['FL']*contact_state['FR']*contact_state['RL']:
            list_margins=(temp_marg_1,temp_marg_4,temp_marg_6_1)
        elif contact_state['FL']*contact_state['FR']*contact_state['RR']:
            list_margins=(temp_marg_1,temp_marg_2,temp_marg_5_2)
            # print("Diagonal 5.2")
            # print("temp 5.2",temp_marg_5_2)
        elif contact_state['FL']*contact_state['RL']*contact_state['RR']:
            list_margins=(temp_marg_4,temp_marg_3,temp_marg_5_1)
            # print("Diagonal 5.1 ")
            # print("temp 5.1",temp_marg_5_1)

        elif contact_state['FR']*contact_state['RL']*contact_state['RR']:
            list_margins=(temp_marg_2,temp_marg_3,temp_marg_6_2)
        elif contact_state['FL']*contact_state['RR']:
            list_margins=(signed_distance(zmp, feet_pos['FL'], feet_pos['RR']),)
        elif contact_state['FR']*contact_state['RL']:
            list_margins=(signed_distance(zmp, feet_pos['FR'], feet_pos['RL']),)
        else:
            list_margins=[0,0,0,0]
        # list_margins
        # zmp_margin_M=max(list_margins)
        zmp_margin_m=min(list_margins)
        # if zmp_margin_m<0:
        #     zmp_margin_m=0
        return zmp_margin_m



def compute_capture_point(robot_com, robot_com_vel):
    '''
    A Capture Point is a point on the ground where the robot can step to in order to bring itself to a complete stop.
    The formula is given according to "Capture Point: A Step Toward Humanoid Push Recovery" by Pratt et al.

    '''

    k= np.sqrt(9.81/robot_com[2])

    capture_point = robot_com[0:2] + robot_com_vel[0:2] / k

    return capture_point



'''


def compute_zmp_arm_contribution(eef_pos=np.zeros(3),force=np.zeros(6),m2=0.5,m_tot=24):
        

    temp1=(eef_pos[0]*(9.81*m2))   # fx*z
    temp2=(eef_pos[2]*force[0]) # fz*x
    temp3=(m_tot*9.81+(9.81*m2))
    zmp= (eef_pos[0]*force[2] + eef_pos[2]*force[0])/(m_tot*9.81+(9.81*m2))

    zmp=np.array([-zmp,0,0])
    return zmp

def compute_zmp(base_position,feet_pos,base_or, contact_status, forces,ext_wrenches,
                                              Pee,lift_off_pos):
    # TODO: This import should go
    base_w = base_position
    # base_vel_w = copy.deepcopy(state['linear_velocity'])

    # FL = copy.deepcopy(feet_pos['FL'])
    # FR = copy.deepcopy(feet_pos['FR'])
    # RL = copy.deepcopy(feet_pos['RL'])
    # RR = copy.deepcopy(feet_pos['RR'])

    yaw = base_or[2]
    h_R_w = np.zeros((2, 2))
    h_R_w[0, 0] = np.cos(yaw)
    h_R_w[0, 1] = np.sin(yaw)
    h_R_w[1, 0] = -np.sin(yaw)
    h_R_w[1, 1] = np.cos(yaw)
    #h_R_w is the rotation matrix from world to base

    h_R_w=R.from_euler('xyz', base_or).as_matrix()

    # print('rotation matrix',h_R_w)
    # h_R_w=R.from_euler('z', base_or[2]).as_matrix()
    # print('rotation matrix',h_R_w)

    foot_force_fl = forces['FL']  #@param[0]
    foot_force_fr = forces['FR']  #@param[1]
    foot_force_rl = forces['RL']  #@param[2]
    foot_force_rr = forces['RR']  #@param[3]
    temp = foot_force_fl + foot_force_fr + foot_force_rl + foot_force_rr
    gravity = np.array([0, 0, -9.81])
    mass=24  #robot mass only the total mass is given by the external force along x!
    linear_com_acc = (1 / mass) * temp + gravity
    # ext_wrenches=np.zeros(6)

    # Pee=h_R_w@Pee

    if (config.mpc_params['use_zmp_stability']):
        # zmp = base_w[0:2] - linear_com_acc[0:2] * (robotHeight / -gravity_z) #zmp formulation no external forces
        # zmp_x = zmp[0]
        # zmp_y = zmp[1] 
        # Ok up until here the next step is to calculate the zmp from the external wrenches

        # # Lets do it with external forces

        # zmp_x = (mass*-gravity_z*base_w[0])\
        #         -(base_w[2]*mass*linear_com_acc[0]) \
        #          + (Pee[0]*ext_wrenches[2] - Pee[2]*ext_wrenches[0])
        # zmp_x = zmp_x/(mass*-gravity_z + ext_wrenches[2])

        # zmp_y = (mass*-gravity_z*base_w[1])\
        #         -(base_w[2]*mass*linear_com_acc[1]) \
        #          + (Pee[1]*ext_wrenches[2] - Pee[2]*ext_wrenches[1])
        # zmp_y=zmp_y/(mass*-gravity_z+ext_wrenches[2])

        ## adding zmp from controller

        #add -0.1 this is for the com position to center the base one when a weight is applied

        zmp_com_pos = mass*gravity[2]*(base_w[0])/(mass*gravity[2]+ext_wrenches[2])

        zmp_com_acc= base_w[2]*mass*linear_com_acc[0]/(mass*gravity[2]+ext_wrenches[2])

        # temp_x = Pee[0]*ext_wrenches[2]

        zmp_x_ext_forces =  (Pee[0]*ext_wrenches[2] - Pee[2]*ext_wrenches[0])/(mass*gravity[2]+ext_wrenches[2])

        # zmp_x = zmp_x/(mass*-gravity[2] +ext_wrenches[2])

        # This is without external forces
        # zmp_x = (zmp_com_pos - zmp_com_acc)/(mass*-gravity[2] )

        # Lets do it with external forces
        zmp_x = (zmp_com_pos + zmp_com_acc + zmp_x_ext_forces)
        # zmp_x = zmp_x + 0.08
        
        # zmp_y = mass*-gravity[2]*base_w[1] - base_w[2]*mass*linear_com_acc[1]
        # zmp_y = zmp_y + Pee[1]*ext_wrenches[2] - Pee[2]*ext_wrenches[1]
        # zmp_y = zmp_y/(mass*-gravity[2] +ext_wrenches[2])


        zmp_com_pos_y = mass*gravity[2]*(base_w[1])/(mass*gravity[2]+ext_wrenches[2])

        zmp_com_acc_y= base_w[2]*mass*linear_com_acc[1]/(mass*gravity[2]+ext_wrenches[2])

        # temp_x = Pee[0]*ext_wrenches[2]

        zmp_y_ext_forces =  (Pee[1]*ext_wrenches[2] - Pee[2]*ext_wrenches[1])/(mass*gravity[2]+ext_wrenches[2])

        zmp_y = (zmp_com_pos_y + zmp_com_acc_y + zmp_y_ext_forces)
        ## debug

        # print("Pee",Pee[0])
        # print("base_pos_x",base_w[0])
        # print("diff",-Pee[0]+base_w[0])
        # print("zmp_x",zmp_x)
        # print("zmp_com_pos",zmp_com_pos)
        # print("zmp_com_acc",zmp_com_acc)
        # print("zmp_x_ext_forces",zmp_x_ext_forces)

        zmp = np.array([zmp_x,zmp_y,0])


    else:
        x = base_position[0]
        y = base_position[1]
        zmp = np.array([x,y,0])

    # zmp[0:2] =  (zmp[0:2] +base_w[0:2])
    return zmp

    
    def zmp_check_and_save(contact_state,legs_contact_pos,triangle_pass,colour_list,zmp_pass,zmpl_rot,triangle_scaled_list,zmp_scaled_list,triangle_one,triangle_two,triangle_three,triangle_four):
    legs_contact_pos_copy=copy.deepcopy(legs_contact_pos)
    zmp_rot_copy=copy.deepcopy(zmpl_rot)
    if contact_state['FL'] and contact_state['RR'] and contact_state['FR']:
        if contact_state['RL']:
            colour_leg=[1,1,1,1]
            colour_list.append(colour_leg)
            triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]])
            zmp_pass.append(zmpl_rot)
            # continue
        else:

            colour_leg= [0,0,1, 1]
            colour_list.append(colour_leg)
            triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
            zmp_pass.append(zmpl_rot)
            ############################################
            legs_contact_pos_copy[0][1]=legs_contact_pos_copy[0][1]-0.3
            legs_contact_pos_copy[1][1]=legs_contact_pos_copy[1][1]-0.3
            legs_contact_pos_copy[2][1]=legs_contact_pos_copy[2][1]-0.3
            legs_contact_pos_copy[0][0]=legs_contact_pos_copy[0][0]+0.3
            legs_contact_pos_copy[1][0]=legs_contact_pos_copy[1][0]+0.3
            legs_contact_pos_copy[2][0]=legs_contact_pos_copy[2][0]+0.3
            zmp_rot_copy[0]=zmp_rot_copy[0]+0.3
            zmp_rot_copy[1]=zmp_rot_copy[1]-0.3
            zmp_scaled_list.append(zmp_rot_copy)
            triangle_scaled_list.append([legs_contact_pos_copy[0],legs_contact_pos_copy[1],legs_contact_pos_copy[2]])
            triangle_one.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
    #triangle 2 azzurro
    elif contact_state['FR'] and contact_state['RR'] and contact_state['RL']:
        if contact_state['FL']:
            colour_leg=[1,1,1,0.4]
            colour_list.append(colour_leg)
            triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]])
            zmp_pass.append(zmpl_rot)
            # continue
        else:
            colour_leg= [1, 0, 1, 0.4]
            colour_list.append(colour_leg)
            triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
            zmp_pass.append(zmpl_rot)
            legs_contact_pos_copy[0][1]=legs_contact_pos_copy[0][1]-0.3
            legs_contact_pos_copy[1][1]=legs_contact_pos_copy[1][1]-0.3
            legs_contact_pos_copy[2][1]=legs_contact_pos_copy[2][1]-0.3
            legs_contact_pos_copy[0][0]=legs_contact_pos_copy[0][0]-0.3
            legs_contact_pos_copy[1][0]=legs_contact_pos_copy[1][0]-0.3
            legs_contact_pos_copy[2][0]=legs_contact_pos_copy[2][0]-0.3
            zmp_rot_copy[0]=zmp_rot_copy[0]-0.3
            zmp_rot_copy[1]=zmp_rot_copy[1]-0.3
            zmp_scaled_list.append(zmp_rot_copy)
            triangle_scaled_list.append([legs_contact_pos_copy[0],legs_contact_pos_copy[1],legs_contact_pos_copy[2]])
            triangle_two.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
    #triangle 3 magenta
    elif contact_state['FR'] and contact_state['FL'] and contact_state['RL']:
        if contact_state['RR']:
            colour_leg=[1,1,1,1]
            colour_list.append(colour_leg)
            triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]])
            zmp_pass.append(zmpl_rot)
        #     # continue
        else:

            colour_leg= [0, 1,1, 0.4]
            colour_list.append(colour_leg)
            zmp_pass.append(zmpl_rot)
            triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
            legs_contact_pos_copy[0][1]=legs_contact_pos_copy[0][1]+0.3
            legs_contact_pos_copy[1][1]=legs_contact_pos_copy[1][1]+0.3
            legs_contact_pos_copy[2][1]=legs_contact_pos_copy[2][1]+0.3
            legs_contact_pos_copy[0][0]=legs_contact_pos_copy[0][0]+0.3
            legs_contact_pos_copy[1][0]=legs_contact_pos_copy[1][0]+0.3
            legs_contact_pos_copy[2][0]=legs_contact_pos_copy[2][0]+0.3
            zmp_rot_copy[0]=zmp_rot_copy[0]+0.3
            zmp_rot_copy[1]=zmp_rot_copy[1]+0.3
            zmp_scaled_list.append(zmp_rot_copy)
            triangle_scaled_list.append([legs_contact_pos_copy[0],legs_contact_pos_copy[1],legs_contact_pos_copy[2]])
            triangle_three.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
    #triangle 4 giallo 
    elif contact_state['RR'] and contact_state['FL'] and contact_state['RL']:
        if contact_state['FR']:
            colour_leg=[1,1,1,1]
            colour_list.append(colour_leg)
            triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]])
            zmp_pass.append(zmpl_rot)
            # continue
        else:

            colour_leg= [1, 1, 0, 0.4]
            colour_list.append(colour_leg)
            triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
            zmp_pass.append(zmpl_rot)
            legs_contact_pos_copy[0][1]=legs_contact_pos_copy[0][1]+0.3
            legs_contact_pos_copy[1][1]=legs_contact_pos_copy[1][1]+0.3
            legs_contact_pos_copy[2][1]=legs_contact_pos_copy[2][1]+0.3
            legs_contact_pos_copy[0][0]=legs_contact_pos_copy[0][0]-0.3
            legs_contact_pos_copy[1][0]=legs_contact_pos_copy[1][0]-0.3
            legs_contact_pos_copy[2][0]=legs_contact_pos_copy[2][0]-0.3
            zmp_rot_copy[0]=zmp_rot_copy[0]-0.3
            zmp_rot_copy[1]=zmp_rot_copy[1]+0.3
            zmp_scaled_list.append(zmp_rot_copy)
            triangle_scaled_list.append([legs_contact_pos_copy[0],legs_contact_pos_copy[1],legs_contact_pos_copy[2]])
            triangle_four.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
    # else:
    #     colour_leg=[1,1,1,1]
    # triangle_list_seprated=[triangle_one,triangle_two,triangle_three,triangle_four]
    return zmp_pass,triangle_pass,colour_list,triangle_scaled_list,zmp_scaled_list,triangle_one,triangle_two,triangle_three,triangle_four

    

def compute_zmp_margin_gpt(zmp, feet_pos, contact_state):
    from matplotlib.path import Path
    """
    Compute the ZMP margin based on the support polygon.

    Parameters:
        zmp (tuple): ZMP position (x, y).
        feet_pos (list of tuples): List of feet positions [(x, y, z), ...].
        contact_state (list): List of contact states for each foot [1, 0, 1, 0].

    Returns:
        float: ZMP margin.
    """
    # Filter active contact points
    """
    Compute the ZMP margin based on the support polygon.

    Parameters:
        zmp (tuple): ZMP position (x, y).
        feet_pos (list of tuples): List of feet positions [(x, y, z), ...].
        contact_state (list): List of contact states for each foot [1, 0, 1, 0].

    Returns:
        float: ZMP margin. Returns None if no valid contacts exist.
    """
    # Filter active contact points
    active_contacts = [pos[:2] for pos, contact in zip(feet_pos, contact_state) if contact == 1]

    # Safety check: Ensure there are at least two contact points
    if len(active_contacts) < 2:
        # print("Warning: Not enough contacts to compute a valid support polygon.")
        zmp_margin= 100
        return zmp_margin

    active_contacts = np.array(active_contacts)

    # Close the support polygon
    support_polygon = np.vstack([active_contacts, active_contacts[0]])
    
    # Compute minimum distance to polygon edges
    zmp_margin = float("inf")
    ## Check if ZMP is inside the support polygon
    # path = Path(support_polygon)
    # if not path.contains_point(zmp):
    #     zmp_margin = 0  # ZMP is outside the polygon
    #     return zmp_margin # ZMP is outside the polygon
    

    for i in range(len(support_polygon) - 1):
        p1 = support_polygon[i]
        p2 = support_polygon[i + 1]

        # Compute perpendicular distance
        x1, y1 = p1
        x2, y2 = p2
        zmp_x, zmp_y = zmp

        numerator = (x2 - x1) * (y1 - zmp_y) - (y2 - y1) * (x1 - zmp_x)
        denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if denominator == 0:
            continue
        distance = numerator / denominator
        zmp_margin = min(zmp_margin, distance)

    return zmp_margin



def plot_zmp_trajectory(time, zmp_list):
    
    """
    Plot the ZMP trajectory over time.

    Parameters:
        time (list): A list of time values (in seconds).
        zmp_list (list of tuples): A list of (x_z, y_z) positions of the ZMP at each time step.
    """
    # Extract x_z and y_z positions

    # breakpoint()

    # Extract x_z and y_z positions
    zmp_x, zmp_y = [], []
    for pos in zmp_list:
        zmp_x.append(pos[0])
        zmp_y.append(pos[1])                    
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    
    # Plot the trajectory
    plt.plot(zmp_x, zmp_y, marker='o', label="ZMP Trajectory")
    
    # Highlight start and end points
    plt.scatter(zmp_x[0], zmp_y[0], color='green', label="Start Point", zorder=5)
    plt.scatter(zmp_x[-1], zmp_y[-1], color='red', label="End Point", zorder=5)
    
    # Add labels, legend, and title
    plt.xlabel("ZMP x-coordinate ($x_z$) [m]")
    plt.ylabel("ZMP y-coordinate ($y_z$) [m]")
    plt.title("ZMP Trajectory in x-y Plane")
    plt.legend()
    plt.grid()
    plt.axis("equal")  # Ensure equal scaling for x and y axes
    
    # Show the plot
    plt.show()




def check_zmp_constraint_satisfaction(base_position,feet_pos,base_or, contact_status, forces):
    # TODO: This import should go
   

    base_w = copy.deepcopy(base_position)
    # base_vel_w = copy.deepcopy(state['linear_velocity'])

    # FL = copy.deepcopy(feet_pos['FL'])
    # FR = copy.deepcopy(feet_pos['FR'])
    # RL = copy.deepcopy(feet_pos['RL'])
    # RR = copy.deepcopy(feet_pos['RR'])
    FL = copy.deepcopy(feet_pos['FL'])
    FR = copy.deepcopy(feet_pos['FR'])
    RL = copy.deepcopy(feet_pos['RL'])
    RR = copy.deepcopy(feet_pos['RR'])
    yaw = copy.deepcopy(base_or[2])
    # h_R_w = np.zeros((2, 2))
    # h_R_w[0, 0] = np.cos(yaw)
    # h_R_w[0, 1] = np.sin(yaw)
    # h_R_w[1, 0] = -np.sin(yaw)
    # h_R_w[1, 1] = np.cos(yaw)

    h_R_w=R.from_euler('xyz', base_or).as_matrix()
    


    FL[0:2] = h_R_w @ (FL[0:2] - base_w[0:2])
    FR[0:2] = h_R_w @ (FR[0:2] - base_w[0:2])
    RL[0:2] = h_R_w @ (RL[0:2] - base_w[0:2])
    RR[0:2] = h_R_w @ (RR[0:2] - base_w[0:2])

    foot_force_fl = forces['FL2']  # @param[0]
    foot_force_fr = forces['FR2']  # @param[1]
    foot_force_rl = forces['RL2']  # @param[2]
    foot_force_rr = forces['RR2']  # @param[3]
    temp = foot_force_fl + foot_force_fr + foot_force_rl + foot_force_rr
    gravity = np.array([0, 0, -9.81])
    linear_com_acc = (1 / config.mass) * temp + gravity

    if (config.mpc_params['use_zmp_stability']):
        gravity_z = 9.81
        robotHeight = base_w[2]
        zmp = base_w[0:2] - linear_com_acc[0:2] * (robotHeight / gravity_z)
        zmp = h_R_w @ (zmp - base_w[0:2])
        x = zmp[0]
        y = zmp[1]
    else:
        x = 0.0
        y = 0.0

    y_FL = FL[1]
    y_FR = FR[1]
    y_RL = RL[1]
    y_RR = RR[1]

    x_FL = FL[0]
    x_FR = FR[0]
    x_RL = RL[0]
    x_RR = RR[0]

    # LF - RF : x < (x2 - x1) (y - y1) / (y2 - y1) + x1
    # RF - RH: y > (y2 - y1) (x - x1) / (x2 - x1) + y1
    # RH - LH : x > (x2 - x1) (y - y1) / (y2 - y1) + x1
    # LH - LF: y < (y2 - y1) (x - x1) / (x2 - x1) + y1

    # FL and FR cannot stay at the same x! #constrint should be less than zero
    constraint_FL_FR = x - (x_FR - x_FL) * (y - y_FL) / (y_FR - y_FL + 0.001) - x_FL

    # FR and RR cannot stay at the same y! #constraint should be bigger than zero
    constraint_FR_RR = y - (y_RR - y_FR) * (x - x_FR) / (x_RR - x_FR + 0.001) - y_FR

    # RL and RR cannot stay at the same x! #constraint should be bigger than zero
    constraint_RR_RL = x - (x_RL - x_RR) * (y - y_RR) / (y_RL - y_RR + 0.001) - x_RR

    # FL and RL cannot stay at the same y! #constraint should be less than zero
    constraint_RL_FL = y - (y_FL - y_RL) * (x - x_RL) / (x_FL - x_RL + 0.001) - y_RL

    # the diagonal stuff can be at any point...
    constraint_FL_RR = y - (y_RR - y_FL) * (x - x_FL) / (x_RR - x_FL + 0.001) - y_FL  # bigger than zero
    constraint_FR_RL = y - (y_RL - y_FR) * (x - x_FR) / (x_RL - x_FR + 0.001) - y_FR  # bigger than zero

    FL_contact = contact_status['FL2']
    FR_contact = contact_status['FR2']
    RL_contact = contact_status['RL2']
    RR_contact = contact_status['RR2']

    violation = 0

    if (FL_contact == 1):
        if (FR_contact == 1):
            # ub_support_FL_FR = -0.0
            # lb_support_FL_FR = -1000
            if (constraint_FL_FR > 0):
                violation = violation + 1
                constr_value = constraint_FL_FR

        else:
            # ub_support_FL_RR = 1000
            # lb_support_FL_RR = 0.0
            if (constraint_FL_RR < 0):
                violation = violation + 1
                constr_value = constraint_FL_FR

    if (FR_contact == 1):
        if (RR_contact == 1):
            # ub_support_FR_RR = 1000
            # lb_support_FR_RR = 0.0
            if (constraint_FR_RR < 0):
                violation = violation + 1
                constr_value = constraint_FL_FR

        else:
            ub_support_FR_RL = 1000
            lb_support_FR_RL = 0.0
            # if(constraint_FR_RL < 0):
            # violation = violation + 1
            # TOCHECK

    if (RR_contact == 1):
        if (RL_contact == 1):
            # ub_support_RR_RL = 1000
            # lb_support_RR_RL = 0.0
            if (constraint_RR_RL < 0):
                violation = violation + 1
                constr_value = constraint_FL_FR

        else:
            # ub_support_FL_RR = -0.0
            # lb_support_FL_RR = -1000
            if (constraint_FL_RR > 0):
                violation = violation + 1 * 0
                constr_value = constraint_FL_FR

    if (RL_contact == 1):
        if (FL_contact == 1):
            # ub_support_RL_FL = -0.0
            # lb_support_RL_FL = -1000
            if (constraint_RL_FL > 0):
                violation = violation + 1
                constr_value = constraint_FL_FR
        else:
            # ub_support_FR_RL = -0.0
            # lb_support_FR_RL = -1000
            if (constraint_FR_RL > 0):
                violation = violation + 1
                constr_value = constraint_FL_FR

    if (FL_contact == 1 and FR_contact == 1 and RL_contact == 1 and RR_contact == 1):
        violation = 0

    if (violation >= 1):
        return constr_value 
    else:
        constr_value = 0.04
        return constr_value

        

        ## plotting the trinaglesss


def zmp_triangle_compute(zmp,legs_order,contact_state,feet_pos):
    # legs_contact_pos_copy=copy.deepcopy(legs_contact_pos)
    # zmp_rot_copy=copy.deepcopy(zmpl_rot)
    legs_contact_pos=[]
    
    triangle_scaled=[]
    triangle=[]
    zmp_scaled=[0,0]
    colour_leg=[]

    for leg_id in legs_order :
        if contact_state[leg_id]:
            # # # # # # # print("contact")
            legs_contact_pos.append(feet_pos[leg_id])
        # else:
        #     legs_contact_pos.append([0,0,0,0])
    
    legs_contact_pos_scaled=copy.deepcopy(legs_contact_pos)

    if contact_state['FL'] and contact_state['RR'] and contact_state['FR']:
        if contact_state['RL']:
            colour_leg=[1,1,1,1]
            triangle=[legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]]
            zmp_scaled=zmp
            triangle_scaled=[legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]]
            # colour_list.append(colour_leg)
            # triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]])\
            # zmp_pass.append(zmpl_rot)
            # continue
        else:

            colour_leg= [0,0,1, 1] #blue
            triangle=[legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]]
            # colour_list.append(colour_leg)
            # triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
            # zmp_pass.append(zmpl_rot)
            ############################################
            # breakpoint()
            legs_contact_pos_scaled[0][1]=legs_contact_pos[0][1]-0.5
            legs_contact_pos_scaled[1][1]=legs_contact_pos[1][1]-0.5
            legs_contact_pos_scaled[2][1]=legs_contact_pos[2][1]-0.5
            legs_contact_pos_scaled[0][0]=legs_contact_pos[0][0]+0.5
            legs_contact_pos_scaled[1][0]=legs_contact_pos[1][0]+0.5
            legs_contact_pos_scaled[2][0]=legs_contact_pos[2][0]+0.5
            zmp_scaled[0]=zmp[0]+0.5
            zmp_scaled[1]=zmp[1]-0.5
            triangle_scaled=[legs_contact_pos_scaled[0],legs_contact_pos_scaled[1],legs_contact_pos_scaled[2]]
            # zmp_scaled_list.append(zmp_rot_copy)
            # triangle_scaled_list.append([legs_contact_pos_copy[0],legs_contact_pos_copy[1],legs_contact_pos_copy[2]])
            # triangle_one.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
    #triangle 2 azzurro
    elif contact_state['FR'] and contact_state['RR'] and contact_state['RL']:
        if contact_state['FL']:
            colour_leg=[1,1,1,1] #white
            triangle=[legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]]
            zmp_scaled=zmp
            triangle_scaled=[legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]]
            # colour_list.append(colour_leg)
            # triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]])
            # zmp_pass.append(zmpl_rot)
            # continue
        else:
            colour_leg= [1, 0, 1, 0.4] #magenta
            triangle=[legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]]
            # colour_list.append(colour_leg)
            # triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
            # zmp_pass.append(zmpl_rot)
            # breakpoint()
            legs_contact_pos_scaled[0][1]=legs_contact_pos[0][1]-0.5
            legs_contact_pos_scaled[1][1]=legs_contact_pos[1][1]-0.5
            legs_contact_pos_scaled[2][1]=legs_contact_pos[2][1]-0.5
            legs_contact_pos_scaled[0][0]=legs_contact_pos[0][0]-0.5
            legs_contact_pos_scaled[1][0]=legs_contact_pos[1][0]-0.5
            legs_contact_pos_scaled[2][0]=legs_contact_pos[2][0]-0.5
            zmp_scaled[0]=zmp[0]-0.5
            zmp_scaled[1]=zmp[1]-0.5
            triangle_scaled=[legs_contact_pos_scaled[0],legs_contact_pos_scaled[1],legs_contact_pos_scaled[2]]

            # triangle=[legs_contact_pos_scaled[0],legs_contact_pos_scaled[1],legs_contact_pos_scaled[2]]
            # zmp_scaled_list.append(zmp_rot_copy)
            # triangle_scaled_list.append([legs_contact_pos_copy[0],legs_contact_pos_copy[1],legs_contact_pos_copy[2]])
            # triangle_two.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
    #triangle 3 magenta
    elif contact_state['FR'] and contact_state['FL'] and contact_state['RL']:
        if contact_state['RR']:
            colour_leg=[1,1,1,1]
            triangle=[legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]]
            zmp_scaled=zmp
            triangle_scaled=[legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]]
            # colour_list.append(colour_leg)
            # triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]])
            # zmp_pass.append(zmpl_rot)
        #     # continue
        else:

            colour_leg= [0, 1,1, 0.4] #cyan
            triangle=[legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]]
            # colour_list.append(colour_leg)
            # zmp_pass.append(zmpl_rot)
            # triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
            legs_contact_pos_scaled[0][1]=legs_contact_pos[0][1]+0.5
            legs_contact_pos_scaled[1][1]=legs_contact_pos[1][1]+0.5
            legs_contact_pos_scaled[2][1]=legs_contact_pos[2][1]+0.5
            legs_contact_pos_scaled[0][0]=legs_contact_pos[0][0]+0.5
            legs_contact_pos_scaled[1][0]=legs_contact_pos[1][0]+0.5
            legs_contact_pos_scaled[2][0]=legs_contact_pos[2][0]+0.5
            zmp_scaled[0]=zmp[0]+0.5
            zmp_scaled[1]=zmp[1]+0.5
            triangle_scaled=[legs_contact_pos_scaled[0],legs_contact_pos_scaled[1],legs_contact_pos_scaled[2]]

            # triangle=[legs_contact_pos_scaled[0],legs_contact_pos_scaled[1],legs_contact_pos_scaled[2]]
            # zmp_scaled_list.append(zmp_rot_copy)
            # triangle_scaled_list.append([legs_contact_pos_copy[0],legs_contact_pos_copy[1],legs_contact_pos_copy[2]])
            # triangle_three.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
    #triangle 4 giallo 
    elif contact_state['RR'] and contact_state['FL'] and contact_state['RL']:
        if contact_state['FR']:
            colour_leg=[1,1,1,1]
            triangle=[legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]]
            zmp_scaled=zmp
            triangle_scaled=[legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]]
            # colour_list.append(colour_leg)
            # triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2],legs_contact_pos[3]])
            # zmp_pass.append(zmpl_rot)
            # continue
        else:

            colour_leg= [1, 1, 0, 0.4] #yellow
            triangle=[legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]]
            # colour_list.append(colour_leg)
            # triangle_pass.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
            # zmp_pass.append(zmpl_rot)
            legs_contact_pos_scaled[0][1]=legs_contact_pos[0][1]+0.5
            legs_contact_pos_scaled[1][1]=legs_contact_pos[1][1]+0.5
            legs_contact_pos_scaled[2][1]=legs_contact_pos[2][1]+0.5
            legs_contact_pos_scaled[0][0]=legs_contact_pos[0][0]-0.5
            legs_contact_pos_scaled[1][0]=legs_contact_pos[1][0]-0.5
            legs_contact_pos_scaled[2][0]=legs_contact_pos[2][0]-0.5
            zmp_scaled[0]=zmp[0]-0.5
            zmp_scaled[1]=zmp[1]+0.5
            triangle_scaled=[legs_contact_pos_scaled[0],legs_contact_pos_scaled[1],legs_contact_pos_scaled[2]]
            # zmp_scaled_list.append(zmp_rot_copy)
            # triangle_scaled_list.append([legs_contact_pos_copy[0],legs_contact_pos_copy[1],legs_contact_pos_copy[2]])
            # triangle_four.append([legs_contact_pos[0],legs_contact_pos[1],legs_contact_pos[2]])
    # else:
    #     colour_leg=[1,1,1,1]
    # triangle_list_seprated=[triangle_one,triangle_two,triangle_three,triangle_four]
    # return zmp_pass,triangle_pass,colour_list,triangle_scaled_list,zmp_scaled_list,triangle_one,triangle_two,triangle_three,triangle_four
    return triangle,triangle_scaled,colour_leg,zmp_scaled
'''
