import mujoco
import numpy as np
from scipy.signal import savgol_filter, butter, lfilter,filtfilt
from scipy.spatial.transform import Rotation as R
import copy
class PassiveArmInterface:
    def __init__(self, 
                 spring_gains,
                 robot,
                 joint_pos0,
                 mass_matrix,
                 damping_gains,
                 com_offset):
        
        ###
        #parameters from config
        self.robot=robot
        self.mass=np.array([mass_matrix[0],mass_matrix[1],mass_matrix[2]])
        self.spring_gains = spring_gains
        self.damping_gains = damping_gains
        # Define the parameters of the arm
        self.Jee_transpose_inv=np.zeros((3,3))
        self.Ree_Barm=np.zeros((3,3))
        self.Jee=np.zeros((3,3))
        self.Pee_Barm = np.zeros((3, 1))
        # arm referene position
        self.joint_pos0= joint_pos0
        # Follower guidance (compute_reference_velocity): dead zone and level angles [deg] and commands.
        # Defaults reproduce the original implementation; PACC paper Sec. IV: theta2 = 25 deg,
        # yaw rates 0.3 / 0.4 rad/s.
        self.guidance = dict(theta1=10.0, theta2=20.0, vx1=0.1, vx2_pos=0.15, vx2_neg=0.2,
                             psi1=10.0, psi2=20.0, wz1=0.4, wz2=0.5)
        self.p0 = np.array([0.1183, 0.0, 0.074])
        self.p1 = np.array([0.0498, 0.0, 0.034])
        self.p2 = np.array([0.1817, 0.0, 0.1845])
        self.p3 = np.array([0.0388, 0.0, -0.25])
        ###
        self.gravity_value=9.81
        self.flag_gravity_torque= 1
        self.flag_damping_torque= 1
        self.flag_spring_torque = 1
        self.torque0=np.zeros(3,)
        #this lines are dependent on the leader/follower
        if self.robot=='aliengo_leader' :

            self.PBarm_Brobot = np.array([-0.28, 0.0, 0.09]) #this is the transform from the base of the arm to the base of the robot
            self.RBarm_Brobot=-1*np.eye(3,3) #rotation base of the arm to base of the robot
            self.RBarm_Brobot[2,2]=1.0 
            self.com_offset= com_offset

        elif self.robot == 'aliengo_follower':

            self.RBarm_Brobot=np.eye(3,3) #rotation base of the arm to base of the robot
            self.PBarm_Brobot = np.array([0.28, 0.0, 0.09]) #this is the transform from the base of the arm to the base of the robot
            self.com_offset= com_offset
        else:

            print("robot not recognized")
            quit()
    
    def compute_eef_spring_gain(self,spring_gains):
            end_effector_spring_gain_33 = np.diag(spring_gains)
            K_cartesian_vec_spring = np.diag(end_effector_spring_gain_33)
            return K_cartesian_vec_spring

    def updatePeeBarm(self,q,p0,p1,p2,p3):
        '''
        This function computes the position of the end effector in the base frame
        Takes as input the joint reference positions and q which is the current position of joint 3 I suppose 
        '''
        
        self.Pee_Barm[0] = (p0[0] + p1[0]*np.cos(q[0]) - p1[1]*np.sin(q[0]) - p2[1]*np.sin(q[0]) - p3[1]*np.sin(q[0]) +
                          p2[0]*np.cos(q[0])*np.cos(q[1]) + p2[2]*np.cos(q[0])*np.sin(q[1]) +
                          p3[0]*np.cos(q[0])*np.cos(q[1])*np.cos(q[2]) + p3[2]*np.cos(q[0])*np.cos(q[1])*np.sin(q[2]) +
                          p3[2]*np.cos(q[0])*np.cos(q[2])*np.sin(q[1]) - p3[0]*np.cos(q[0])*np.sin(q[1])*np.sin(q[2])) -0.1
        #changed the signb here to match arm movement along y
        self.Pee_Barm[1] = (p0[1] + p1[1]*np.cos(q[0]) + p2[1]*np.cos(q[0]) + p3[1]*np.cos(q[0]) +
                            
                          p1[0]*np.sin(q[0]) 
                          
                          + p2[0]*np.cos(q[1])*np.sin(q[0]) 
                          
                          + p2[2]*np.sin(q[0])*np.sin(q[1]) +


                          p3[0]*np.cos(q[1])*np.cos(q[2])*np.sin(q[0]) 
                          
                          + p3[2]*np.cos(q[1])*np.sin(q[0])*np.sin(q[2]) +

                          p3[2]*np.cos(q[2])*np.sin(q[0])*np.sin(q[1]) 

                          - p3[0]*np.sin(q[0])*np.sin(q[1])*np.sin(q[2])) 
        



        self.Pee_Barm[2] = (p0[2] + p1[2] + p3[2]*np.cos(q[1] + q[2]) - p3[0]*np.sin(q[1] + q[2]) +
                          p2[2]*np.cos(q[1]) - p2[0]*np.sin(q[1])) - 0.05
        return self.Pee_Barm
    
    def updateJacobian(self,Jee,q,p1,p2,p3):
        ''''
        This function computes the jacobian of the end effector from the joint positions.
        The mapping is in the reference frame of the base of the robot.
        The jacobian is computed using the following formula:
        ****
        the inputs are
        q: joint positions
        p1: position of the first joint
        p2: position of the second joint
        p3: position of the third joint
        ****
        The output is the jacobian matrix of the eef which is 3x3
        
        '''
        # breakpoint()
        #FOLLOWER CONFIGURATION
        if self.robot == 'aliengo_follower':
            q[0]=q[0]
            q[1]=abs(q[1])
            q[2]=-q[2]

        

        Jee[0, 0] = p3[0] * (np.sin(q[0]) * np.sin(q[1]) * np.sin(q[2]) - np.cos(q[1]) * np.cos(q[2]) * np.sin(q[0])) - \
                    p3[2] * (np.cos(q[1]) * np.sin(q[0]) * np.sin(q[2]) + np.cos(q[2]) * np.sin(q[0]) * np.sin(q[1])) - \
                    p1[1] * np.cos(q[0]) - p2[1] * np.cos(q[0]) - p3[1] * np.cos(q[0]) - p1[0] * np.sin(q[0]) - \
                    p2[0] * np.cos(q[1]) * np.sin(q[0]) - p2[2] * np.sin(q[0]) * np.sin(q[1])

        Jee[0, 1] = np.cos(q[0]) * (p3[2] * np.cos(q[1] + q[2]) - p3[0] * np.sin(q[1] + q[2]) + \
                                      p2[2] * np.cos(q[1]) - p2[0] * np.sin(q[1]))

        Jee[0, 2] = np.cos(q[0]) * (p3[2] * np.cos(q[1] + q[2]) - p3[0] * np.sin(q[1] + q[2]))

        Jee[1, 0] = p3[0] * (np.cos(q[0]) * np.cos(q[1]) * np.cos(q[2]) - np.cos(q[0]) * np.sin(q[1]) * np.sin(q[2])) + \
                    p3[2] * (np.cos(q[0]) * np.cos(q[1]) * np.sin(q[2]) + np.cos(q[0]) * np.cos(q[2]) * np.sin(q[1])) + \
                    p1[0] * np.cos(q[0]) - p1[1] * np.sin(q[0]) - p2[1] * np.sin(q[0]) - p3[1] * np.sin(q[0]) + \
                    p2[0] * np.cos(q[0]) * np.cos(q[1]) + p2[2] * np.cos(q[0]) * np.sin(q[1])

        Jee[1, 1] = np.sin(q[0]) * (p3[2] * np.cos(q[1] + q[2]) - p3[0] * np.sin(q[1] + q[2]) + \
                                      p2[2] * np.cos(q[1]) - p2[0] * np.sin(q[1]))

        Jee[1, 2] = np.sin(q[0]) * (p3[2] * np.cos(q[1] + q[2]) - p3[0] * np.sin(q[1] + q[2]))

        Jee[2, 0] = 0

        Jee[2, 1] = -p3[0] * np.cos(q[1] + q[2]) - p3[2] * np.sin(q[1] + q[2]) - \
                    p2[0] * np.cos(q[1]) - p2[2] * np.sin(q[1])

        Jee[2, 2] = -p3[0] * np.cos(q[1] + q[2]) - p3[2] * np.sin(q[1] + q[2])



        return Jee
    
    def updateGravityTorque(self,q,p2,p3):
        ''''
        in the end the gravity component that counts the most should be the one acting on the second joint since the hook wight is negligible
        '''
        tau_g=np.zeros(3,)

        tau_g[0] = 0.0

        tau_g[1] = (
            -self.gravity_value * self.mass[2] * (
                (p3[0] * np.cos(q[1] + q[2])) / 2 +
                (p3[2] * np.sin(q[1] + q[2])) / 2 +
                p2[0] * np.cos(q[1]) +
                p2[2] * np.sin(q[1])
            )
            - self.gravity_value * self.mass[1]* (
                (p2[0] * np.cos(q[1])) / 2 +
                (p2[2] * np.sin(q[1])) / 2
            )
        )
        tau_g[2] = -self.gravity_value * self.mass[2] * (
            (p3[0] * np.cos(q[1] + q[2])) / 2 +
            (p3[2] * np.sin(q[1] + q[2])) / 2
        )

        return tau_g

    def updateSpringTorque(self,q,k_spring):
        '''
        This function computes the spring torque of the arm
        The spring torque is computed using the following formula:
        tau_spring = -k_spring * (qa - qoa)
        where
        qa is the actual joint position
        qoa is the joint position at the equilibrium point
        k_spring is the spring gain
        
        '''


        num_of_joints = len(q)
        length_spring = np.zeros(num_of_joints)
        tau_spring = np.zeros(num_of_joints)

        for i in range(num_of_joints):
            length_spring[i] = q[i] 

            # third joint
            if i == 2:
                tau_spring[i] = - k_spring[i] * (q[i] + self.joint_pos0[i])  
            elif i == 1:
                tau_spring[i] = - k_spring[i] * (q[i]-self.joint_pos0[i])  

            else:
                tau_spring[i] = - k_spring[i] * length_spring[i]

        return tau_spring
    
    def updateDampingTorque(self,q_dot,damping_gains):
        '''
        This function computes the damping torque of the arm
        '''
        num_of_joints = len(q_dot)
        tau_damping = np.zeros(num_of_joints)

        for i in range(num_of_joints):
            tau_damping[i] = - damping_gains[i] * q_dot[i]
        
        return tau_damping
    
    def update_Ree_Barm(self,q,Ree_Barm):
        Ree_Barm[0, 0] = np.cos(q[1] + q[2]) * np.cos(q[0])
        Ree_Barm[0, 1] = -np.sin(q[0])
        Ree_Barm[0, 2] = np.sin(q[1] + q[2]) * np.cos(q[0])
        Ree_Barm[1, 0] = np.cos(q[1] + q[2]) * np.sin(q[0])
        Ree_Barm[1, 1] = np.cos(q[0])
        Ree_Barm[1, 2] = np.sin(q[1] + q[2]) * np.sin(q[0])
        Ree_Barm[2, 0] = -np.sin(q[1] + q[2])
        Ree_Barm[2, 1] = 0
        Ree_Barm[2, 2] = np.cos(q[1] + q[2])
        return Ree_Barm
    def update_PeeBrobot(self,q,RBarm_Brobot,PBarm_Brobot,Ree_Barm,Pee_Barm):
        Ree_Brobot=RBarm_Brobot@Ree_Barm #rotation arm to base of robot
        Pee_Brobot =  PBarm_Brobot + RBarm_Brobot@Pee_Barm
        # Pee_Brobot=Pee_Brobot.flatten()
        return Pee_Brobot
    
    def skew_mat(self,Pee_Brobot):
        '''
        a skew symmetric matrix is defined as the square matrix that is equal to the negative of its transpose matrix
        '''
        skew_mat = np.array(([0, -Pee_Brobot[2], Pee_Brobot[1]],
                             [Pee_Brobot[2], 0, -Pee_Brobot[0]],
                             [-Pee_Brobot[1], Pee_Brobot[0], 0]))
        return skew_mat
    
    def update_rest_position(self,arm_joint_pos,robot_com_offset):
        # self.joint_pos0[0]= arm_joint_pos[0]
        # self.joint_pos0[1]=arm_joint_pos[1]
        # self.joint_pos0[2]=arm_joint_pos[2]
        # add a compensation for the gravity torque
        # self.torque0= robot_com_offset*50*9.81

        return self.joint_pos0


    


    def calculate_force_estimates_damping(self,Jee,arm_joint_pos,arm_joint_vel,base_orientation):
        '''
        This function calculates the estimated end-effector force according to section C1 of the paper
        fee=-(Jee^T)^-1 * (tau_spring+tau_g+tau_d)
        where
        tau_spring is the spring torque
        tau_g is the gravity torque
        tau_d is the damping torque
        the value of tau_d and tau_s are given by the spring model of the robot.xml how to get this inmujoco i set all to 0.1
        RCF code deos not account the damping matrix it is computed as Fee_Barm_est_jacobian = -arm_model.Jee_transpose_inv*(arm_model.tau_spring + arm_model.tau_g*flag_gravity_torque);


        fee is the force in the end effector frame

        '''
        # Arm Jacobian Calculation --------------------------------------------------------------------------------------

        Jee_analytic = self.updateJacobian(np.array(Jee, copy=True), arm_joint_pos, self.p1, self.p2, self.p3)
        Jee_passed = np.array(Jee, copy=True)
        use_passed_jacobian = np.all(np.isfinite(Jee_passed)) and np.linalg.norm(Jee_passed) > 1e-12
        Jee_calc = Jee_passed if use_passed_jacobian else Jee_analytic
        try:
            Jee_transpose_inv = np.linalg.inv(Jee_calc.T)
        except np.linalg.LinAlgError:
            Jee_transpose_inv = np.linalg.pinv(Jee_calc.T)

        # Update the position of the end effector in the base frame ----------------------------------------------------

        Pee_Barm=self.updatePeeBarm(arm_joint_pos,self.p0,self.p1,self.p2,self.p3)
        Pee_Brobot=self.update_PeeBrobot(arm_joint_pos,self.RBarm_Brobot,self.PBarm_Brobot,self.Ree_Barm,Pee_Barm.flatten())

        rot_B_to_W = R.from_euler('xyz', base_orientation).as_matrix()

        ###
        
        Pee_Brobot=Pee_Brobot.flatten() 

        # Pee_W_robot = rot_B_to_W@(Pee_Brobot) 
        # Pee_b_r_2=copy.deepcopy(Pee_Brobot)
        # Pee_b_r_2=Pee_b_r_2 
        # print("Pee_b_r_2",Pee_b_r_2)
        skew_matrix = self.skew_mat(Pee_Brobot) #thew skew matrix uses the eef pos in the base of the robot
        # print("skew_matrix",skew_matrix)
        tau_spring=self.updateSpringTorque(arm_joint_pos,self.spring_gains)
        tau_g=self.updateGravityTorque(arm_joint_pos,self.p2,self.p3) #gravity torque

        tau_d=self.updateDampingTorque(arm_joint_vel,self.damping_gains)

        Fee_Barm_est_jacobian = -Jee_transpose_inv @ (tau_spring * self.flag_spring_torque+tau_g*self.flag_gravity_torque + tau_d*self.flag_damping_torque)
        Fee_B_est = self.RBarm_Brobot @ Fee_Barm_est_jacobian

        Fee_W_est = rot_B_to_W @ Fee_B_est

        wrench_B_est = skew_matrix @ Fee_B_est
        wrench_W_est = rot_B_to_W  @ (wrench_B_est)

        # print("wrench_W_est",wrench_W_est)
        # print("Fee_B_est",Fee_B_est)
        # print("Fee_W_est",Fee_W_est)
        # print("gravity torque",tau_g)
        # print("spring torque",tau_spring)
        # # compute magnitude of the wrench
        # print("magnitude of the wrench",np.linalg.norm(wrench_W_est))
        # # magnitude force
        # print("magnitude of the force",np.linalg.norm(Fee_W_est))
        # #magnitude torque gravity and spring
        # print("magnitude of the torque spring",np.linalg.norm(tau_spring))
        # print("magnitude of the torque graivty",np.linalg.norm(tau_g))


        wrench_estimate=np.array([Fee_W_est[0],Fee_W_est[1],Fee_W_est[2],wrench_W_est[0],wrench_W_est[1],wrench_W_est[2]])

        # pass them in base
        # wrench_estimate=np.array([Fee_W_est[0],Fee_W_est[1],Fee_W_est[2],wrench_B_est[0],wrench_B_est[1],wrench_B_est[2]])

        return wrench_estimate, Jee_calc, Pee_Brobot

    def calculate_torque_estimates(self,arm_joint_pos,arm_joint_vel):
        '''
        This function calculates the estimated end-effector force according to section C1 of the paper
        In this case I need to pass the torques directly to my controller, this function will not return the force but the torque
        fee=-(Jee^T)^-1 * (tau_spring+tau_g+tau_d)
        where
        tau_spring is the spring torque
        tau_g is the gravity torque
        tau_d is the damping torque
        Since I need the torques this are given by:
        t_ext = -(Jee^T) @ fee 


        '''



        tau_spring=self.updateSpringTorque(arm_joint_pos,self.spring_gains)
        tau_g=self.updateGravityTorque(arm_joint_pos,self.p2,self.p3) #gravity torque

        tau_d=self.updateDampingTorque(arm_joint_vel,self.damping_gains)

        wrench_estimate = (tau_spring+tau_g + tau_d)
        



        return wrench_estimate
    
    # def calculate_force_estimates(self,Jee,arm_joint_pos,base_orientation):
    #     '''
    #     This function calculates the estimated end-effector force according to section C1 of the paper
    #     fee=-(Jee^T)^-1 * (tau_spring+tau_g+tau_d)
    #     where
    #     tau_spring is the spring torque
    #     tau_g is the gravity torque
    #     tau_d is the damping torque
    #     the value of tau_d and tau_s are given by the spring model of the robot.xml how to get this inmujoco i set all to 0.1
    #     RCF code deos not account the damping matrix it is computed as Fee_Barm_est_jacobian = -arm_model.Jee_transpose_inv*(arm_model.tau_spring + arm_model.tau_g*flag_gravity_torque);


    #     fee is the force in the end effector frame

    #     '''
    #     # Arm Jacobian Calculation --------------------------------------------------------------------------------------

    #     Jee_calc =self.updateJacobian(Jee,arm_joint_pos,self.p1,self.p2,self.p3)
    #     Jee_transpose_inv = np.linalg.inv(Jee_calc.T)

    #     # Update the position of the end effector in the base frame ----------------------------------------------------

    #     Pee_Barm=self.updatePeeBarm(arm_joint_pos,self.p0,self.p1,self.p2,self.p3)
    #     Pee_Brobot=self.update_PeeBrobot(arm_joint_pos,self.RBarm_Brobot,self.PBarm_Brobot,self.Ree_Barm,Pee_Barm.flatten())

    #     rot_B_to_W = R.from_euler('xyz', base_orientation).as_matrix()
    #     # self.rotmat=rot_B_to_W 
    #     ###
        
    #     Pee_Brobot=Pee_Brobot.flatten()

    #     Pee_W_robot = rot_B_to_W@(Pee_Brobot) 

    #     skew_matrix = self.skew_mat(Pee_Brobot) #thew skew matrix uses the eef pos in the base of the robot

    #     tau_spring=self.updateSpringTorque(arm_joint_pos,self.spring_gains)
    #     tau_g=self.updateGravityTorque(arm_joint_pos,self.p2,self.p3) #gravity torque
        

    #     Fee_Barm_est_jacobian = -Jee_transpose_inv@(tau_spring+tau_g*self.flag_gravity_torque)
    #     Fee_B_est = self.RBarm_Brobot@Fee_Barm_est_jacobian

    #     Fee_W_est = rot_B_to_W@Fee_B_est

    #     wrench_B_est = skew_matrix @ Fee_B_est
    #     wrench_W_est = rot_B_to_W@wrench_B_est
    #     # wrench_W_est[1]=0

    #     wrench_estimate=np.array([Fee_W_est[0],Fee_W_est[1],Fee_W_est[2],wrench_W_est[0],wrench_W_est[1],wrench_W_est[2]])

    #     return wrench_estimate

    def compute_reference_velocity(self,arm_joint_pos_curr,previous_ref_vel_x,previous_ref_vel_yaw):
        ''''
        The leader-follower interface commands the follower in velocity according to the following parameters:
        - ref_vel_x: reference velocity in the x direction
        - ref_base_ang_vel: reference angular velocity of the base
        teta_1= 10 degrees
        teta_2= 20 degrees are the longitudinal angles

        sigma_1=10, sigma_2=20 are the one controlling the angular velocity 

        The intervals given p the displacement of joint 3 of the arm with respect to the reference position(gravity vector) are:
        - [-10,10] degrees is the neutral position of the arm so the velocity is 0
        - [-25,-10] degrees the arm is in the blue area, first velocity level and moves forward the velocuty is 0.1 m/s
        - [...,-25] degrees the arm is in the red area, second velocity level and moves forward the velocity is 0.2 m/s
        Similarly for the angular velocity:
        - [-10,10] degrees the arm is in the neutral position so the velocity is 0
        - [-20,-10] or [20,10] degrees the arm is in the blue area, first velocity level the angular velocuty is 0.3 rad/s
        - [...,-20] or [...,20] degrees the arm is in the red area, second velocity level the angular velocity is 0.4 rad/s


        '''
            #compute the velocityself
        g = self.guidance
        if (arm_joint_pos_curr[2] -  self.joint_pos0[2]  > g['theta2'] *np.pi/180.0):
            ref_vel_x=g['vx2_pos']
        elif (arm_joint_pos_curr[2]-self.joint_pos0[2]  > g['theta1'] *np.pi/180.0):
            ref_vel_x=g['vx1']
        elif (arm_joint_pos_curr[2]-self.joint_pos0[2]  < -g['theta2'] *np.pi/180.0):
            ref_vel_x=g['vx2_neg']
        elif (arm_joint_pos_curr[2]-self.joint_pos0[2] < -g['theta1'] *np.pi/180.0):
            ref_vel_x=g['vx1']
        else:
            ref_vel_x=0.0

        # else:
        #     ref_vel_x=config.simulation_params['ref_x_dot']
        #yaw motion
        if (arm_joint_pos_curr[0]-self.joint_pos0[0] > g['psi1'] *np.pi/180.0 and arm_joint_pos_curr[0]-self.joint_pos0[0] < g['psi2']*np.pi/180.0):
            ref_vel_yaw=g['wz1']
        elif (arm_joint_pos_curr[0]-self.joint_pos0[0] < -g['psi1'] *np.pi/180.0 and arm_joint_pos_curr[0]-self.joint_pos0[0] > -g['psi2'] *np.pi/180.0):
            ref_vel_yaw=-g['wz1']

        elif (arm_joint_pos_curr[0]-self.joint_pos0[0] > g['psi2']*np.pi/180.0):
            ref_vel_yaw=g['wz2']

        elif (arm_joint_pos_curr[0]-self.joint_pos0[0] < -g['psi2'] *np.pi/180.0):
            ref_vel_yaw=-g['wz2']

        else:
            ref_vel_yaw=0.0
        # Once the robot starts walking it should keep walking unless the arm is in the neutral position
        # to do so I need a filter to smooth the velocity since the input is non continuous

        ref_vel_x=low_pass_filter(ref_vel_x,previous_ref_vel_x)
        ref_vel_yaw=low_pass_filter(ref_vel_yaw,previous_ref_vel_yaw)
        
        ref_base_lin_vel_follower = np.array([ref_vel_x,0.0,0.0])
        ref_base_ang_vel = np.array([0.0,0.0,ref_vel_yaw])
        return ref_base_lin_vel_follower,ref_base_ang_vel
#########
def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
def low_pass_filter(signal_in,signal_in_prev):
    alpha = 0.5  # Lower alpha provides smoother output
    signal_out = alpha * signal_in + (1 - alpha) * signal_in_prev
    return signal_out

# Function to filter the linear and angular velocities
def filter_state(signal_in, vel_kleader,signal_in_prev,filter_type="mean"):

    # simple mean filter
    if(filter_type == "mean"):
        # signal_out = signal_in[-1]*0.8 + signal_in[-2]*0.2
        signal_out = signal_in[-1]*0.8 + signal_in_prev*0.2

        # signal_out = low_pass_filter(signal_out,0.05)

        
        signal_out=signal_out
        # signal_out = max(signal_out, signal_in_prev)

    elif(filter_type == "savgol"):
        signal_out = savgol_filter(signal_in, window_length=5, polyorder=2)
        # signal_out = signal_out[-1]

    elif(filter_type == "none"):
        signal_out = signal_in[-1]
    elif(filter_type == "butter"):
        signal_out = butter_lowpass_filter(signal_in, 0.1, 100, 2)
        signal_out = signal_out[-1]
        # signal_out = np.maximum(signal_out, 0.1)

    return signal_out


#### LEGACY CODE

'''
    # def force_est_mujoco(self,jacobian,base_pos,base_orientation,position_eef):
    #     Jee_transpose_inv = np.linalg.inv(jacobian.T)
    #     tau_spring= self.updateSpringTorque(position_eef,self.spring_gains)
    #     tau_g=self.tau_g(position_eef,self.mass,9.81,self.p2,self.p3)
    #     flag_gravity_torque=1
    #     Fee_Barm_est_jacobian = -1*Jee_transpose_inv@(tau_spring+tau_g*flag_gravity_torque) #fee
    #     Fee_B_est = Fee_Barm_est_jacobian
    #     #ALL OF THIS IS FOR THE torques
    #     # self.Pee_Barm=self.updatePeeBarm(arm_joint_pos,self.p0,self.p1,self.p2,self.p3)
    #     # print("Pee_Barm",Pee_Barm,"\n","PBarm_Brobot",PBarm_Brobot)
    #     # Pee_Brobot=self.update_PeeBrobot(arm_joint_pos,self.RBarm_Brobot,self.PBarm_Brobot,self.Ree_Barm,self.Pee_Barm.flatten())
    #     rot_B_to_W=self.RPY_to_rot(base_orientation)
    #     # print("rot_B_to_W",rot_B_to_W)
    #     ###
    #     skew_matrix = self.skew_mat(position_eef)
    #     #thw torque is computed as rBw*Pee_Brobot*fee
    #     wrench_B_est = skew_matrix@Fee_B_est
    #     wrench_W_est = wrench_B_est
    #     return wrench_W_est,Fee_B_est

    # def transform_from_world_to_eef(self,point,base_orientation,arm_joint_pos):
    #     #this function transforms a point from the world frame to the eef frame
    #     rot_B_to_W=self.RPY_to_rot(base_orientation)
    #     rot_w_to_B=rot_B_to_W.T
    #     point_B=rot_w_to_B@point
    #     # point_Barm=point_B-self.PBarm_Brobot
    #     Jee =self.updateJacobian(arm_joint_pos,self.p1,self.p2,self.p3)
    #     Jee_transpose_inv = np.linalg.inv(Jee.T)
    #     point_eef=-Jee_transpose_inv@point_B
    #     return point_eef
    # def transform_eef_to_world(self,point,base_orientation,arm_joint_pos):
    #     #this function transforms a point from the world frame to the eef frame
    #     rot_x=self.Rx(100)
    #     rot_B_to_W=self.RPY_to_rot(base_orientation)
    #     rot_w_to_B=rot_B_to_W.T
    #     point_B=rot_w_to_B@point
    #     point_Barm=point_B-self.PBarm_Brobot
    #     Jee =self.updateJacobian(arm_joint_pos,self.p1,self.p2,self.p3)
    #     point_eef=Jee@point_Barm
    #     # point_eef=rot_x@point
    #     return point_eef
    
    
    # the function applyFT from toco applies the force in the geom frame. so the EEF is applied in the geom frame not at the  eef.
    # Since i want to apply the force in the world frame I need to transform the force from the world frame to the geom frame to have the proper oreintation
    # this goes both for plotting and applying the force
    # in mujoco using env.mjData.geom('apply_force_frame') return the xpos and xmat of the geom the x in mujoco is the global coordinates
    
    # def transform_force_from_world_to_geom(self,rot_mat,force):
    #     # this function transforms a force from the world frame to the geom frame
    #     # the geom frame is the frame where the force is applied in mujoco
    #     # the force is applied in the geom frame
    #     #rot mat is the rotation matrix of the geom frame
    #     #pos is the position of the geom frame
    #     #force is the force in the world frame
    #     #the force in the geom frame is given by the following formula
    #     rot_mat=rot_mat.reshape(3,3)
    #     force_geom=rot_mat.T@force
    #     force_geom_2=rot_mat@force
    #     #rot_mat=env.mjData.geom('apply_force_frame').xmat
    #     #pos=env.mjData.geom('apply_force_frame').xpos
    #     return force_geom

    # def compute_arm_distances_world_frame(self,arm_joint_pos,base_pos):
        
    #     base_world=base_pos
    #     p0_world=np.array([base_world[0]-self.p0[0]-0.3+0.12,base_world[1]+self.p0[1],base_world[2]+self.p0[2]+0.05])

    #     p1_world=np.array([p0_world[0]-self.p1[0],p0_world[1]+self.p1[1],p0_world[2]+self.p1[2]])
    #     p2_world=np.array([p1_world[0]-self.p2[0],p1_world[1]+self.p2[1],p1_world[2]+self.p2[2]])
    #     p3_world=np.array([p2_world[0]-self.p3[0],p2_world[1]+self.p3[1],p2_world[2]+self.p3[2]])
        
    #     pee_Barm_w=self.updatePeeBarm(arm_joint_pos,self.p0,self.p1,self.p2,self.p3)
    #     pee_Barm_w=pee_Barm_w.flatten()
    #     pee_world=pee_Barm_w
    #     pee_world_2=np.array([-pee_Barm_w[0]+base_world[0]-0.2,pee_Barm_w[1]+base_world[1],pee_Barm_w[2]+base_world[2]])

    #     p_pos_w=np.array([pee_world[0],pee_world[1],pee_world[2]])
    #     return p_pos_w

'''
