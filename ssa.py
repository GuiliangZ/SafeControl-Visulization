from glob import escape
from lib2to3.refactor import MultiprocessRefactoringTool
import numpy as np
import math
import cvxopt
import sys
import collections
#zgl
import ISSA_AdamBA
import evaluator
from cma_es import SafeLearning
import yaml


  
class SafeSetAlgorithm():
    def __init__(self, max_speed, is_qp = False, dmin = 0.12, k = 1, max_acc = 0.04):
        """
        Args:
            dmin: dmin for phi
            k: k for d_dot in phi
        """
        self.dmin = dmin
        self.k = k
        self.max_speed = max_speed
        self.max_acc = max_acc
        self.forecast_step = 3
        self.records = collections.deque(maxlen = 10)
        self.acc_reward_normal_ssa = 0
        self.acc_reward_qp_ssa = 0
        self.acc_phi_dot_ssa = 0
        self.acc_phi_dot_qp = 0
        self.is_qp = is_qp

    def get_safe_control(self, robot_state, obs_states, f, g, u0, phi_params, unsafe_obstacle_ids):
        """
        Args:
            robot_state <x, y, vx, vy>
            obs_state: np array closest static obstacle state <x, y, vx, vy, ax, ay>
        """
        u0 = np.array(u0).reshape((2,1))
        robot_vel = np.linalg.norm(robot_state[-2:])
        
        L_gs = []
        L_fs = []
        obs_dots = []
        reference_control_laws = []
        is_safe = True
        constrain_obs = []
        x_parameter = 0.
        y_parameter = 0.
        phis = []


#Newly added parameters
        phis_unsafe = []

        phi_temp = 0
        phi = 0
#adjust this parameter
        d_info = {}


        warning_indexs = []
        danger_indexs = [] 
        danger_obs = []
        record_data = {}
        record_data['obs_states'] = [obs[:2] for obs in obs_states]
        record_data['robot_state'] = robot_state
        record_data['phi'] = []
        record_data['CMAES_phi'] = []
        record_data['phi_dot'] = []
        record_data['is_safe_control'] = False
        record_data['is_multi_obstacles'] = True if len(obs_states) > 1 else False

        phis_temp = []
        most_dangerObs_state = {}

        MO_SIndex_sum = 0
        MO_SIndex = 0
        MO_Count = 0
        MO_Unsafe_Obs_Consider = 3

        for i, obs_state in enumerate(obs_states):
            obs_cur_state = obs_state[:4]
            d = np.array(robot_state - obs_cur_state)
            d_pos = d[:2] # pos distance
            d_vel = d[2:] # vel 
            d_abs = np.linalg.norm(d_pos)
            d_dot = self.k * (d_pos @ d_vel.T) / np.linalg.norm(d_pos)

            alpha = phi_params[0]
            n = phi_params[1]
            k = phi_params[2]






#add CMA-ES evaluator to optimize three parameters
#For all the unsafe obstacles, use CMA-ES get parameters to get the smallest phi 

            phi = alpha + np.power(self.dmin, n) - np.power(np.linalg.norm(d_pos), n) - k * d_dot
            # if (phi>0):
            #     phis_temp.append(phi)
            #     d_info[phi] = [d_pos, d_dot]
            #     print(d_info[phi][0])
            #     print(d_info[phi][1])

            # print('Before CMAES:')
            # print(phi) 

# #implement CMA-ES to optimize three parameters of phi
#             with open('SafetyIndex_config.yaml', 'r') as stream:
#                 try:
#                     config = yaml.safe_load(stream)
#                     config["evaluator"] = evaluator.SafetyIndexEvaluator(self.dmin, d_pos, d_dot)
#                     learner = SafeLearning(config,self.dmin, d_pos, d_dot)
#                     alpha, n, k = learner.learn()
#                 except yaml.YAMLError as exc:
#                     print(exc)

#             #alpha, n, k = SafeLearning(self.dmin, d_pos, d_dot)


            # CMAES_phi = alpha + np.power(self.dmin, n) - np.power(np.linalg.norm(d_pos), n) - k * d_dot 
            # print('After CMAES:')
            # print(CMAES_phi) 
#ZGL: Safety index rule (dmin^2 - d^2 - d_dot )  might modify            
            #phi = np.power(self.dmin, 2) - np.power(np.linalg.norm(d_pos), 2) - d_dot
            #print(f"phi {phi}, obs_states {obs_states}")
            record_data['phi'].append(phi)
            # record_data['CMAES_phi'].append(CMAES_phi)









#ZGL: Not sure what's going on here
            # calculate Lie derivative
            # p d to p robot state and p obstacle state
            p_d_p_robot_state = np.hstack([np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_d_p_obs_state = np.hstack([-1*np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_d_pos_p_d = np.array([d_pos[0], d_pos[1]]).reshape((1,2)) / d_abs # shape (1, 2)
            p_d_pos_p_robot_state = p_d_pos_p_d @ p_d_p_robot_state # shape (1, 4)
            p_d_pos_p_obs_state = p_d_pos_p_d @ p_d_p_obs_state # shape (1, 4)

            # p d_dot to p robot state and p obstacle state
            p_vel_p_robot_state = np.hstack([np.zeros((2,2)), np.eye(2)]) # shape (2, 4)
            p_vel_p_obs_state = np.hstack([np.zeros((2,2)), -1*np.eye(2)]) # shape (2, 4)
            p_d_dot_p_vel = d_pos.reshape((1,2)) / d_abs # shape (1, 2)
            
            p_pos_p_robot_state = np.hstack([np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_pos_p_obs_state = np.hstack([-1*np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_d_dot_p_pos = d_vel / d_abs - 0.5 * (d_pos @ d_vel.T) * d_pos / np.power(d_abs, 3) 
            p_d_dot_p_pos = p_d_dot_p_pos.reshape((1,2)) # shape (1, 2)

            p_d_dot_p_robot_state = p_d_dot_p_pos @ p_pos_p_robot_state + p_d_dot_p_vel @ p_vel_p_robot_state # shape (1, 4)
            p_d_dot_p_obs_state = p_d_dot_p_pos @ p_pos_p_obs_state + p_d_dot_p_vel @ p_vel_p_obs_state # shape (1, 4)

            p_phi_p_robot_state = -2 * np.linalg.norm(d_pos) * p_d_pos_p_robot_state - \
                            self.k * p_d_dot_p_robot_state # shape (1, 4)
            p_phi_p_obs_state = -2 * np.linalg.norm(d_pos) * p_d_pos_p_obs_state - \
                            self.k * p_d_dot_p_obs_state # shape (1, 4)
        
            L_f = p_phi_p_robot_state @ (f @ robot_state.reshape((-1,1))) # shape (1, 1)
            L_g = p_phi_p_robot_state @ g # shape (1, 2) g contains x information
            obs_dot = p_phi_p_obs_state @ obs_state[-4:]
            L_fs.append(L_f)
            phis.append(phi)  
            obs_dots.append(obs_dot)




#ZGL:
#change here to adamBA and ISSA
#(current AdamBA takes in 3 states(x,y,theta), 2 controls(ax,ay) )


#Maybe the goal is to generate a comparison graph of the original method to generate safe control u 
# to the current method to generate u using ISSA+AdamBA

#Next step is to modify the way ISSA is design safety index phi(modify above line)

#Other frame work all the same

#When encounter unsafe scenario, SSA is triggered. Modify

            if (phi > 0):
                L_gs.append(L_g)                                              
                reference_control_laws.append( -0.5*phi - L_f - obs_dot)
                is_safe = False
                danger_indexs.append(i)
                danger_obs.append(obs_state[:2])
                # constrain_obs.append(obs_state[:2])
                phis_unsafe.append(phi)
                most_dangerObs_state[phi] =  [unsafe_obstacle_ids[i], obs_state[:4]]


#Found the new u by solving the QP problem
        if (not is_safe):
            # Solve safe optimization problem
            # min_x (1/2 * x^T * Q * x) + (f^T * x)   s.t. Ax <= b
            u0 = u0.reshape(-1,1)
            qp_parameter = self.find_qp(robot_state, obs_states, u0.flatten())
            u, reference_control_laws = self.solve_qp(robot_state, u0, L_gs, reference_control_laws, phis, qp_parameter, danger_indexs, warning_indexs)
            reward_qp_ssa = robot_state[1] + (robot_state[3] + u[1]) + 1
            self.acc_reward_qp_ssa += reward_qp_ssa
            
            phi_dots = []
            phi_dots_vanilla = []
            unavoid_collision = False
            
            '''
            for i in range(len(L_gs)):
                phi_dot = L_fs[i] + L_gs[i] @ u + obs_dots[i]
                phi_dot_vanilla = L_fs[i] + L_gs[i] @ u_vanilla + obs_dot
                phi_dots.append(phi_dot)
                phi_dots_vanilla.append(phi_dot_vanilla)
                record_data['phi_dot'].append(phi_dot)
                if (phi_dot > 0 or (phis[i] + phi_dot) > 0):
                    unavoid_collision = True
            '''
            record_data['control'] = u
            record_data['is_safe_control'] = True
            #self.records.append(record_data)
#find the largest phi in the unsafe phis

            phi_temp = phis_unsafe[0]



#Considering Multiple obstacles, Index = (w1*phi1+w2*phi2+w3*phi3...)/number of unsafe obstacles

            #print(phis_unsafe)

            phis_unsafe = np.sort(phis_unsafe)
            phis_unsafe = phis_unsafe[::-1]
            #print(phis_unsafe)
            MO_weights = []
            if len(phis_unsafe)>1:
                for i in phis_unsafe:
                    if MO_Count < MO_Unsafe_Obs_Consider:
                        MO_SIndex_sum += i
                        MO_Count += 1
                MO_Count = 0
                for i in phis_unsafe:
                    if MO_Count < MO_Unsafe_Obs_Consider:
                        MO_weights.append(i/MO_SIndex_sum)
                        MO_Count += 1
                for i in range(MO_Count):
                    MO_SIndex += MO_weights[i]*phis_unsafe[i]
            else:
                MO_SIndex = phis_unsafe[0]



            if len(phis_unsafe)>1:
                for i in phis_unsafe:
                    if i>phi_temp:
                        phi_temp = i
            return u, True, unavoid_collision, danger_obs, phi_temp, most_dangerObs_state[phi_temp][1], most_dangerObs_state[phi_temp][0], MO_SIndex                            

        u0 = u0.reshape(1,2)
        u = u0
        record_data['control'] = u[0]
        self.records.append(record_data)  
        phi_temp = phi
        most_dangerObs_ids = 0
        MO_SIndex = phi
#In the end of day, it return a new control u[0]   
        return u[0], False, False, danger_obs, phi_temp, most_dangerObs_state, most_dangerObs_ids, MO_SIndex    






#Solve for QP Problem(procedure)
    def solve_qp(self, robot_state, u0, L_gs, reference_control_laws, phis, qp_parameter, danger_indexs, warning_indexs):
        q = qp_parameter
        Q = cvxopt.matrix(q)
        u_prime = -u0
        u_prime = qp_parameter @ u_prime
        p = cvxopt.matrix(u_prime) #-u0
        G = cvxopt.matrix(np.vstack([np.eye(2), -np.eye(2), np.array([[1,0],[-1,0]]), np.array([[0,1],[0,-1]])]))
        S_saturated = cvxopt.matrix(np.array([self.max_acc, self.max_acc, self.max_acc, self.max_acc, \
                                    self.max_speed-robot_state[2], self.max_speed+robot_state[2], \
                                    self.max_speed-robot_state[3], self.max_speed+robot_state[3]]).reshape(-1, 1))
        #G = cvxopt.matrix(np.vstack([np.eye(2), -np.eye(2)]))
        #S_saturated = cvxopt.matrix(np.array([self.max_acc, self.max_acc, self.max_acc, self.max_acc]).reshape(-1, 1))
        L_gs = np.array(L_gs).reshape(-1, 2)
        reference_control_laws = np.array(reference_control_laws).reshape(-1,1)
        A = cvxopt.matrix([[cvxopt.matrix(L_gs), G]])
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = 600
        while True:
            try:
                b = cvxopt.matrix([[cvxopt.matrix(reference_control_laws), S_saturated]])
                sol = cvxopt.solvers.qp(Q, p, A, b)
                u = sol["x"]
                break
            except ValueError:
                # no solution, relax the constraint   
                is_danger = False                 
                for i in range(len(reference_control_laws)):
                    if (self.is_qp and i in danger_indexs):
                        reference_control_laws[i][0] += 0.01
                        if (reference_control_laws[i][0] + phis[i] > 0):
                            is_danger = True
                    else:
                        reference_control_laws[i][0] += 0.01
                '''
                if (is_danger and self.is_qp):
                    for i in range(len(reference_control_laws)):
                        if (i in warning_indexs):
                            reference_control_laws[i][0] += 0.01
                '''
                #print(f"relax reference_control_law, reference_control_laws {reference_control_laws}")
        u = np.array([u[0], u[1]])
        return u, reference_control_laws

    def find_qp(self, robot_state, obs_states, u0, safest = False):
        if (not self.is_qp):
            return np.eye(2)
        # estimate obstacle positions in next few steps
        obs_poses = []
        for i in range(self.forecast_step):
            for obs in obs_states:
                obs_poses.append([obs[0]+i*obs[2]-robot_state[0], obs[1]+i*obs[3]-robot_state[1]])

        eigenvectors, max_dis_theta, min_dis_theta = self.find_eigenvector(robot_state, obs_poses)
        eigenvalues = self.find_eigenvalue(obs_poses, max_dis_theta, min_dis_theta)
        R = np.array([eigenvectors[0],eigenvectors[1]]).T
        R_inv = np.linalg.pinv(R)
        Omega = np.array([[eigenvalues[0], 0], [0, eigenvalues[1]]])
        qp = R @ Omega @ R_inv
        return qp

    def find_eigenvector(self, robot_state, obs_poses):
        xs = np.array([pos[0] for pos in obs_poses])
        ys = np.array([pos[1] for pos in obs_poses])

        theta1 = 0.5*np.arctan2(2*np.dot(xs,ys), np.sum(xs**2-ys**2))
        theta2 = theta1+np.pi/2

        first_order_theta1 = 0.5*np.sin(2*theta1)*np.sum(xs**2-ys**2) - np.cos(2*theta1)*np.dot(xs,ys)
        first_order_theta2 = 0.5*np.sin(2*theta2)*np.sum(xs**2-ys**2) - np.cos(2*theta2)*np.dot(xs,ys)

        second_order_theta1 = np.cos(2*theta1)*np.sum(xs**2-ys**2) + 2*np.sin(2*theta1)*np.dot(xs,ys)
        second_order_theta2 = np.cos(2*theta2)*np.sum(xs**2-ys**2) + 2*np.sin(2*theta2)*np.dot(xs,ys)
        
        if (second_order_theta1 < 0):            
            max_dis_theta = theta1
            min_dis_theta = theta2
        else:
            max_dis_theta = theta2
            min_dis_theta = theta1
        lambda1 = [np.cos(max_dis_theta), np.sin(max_dis_theta)]
        lambda2 = [np.cos(min_dis_theta), np.sin(min_dis_theta)]
        return [lambda1, lambda2], max_dis_theta, min_dis_theta

    def find_eigenvalue(self, obs_poses, max_dis_theta, min_dis_theta):
        max_dis = 0.
        min_dis = 0.

        xs = np.array([pos[0] for pos in obs_poses])
        ys = np.array([pos[1] for pos in obs_poses])

        for x, y in zip(xs, ys):
            max_dis += (-np.sin(max_dis_theta)*x + np.cos(max_dis_theta)*y)**2
            min_dis += (-np.sin(min_dis_theta)*x + np.cos(min_dis_theta)*y)**2
        return [min_dis*1e5, max_dis*1e5]
















    def plot_control_subspace(self, robot_state, obs_states, f, g, u0):
        """
        Args:
            robot_state <x, y, vx, vy>
            obs_state: np array closest static obstacle state <x, y, vx, vy, ax, ay>
        """
        u0 = np.array(u0).reshape((2,1))
        robot_vel = np.linalg.norm(robot_state[-2:])
        L_gs = []
        L_fs = []
        obs_dots = []
        reference_control_laws = []
        is_safe = True
        phis = []
        danger_indexs = []
        for i, obs_state in enumerate(obs_states):
            d = np.array(robot_state - obs_state[:4])
            d_pos = d[:2] # pos distance
            d_vel = d[2:] # vel 
            d_abs = np.linalg.norm(d_pos)
            d_dot = self.k * (d_pos @ d_vel.T) / np.linalg.norm(d_pos)
            phi = np.power(self.dmin, 2) - np.power(np.linalg.norm(d_pos), 2) - d_dot
            
            # calculate Lie derivative
            # p d to p robot state and p obstacle state
            p_d_p_robot_state = np.hstack([np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_d_p_obs_state = np.hstack([-1*np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_d_pos_p_d = np.array([d_pos[0], d_pos[1]]).reshape((1,2)) / d_abs # shape (1, 2)
            p_d_pos_p_robot_state = p_d_pos_p_d @ p_d_p_robot_state # shape (1, 4)
            p_d_pos_p_obs_state = p_d_pos_p_d @ p_d_p_obs_state # shape (1, 4)

            # p d_dot to p robot state and p obstacle state
            p_vel_p_robot_state = np.hstack([np.zeros((2,2)), np.eye(2)]) # shape (2, 4)
            p_vel_p_obs_state = np.hstack([np.zeros((2,2)), -1*np.eye(2)]) # shape (2, 4)
            p_d_dot_p_vel = d_pos.reshape((1,2)) / d_abs # shape (1, 2)
            
            p_pos_p_robot_state = np.hstack([np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_pos_p_obs_state = np.hstack([-1*np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_d_dot_p_pos = d_vel / d_abs - 0.5 * (d_pos @ d_vel.T) * d_pos / np.power(d_abs, 3) 
            p_d_dot_p_pos = p_d_dot_p_pos.reshape((1,2)) # shape (1, 2)

            p_d_dot_p_robot_state = p_d_dot_p_pos @ p_pos_p_robot_state + p_d_dot_p_vel @ p_vel_p_robot_state # shape (1, 4)
            p_d_dot_p_obs_state = p_d_dot_p_pos @ p_pos_p_obs_state + p_d_dot_p_vel @ p_vel_p_obs_state # shape (1, 4)

            p_phi_p_robot_state = -2 * np.linalg.norm(d_pos) * p_d_pos_p_robot_state - \
                            self.k * p_d_dot_p_robot_state # shape (1, 4)
            p_phi_p_obs_state = -2 * np.linalg.norm(d_pos) * p_d_pos_p_obs_state - \
                            self.k * p_d_dot_p_obs_state # shape (1, 4)
        
            L_f = p_phi_p_robot_state @ (f @ robot_state.reshape((-1,1))) # shape (1, 1)
            L_g = p_phi_p_robot_state @ g # shape (1, 2) g contains x information
            obs_dot = p_phi_p_obs_state @ obs_state[-4:]
            L_fs.append(L_f)
            phis.append(phi)  
            obs_dots.append(obs_dot)

            if (phi > 0):
                L_gs.append(L_g)                                              
                reference_control_laws.append( -0.5*phi - L_f - obs_dot)
                is_safe = False
                danger_indexs.append(i)

        if (not is_safe):
            u0 = u0.reshape(-1,1)
            qp_parameter = np.eye(2)
            for i in range(len(L_gs)):
                u, _ = self.solve_qp(robot_state, u0, L_gs[i], reference_control_laws[i], [], qp_parameter, [], [])
                print(u, L_gs[i])




    def check_same_direction(self, pcontrol, perpendicular_controls):
        if (len(perpendicular_controls) == 0):
            return True
        for control in perpendicular_controls:
            angle = self.calcu_angle(pcontrol, control)
            if (angle > np.pi/4):
                return False
        return True

    def calcu_angle(self, v1, v2):
        lv1 = np.sqrt(np.dot(v1, v1))
        lv2 = np.sqrt(np.dot(v2, v2))
        angle = np.dot(v1, v2) / (lv1*lv2)
        return np.arccos(angle)
