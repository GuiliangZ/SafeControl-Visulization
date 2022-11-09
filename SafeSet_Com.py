import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import sparse
import osqp
import copy
import math
import sys
import collections
import matplotlib.patches as mpatches
from itertools import cycle

import simu_env
import dynamic_obstacle
import ssa
import robot
import ISSA_AdamBA as IA

def safeset_common(s, u, all_obsStates, unsafe_obstacles_c, env, phi_params, max_acc):
    best_action = []
    safeset_com = []
    safeset_com_bound = []
    good_phi_dot_dict_allAction = {}
    safeset_com_bound_num = -1
    optimal_action = u
    limits = [[-max_acc, max_acc]] * 2 
    flag = 0
    plot_flag = 0
    counter = 0
    best_action_phi_dot = 0

    for i, obstacle, obs_color in unsafe_obstacles_c:
        obstacle = obstacle[:4]
        safeset_temp, good_phi_dot_dict = spread_points_safesets(s, u, env,  obstacle, phi_params)
        # print(len(safeset_temp))
        good_phi_dot_dict_allAction.update(good_phi_dot_dict)

        if (not safeset_com) & (flag == 0):
            safeset_com = safeset_temp
            flag = 1
        safeset_temp = [[round(items[0], 6),round(items[1], 6)] for items in safeset_temp]
        safeset_com = [[round(items[0], 6),round(items[1], 6)] for items in safeset_com]
        safeset_com = list(set(tuple(i) for i in safeset_com).intersection(set(tuple(i) for i in safeset_temp)))
        # for action_com in safeset_com:
        #     if action_com not in safeset_temp:
        #         safeset_com.remove(action_com)
        print(f"numbers of common safe controls{len(safeset_com)}")

    for action_com in safeset_com:
        if not outofbound(limits, action_com):
            safeset_com_bound.append(action_com)
    print(f"numbers of common safe controls in the control bound{len(safeset_com_bound)}")
#Pure AdamBA: Just like original pure AdamBA, it chooses the safe control that is closest to the original safe control, which doesn't work well. (Easy to set into local optima)
    # if safeset_com_bound:
    #     norm_list = np.linalg.norm(safeset_com_bound, axis=1)
    #     optimal_action_index = np.where(norm_list == np.amin(norm_list))[0][0]
    #     best_common_action = safeset_com_bound[optimal_action_index]
    # else:  
    #     best_common_action = u
#Modified AdamBA
    phi_col = []
    phi_smallest = 100
    obs_t2_ids_temp = -1 
    if safeset_com_bound:
        for i, action_temp in enumerate(safeset_com_bound):
            local_new_robot = local_step_robot(s, action_temp)
            local_new_obs = local_step_obs(obstacle)
            s_local_new = [local_new_robot.x,local_new_robot.y,local_new_robot.v_x,local_new_robot.v_y]
            s_local_new = np.array(s_local_new)
    #When all obstacles are stationary
            #phi_new = local_phi(s_local_new, obstacle, phi_params)      
    #When all obstacles are dynamic            
            phi_new, obs_t2_temp, obs_t2_ids_temp = local_phi_dynamic(s_local_new, all_obsStates, phi_params)
            phi_col.append(phi_new)
            if phi_smallest > phi_new:
                phi_smallest = phi_new
                optimal_action = action_temp
    else:
        optimal_action = u
    best_common_action = optimal_action
    if best_common_action != u:
        best_action_phi_dot = good_phi_dot_dict_allAction[tuple(best_common_action)]
    else:
        best_action_phi_dot = 0
    print(best_action_phi_dot)
    safeset_com_bound_num = len(safeset_com_bound)
# New methods to choose the best phi - maybe by looking ahead couple steps. -- look into Honeyi's paper, and see if we can use the gap method
    return safeset_com, best_common_action, safeset_com_bound_num, best_action_phi_dot

def spread_points_safesets(s, u, env, obstacle, phi_params, ctrlrange=0.008, max_tral_num=1): #(limit is robot maximum speed(ctrlrange))
    infSet = []
    NP_vec = []
    NP_vec_tmp = []
    phi_temp_all = []
    good_phi_temp = []
    phi_dot_temp = []
    good_phi_dot_temp = []
    NP_vec_tmp_new = []
    good_phi_dot_dict = {}
    valid = 0
    cnt = 0
    out = 0
    yes = 0
    dt = 1

    point_num = 3000

    u = np.clip(u, -ctrlrange, ctrlrange)
    action_space_num = 2
    action = np.array(u).reshape(-1, action_space_num)
    limits = [[-ctrlrange, ctrlrange]] * action_space_num  # each row define the limits for one dimensional action
    NP = action
    max_trials = max_tral_num
    row_num = math.floor(np.sqrt(point_num))
    x_temp = -ctrlrange
    y_temp = -ctrlrange
    for t in range(0, row_num+1):
        for m in range(0, row_num+1):
            NP_vec.append([x_temp, y_temp])
            y_temp += ctrlrange*2/row_num
        x_temp += ctrlrange*2/row_num
        y_temp = -ctrlrange
    for n in range(0, NP.shape[0]):
        trial_num = 0
        at_least_1 = False
        while trial_num < max_trials and not at_least_1:
            at_least_1 = False
            trial_num += 1
            for v in range(0, len(NP_vec)):
                NP_vec_tmp_i = NP_vec[v]
                flag, phi_new, phi_dot = chk_unsafe(s, NP_vec_tmp_i, env, dt, obstacle, phi_params)
                if flag == 0:
                    NP_vec_tmp.append(NP_vec_tmp_i)
                    phi_temp_all.append(phi_new)
                    phi_dot_temp.append(phi_dot)
        NP_vec_tmp = [[round(items[0], 6),round(items[1], 6)] for items in NP_vec_tmp]
        for vnum in range(0, len(NP_vec_tmp)):
            cnt += 1
            if IA.outofbound(limits, NP_vec_tmp[vnum]):
                out += 1
                continue
            if NP_vec_tmp[vnum][0] == u[0] and NP_vec_tmp[vnum][1] == u[1]:
                yes += 1
                continue
            valid += 1
            NP_vec_tmp_new.append(NP_vec_tmp[vnum]) #This is the valid sets doesn't consider control bound
            good_phi_temp.append(phi_temp_all[vnum])
            good_phi_dot_temp.append(phi_dot_temp[vnum])
            good_phi_dot_dict[tuple(NP_vec_tmp[vnum])] = phi_dot_temp[vnum]
    return NP_vec_tmp_new, good_phi_dot_dict

def chk_unsafe(s, point, env, dt, obstacle, phi_params):
    dmin = 0.12
    action = [point[0], point[1]]
    local_new_robot = local_step_robot(s, action)
    local_new_obs = local_step_obs(obstacle)
    s_local_new = [local_new_robot.x,local_new_robot.y,local_new_robot.v_x,local_new_robot.v_y]
    s_local_new = np.array(s_local_new)
#Use safety index phi to judge whether that action is safe or not
    phi_old = local_phi(s, obstacle, phi_params)
    phi_new = local_phi(s_local_new, obstacle, phi_params)
#preview the obstacle location to calculate the new phi and check if it is safe 
    #phi_new = local_phi(s_local_new, obstacle, phi_params)
    phi_dot = phi_new - phi_old
    if phi_old > 0:
        if phi_dot < 0:
            flag = 0  # safe
        else:
            flag = 1  # unsafe
    else:
        flag = 0
    return flag, phi_new, phi_dot

def local_step_robot(s, action):
    local_robot = robot.DoubleIntegratorRobot(s[0],s[1],s[2],s[3],0.03)
    s_local_new = local_robot.steer(action[0],action[1])
    return s_local_new

def local_phi(robot_state, obstacle, phi_params, dmin=0.12):
    alpha = phi_params[0]
    n = phi_params[1]
    k = phi_params[2]
    d = np.array(robot_state - obstacle)
    d_pos = d[:2] # pos distance
    d_vel = d[2:] # vel 
    d_abs = np.linalg.norm(d_pos)
    d_dot = (d_pos @ d_vel.T) / np.linalg.norm(d_pos)
    phi = alpha + np.power(dmin, n) - np.power(np.linalg.norm(d_pos), n) - k * d_dot
    return phi

def local_phi_0(s_local_new, local_new_obs):
    dmin = 0.12
    d = np.array(s_local_new - local_new_obs)
    d_pos = d[:2] # pos distance
    d_vel = d[2:] # vel 
    d_abs = np.linalg.norm(d_pos)
    d_dot = (d_pos @ d_vel.T) / np.linalg.norm(d_pos)
    phi_0 = dmin - d_abs
    return phi_0

def local_phi_dynamic(robot_state, obs_states, phi_params):
    phi_collect = []
    phi_obs_states = {}
    phi_obs_ids = {}
    alpha = phi_params[0]
    n = phi_params[1]
    k = phi_params[2]
    dmin = 0.12
    for i, obs_state in enumerate(obs_states):
        new_obs_state = local_step_obs(obs_state[:4])
        d = np.array(robot_state - new_obs_state)
        d_pos = d[:2] # pos distance
        d_vel = d[2:] # vel 
        d_abs = np.linalg.norm(d_pos)
        d_dot = (d_pos @ d_vel.T) / np.linalg.norm(d_pos)
        phi = alpha + np.power(dmin, n) - np.power(np.linalg.norm(d_pos), n) - k * d_dot 
        phi_collect.append(phi)
        phi_obs_ids[phi] = i 
        phi_obs_states[phi] = obs_state
    phi_temp = phi_collect[0]
#find the largest phi - the most dangerous obstacle case
    if len(phi_collect)>1:
        for i in phi_collect:
            if i > phi_temp:
                phi_temp = i
    obs_t2_state_temp = phi_obs_states[phi_temp]
    obs_t2_ids_temp = phi_obs_ids[phi_temp]         
    return phi_temp, obs_t2_state_temp, obs_t2_ids_temp

def outofbound(limit, p):
    flag = 0
    assert len(limit[0]) == 2
    for i in range(len(limit)):
        assert limit[i][1] > limit[i][0]
        if p[i] < limit[i][0] or p[i] > limit[i][1]:
            flag = 1
            break
    return flag

def local_step_obs(obstalce):
#assuming to be constant velocity for obtacle, cause doesn't know the accerleation
    #a is acceleration, b is velocity, c is position
    obstacle_new = []
    t_step = 1
    obstacle_new.append(obstalce[0] + obstalce[2] * t_step)
    obstacle_new.append(obstalce[1] + obstalce[3] * t_step)
#Since we assume we don't know the accerelation, so can't update precisely of the obstacle state
    obstacle_new.append(obstalce[2])
    obstacle_new.append(obstalce[3])
    return obstacle_new

def plot_Obs_SafeComSet(s, u, unsafe_obstacles_c, all_obsStates, env, phi_params):      
    plot_flag = 0
    counter = 0
    safeset_com = []
    safeset_com_bound = []
    flag = 0
    for i, obstacle, obs_color in unsafe_obstacles_c:
        obstacle = obstacle[:4]
        safeset_temp, good_phi_dot_dict = spread_points_safesets(s, u, env,  obstacle, phi_params)

        if (not safeset_com) & (flag == 0):
            safeset_com = safeset_temp
            flag = 1
        safeset_temp = [[round(items[0], 6),round(items[1], 6)] for items in safeset_temp]
        safeset_com = [[round(items[0], 6),round(items[1], 6)] for items in safeset_com]
        safeset_com = list(set(tuple(i) for i in safeset_com).intersection(set(tuple(i) for i in safeset_temp)))

        if counter == len(all_obsStates)-1:
            plot_flag = 1
        x_safeset_temp = []
        y_safeset_temp = []
        legend_num_alpha = phi_params[0]
        legend_num_n = phi_params[1]
        legend_num_k = phi_params[2]
        for i,action_temp in enumerate(safeset_temp):
            x_safeset_temp.append(action_temp[0]) 
            y_safeset_temp.append(action_temp[1])
        rect = mpatches.Rectangle((-env.max_acc,-env.max_acc),env.max_acc*2,env.max_acc*2, 
                fill = True, alpha = 0.2, color = "green")  
        plt.gca().add_patch(rect)
        point_numbers = np.array(range(len(x_safeset_temp)))
        plt.scatter(x_safeset_temp,y_safeset_temp, color=obs_color, alpha = 0.5) 
        if plot_flag:
            x_safeset_com = []
            y_safeset_com = []
            for i,action_temp in enumerate(safeset_com):
                x_safeset_com.append(action_temp[0]) 
                y_safeset_com.append(action_temp[1])
            plt.scatter(x_safeset_com,y_safeset_com, color='red', alpha = 1)
            plt.title(f'ControlSpace Phi_Params:alpha ={legend_num_alpha},n ={legend_num_n},k ={legend_num_k}')
            plt.draw()
            plt.pause(1)
            return
        plt.title(f'ControlSpace Phi_Params:alpha ={legend_num_alpha},n ={legend_num_n},k ={legend_num_k}')
        # plt.show()
        plt.draw()
        plt.pause(1)

        counter += 1

