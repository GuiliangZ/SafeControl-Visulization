
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import copy
import numpy as np
import math
import robot

  
 

def AdamBA(s, u, env, threshold, phi_old, 
        obstacle, all_obsStates, phi_params, unsafe_obstacle_ids, most_dangerObs_ids,ctrlrange=0.03, max_tral_num=1, vec_num=None): #(limit is robot maximum speed)
    infSet = []
    # generate direction
    NP_vec_dir = []
    NP_vec = []
    phi_temp_all = []
    good_phi_temp = []
    phi_dot_temp = []
    good_phi_dot_temp = []    
    bound = 0.0001
    # record how many boundary points have been found
    collected_num = 0
    valid = 0
    cnt = 0
    out = 0
    yes = 0
    dt = 1
    sigma_vec =[[1, 0], [0, 1]]
    vec_num = 500
    loc = 0
    scale = 0.1
#can change this to number of actions(controls)
    action_space_num = 2
    action = np.array(u).reshape(-1, action_space_num)
    u = np.clip(u, -ctrlrange, ctrlrange)
    np.random.seed(0)
    limits = [[-ctrlrange, ctrlrange]] * action_space_num  # each row define the limits for one dimensional action
    NP = action
    start_time = time.time()

    if action_space_num ==2:
        vec_num = 10 if vec_num == None else vec_num
    elif action_space_num == 12:
        vec_num = 20 if vec_num == None else vec_num
    else:
        raise NotImplementedError

    # num of actions input, default as 1
    for t in range(0, NP.shape[0]):
        if action_space_num == 2:
            vec_set = []
            vec_dir_set = []
            for m in range(0, vec_num):
                # vec_dir = np.random.multivariate_normal(mean=[0, 0], cov=sigma_vec)
                # vec_dir = vec_dir / np.linalg.norm(vec_dir)
                theta_m = m * (2 * np.pi / vec_num)
                vec_dir = np.array([np.sin(theta_m), np.cos(theta_m)]) / 2
                vec_dir_set.append(vec_dir)
                vec = NP[t]
                vec_set.append(vec)
            NP_vec_dir.append(vec_dir_set)
            NP_vec.append(vec_set)
        else:
            vec_dir_set = np.random.normal(loc=loc, scale=scale, size=[vec_num, action_space_num])
            vec_set = [NP[t]] * vec_num
            #import ipdb; ipdb.set_trace()
            NP_vec_dir.append(vec_dir_set)
            NP_vec.append(vec_set)

    max_trials = max_tral_num
    for n in range(0, NP.shape[0]):
        trial_num = 0
        at_least_1 = False
        while trial_num < max_trials and not at_least_1:
            at_least_1 = False
            trial_num += 1
            NP_vec_tmp = copy.deepcopy(NP_vec[n])
            # print(NP_vec)
            if trial_num ==1:
                NP_vec_dir_tmp = NP_vec_dir[n]
            else:
                NP_vec_dir_tmp = np.random.normal(loc=loc, scale=scale, size=[vec_num, action_space_num])
            for v in range(0, vec_num):
                NP_vec_tmp_i = NP_vec_tmp[v]
                NP_vec_dir_tmp_i = NP_vec_dir_tmp[v]
                eta = bound
                decrease_flag = 0
                while True:
                    flag, phi_new, phi_dot = chk_unsafe(s, NP_vec_tmp_i, env, dt, threshold, phi_old, obstacle, all_obsStates, phi_params)
                    # safety gym env itself has clip operation inside
                    if outofbound(limits, NP_vec_tmp_i):
                        # print("\nout\n")
                        #collected_num = collected_num - 1  # not found, discard the recorded number
                        break                   
                    if eta <= bound and flag == 0:
                        at_least_1 = True
                        break
                    # AdamBA procudure
                    if flag == 1 and decrease_flag == 0:
#unsafe control, keep strech out to find the boundary
                        # outreach
                        NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                        eta = eta * 2
                        continue
                    # monitor for 1st reaching out boundary
                    if flag == 0 and decrease_flag == 0:
#safe control found, but the vector is streching, need to set it to decay
                        decrease_flag = 1
                        eta = eta * 0.25  # make sure decrease step start at 0.5 of last increasing step
                        continue
                    # decrease eta
                    if flag == 1 and decrease_flag == 1 :
#the vector has gone too far decaying that exceed the safe control boundary, need to going streching again to land it inside the boundrary
                        NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                        eta = eta * 0.5
                        continue
                    if flag == 0 and decrease_flag == 1 :
#found the safe control set, now going back to find boundary
                        NP_vec_tmp_i = NP_vec_tmp_i - eta * NP_vec_dir_tmp_i
                        eta = eta * 0.5
                        continue
                NP_vec_tmp[v] = NP_vec_tmp_i
                phi_temp_all.append(phi_new)
                phi_dot_temp.append(phi_dot)
        #print(phi_temp)
        NP_vec_tmp_new = []
        for vnum in range(0, len(NP_vec_tmp)):
            cnt += 1
            if outofbound(limits, NP_vec_tmp[vnum]):
                # print("out")
                out += 1
                continue
            if NP_vec_tmp[vnum][0] == u[0] and NP_vec_tmp[vnum][1] == u[1]:
                # print("yes")
                yes += 1
                continue
            valid += 1
            NP_vec_tmp_new.append(NP_vec_tmp[vnum]) #This is the valid sets
            good_phi_temp.append(phi_temp_all[vnum])
            good_phi_dot_temp.append(phi_dot_temp[vnum])
        NP_vec[n] = NP_vec_tmp_new

    end_time = time.time()
    # start to get the A and B for the plane
    if valid > 0:
        valid_adamba = "adamba success"
    elif valid == 0 and yes==vec_num:
        valid_adamba = "itself satisfy"
    elif valid == 0 and out==vec_num:
        valid_adamba = "all out"
    else:
        valid_adamba = "exception"
    if len(NP_vec_tmp_new) > 0:  # at least we have one sampled action satisfying the safety index 
#want to use the action with the smallest phi instead of using the action that is closest to the original action
#Pure AdamBA
        norm_list = np.linalg.norm(NP_vec_tmp_new, axis=1)
        # optimal_action_index = np.where(norm_list == np.amin(norm_list))[0][0]
        # print("out = ", out)
        # print("valid = ", valid)
        # phi_best = good_phi_temp[optimal_action_index]
        # action_best = NP_vec_tmp_new[optimal_action_index]
        # print(action_best)
        # print(phi_best)
        # return action_best, valid_adamba, NP_vec_tmp, out, yes, valid, phi_best, phi_best    
#Modified AdamBA
        phi_col = []
        phi_smallest = 100
        obs_t2_ids_temp = -1        
        for i, action_temp in enumerate(NP_vec_tmp_new):
            local_new_robot = local_step(s, action_temp)
            s_local_new = [local_new_robot.x,local_new_robot.y,local_new_robot.v_x,local_new_robot.v_y]
            s_local_new = np.array(s_local_new)
            #s_new, reward, done, info = env.step(action, is_safe, unsafe_obstacle_ids) 
#When all obstacles are stationary
            phi_new = local_phi_stationary(s_local_new, obstacle, phi_params)
#When all obstacles are dynamic            
            #phi_new, obs_t2_temp, obs_t2_ids_temp = local_phi_dynamic(s_local_new, all_obsStates, phi_params)
            phi_col.append(phi_new)
            if phi_smallest > phi_new:
                phi_smallest = phi_new
                optimal_action = action_temp
        # print(most_dangerObs_ids)
        # print(obstacle)
        # print(obs_t2_ids_temp)
        # print(unsafe_obstacle_ids[obs_t2_ids_temp])       
        # smallest_phi_dot = phi_smallest - phi_old
        # print("out = ", out)
        # print("valid = ", valid)
        # print("phi_smallest=", phi_smallest)
        # print("smallest_phi_dot=",smallest_phi_dot)
        # print("Step -------------------------------")
        return optimal_action, valid_adamba, NP_vec_tmp, out, yes, valid, phi_smallest, smallest_phi_dot, NP_vec_tmp_new, NP_vec_tmp, unsafe_obstacle_ids[obs_t2_ids_temp]
        #return NP_vec_tmp[optimal_action_index], valid_adamba, NP_vec_tmp, out, yes, valid
    else:
        # print(phi_old)
        # print("yes = ", yes)
        smallest_phi_dot = 0
        obs_state_t2 = []
        obs_t2_ids = 0
        obs_t2_ids_temp = 0
        return u, valid_adamba, None, out, yes, valid, phi_old, smallest_phi_dot, NP_vec_tmp_new, NP_vec_tmp, unsafe_obstacle_ids[obs_t2_ids_temp]


#def chk_unsafe(s, point, dt_ratio, dt_adamba, env, threshold):
def chk_unsafe(s, point, env, dt, threshold, phi_old, obstacle, all_obsStates, phi_params):
    fx= np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
    gx = np.array([[1,0],[0,1],[1,0],[0,1]])
    action = [point[0], point[1]]
    local_new_robot = local_step(s, action)
    s_local_new = [local_new_robot.x,local_new_robot.y,local_new_robot.v_x,local_new_robot.v_y]
    s_local_new = np.array(s_local_new)
    #s_new, reward, done, info = env.step(action, is_safe, unsafe_obstacle_ids) 
#If obstacles are stationary, only concern the single most dangerous obstacle all the time to avoid robot go into local optima    
    #phi_new, obs_t2_temp, obs_t2_ids_temp = local_phi_dynamic(s_local_new, all_obsStates, phi_params)
#If obstacles are dynamics, dynamically consider the most dangerous obstacle in next time step after taken AdamBA action 
    phi_new = local_phi_stationary(s_local_new, obstacle, phi_params)
    phi_dot = phi_new - phi_old
    if phi_dot < threshold * dt:
        flag = 0  # safe
    else:
        flag = 1  # unsafe
    return flag, phi_new, phi_dot



def local_step(s, action):
    local_robot = robot.DoubleIntegratorRobot(s[0],s[1],s[2],s[3],0.03)
    # local_robot.x = s[0]
    # local_robot.y=s[1]
    # local_robot.vx=s[2]
    # local_robot.vy=s[3]
    # local_robot.max_speed=0.03
    s_local_new = local_robot.steer(action[0],action[1])
    return s_local_new


def local_phi_stationary(robot_state, obstacle, phi_params):
    phi_collect = []
    alpha = phi_params[0]
    n = phi_params[1]
    k = phi_params[2]
    dmin=0.12
    d = np.array(robot_state - obstacle)
    d_pos = d[:2] # pos distance
    d_vel = d[2:] # vel 
    d_abs = np.linalg.norm(d_pos)
    d_dot = 1 * (d_pos @ d_vel.T) / np.linalg.norm(d_pos)
    phi = alpha + np.power(dmin, n) - np.power(np.linalg.norm(d_pos), n) - k * d_dot
    return phi

def local_phi_dynamic(robot_state, obs_states, phi_params):
    phi_collect = []
    phi_obs_states = {}
    phi_obs_ids = {}
    alpha = phi_params[0]
    n = phi_params[1]
    k = phi_params[2]
    dmin = 0.12
    for i, obs_state in enumerate(obs_states):
        d = np.array(robot_state - obs_state[:4])
        d_pos = d[:2] # pos distance
        d_vel = d[2:] # vel 
        d_abs = np.linalg.norm(d_pos)
        d_dot = (d_pos @ d_vel.T) / np.linalg.norm(d_pos)
        phi = alpha + np.power(dmin, n) - np.power(np.linalg.norm(d_pos), n) - k * d_dot 
        #phi = np.power(dmin, 2) - np.power(np.linalg.norm(d_pos), 2) - d_dot 
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

def CMAES_AdamBA(s, u, env, threshold, phi_old, 
        obstacle, all_obsStates, phi_params,ctrlrange=0.03, max_tral_num=1, vec_num=None): #(limit is robot maximum speed)
    infSet = []
    NP_vec_dir = []
    NP_vec = []
    phi_temp_all = []
    good_phi_temp = []
    phi_dot_temp = []
    good_phi_dot_temp = []
    bound = 0.0001
    collected_num = 0
    valid = 0
    cnt = 0
    out = 0
    yes = 0
    dt = 1
    sigma_vec =[[1, 0], [0, 1]]
    vec_num = 500
    loc = 0
    scale = 0.1
    u = np.clip(u, -ctrlrange, ctrlrange)
    np.random.seed(0)
    action_space_num = 2
    action = np.array(u).reshape(-1, action_space_num)
    limits = [[-ctrlrange, ctrlrange]] * action_space_num  
    NP = action
    start_time = time.time()
    if action_space_num ==2:
        vec_num = 10 if vec_num == None else vec_num
    elif action_space_num == 12:
        vec_num = 20 if vec_num == None else vec_num
    else:
        raise NotImplementedError
    # num of actions input, default as 1
    for t in range(0, NP.shape[0]):
        if action_space_num == 2:
            vec_set = []
            vec_dir_set = []
            for m in range(0, vec_num):
                theta_m = m * (2 * np.pi / vec_num)
                vec_dir = np.array([np.sin(theta_m), np.cos(theta_m)]) / 2
                vec_dir_set.append(vec_dir)
                vec = NP[t]
                vec_set.append(vec)
            NP_vec_dir.append(vec_dir_set)
            NP_vec.append(vec_set)
        else:
            vec_dir_set = np.random.normal(loc=loc, scale=scale, size=[vec_num, action_space_num])
            vec_set = [NP[t]] * vec_num
            #import ipdb; ipdb.set_trace()
            NP_vec_dir.append(vec_dir_set)
            NP_vec.append(vec_set)

    max_trials = max_tral_num
    for n in range(0, NP.shape[0]):
        trial_num = 0
        at_least_1 = False
        while trial_num < max_trials and not at_least_1:
            at_least_1 = False
            trial_num += 1
            NP_vec_tmp = copy.deepcopy(NP_vec[n])

            if trial_num ==1:
                NP_vec_dir_tmp = NP_vec_dir[n]
            else:
                NP_vec_dir_tmp = np.random.normal(loc=loc, scale=scale, size=[vec_num, action_space_num])

            for v in range(0, vec_num):
                NP_vec_tmp_i = NP_vec_tmp[v]
                NP_vec_dir_tmp_i = NP_vec_dir_tmp[v]
                eta = bound
                decrease_flag = 0
                while True:
                    flag, phi_new, phi_dot = chk_unsafe(s, NP_vec_tmp_i, env, dt, threshold, phi_old, obstacle, all_obsStates, phi_params)
                    # safety gym env itself has clip operation inside
                    if outofbound(limits, NP_vec_tmp_i):
                        # print("\nout\n")
                        #collected_num = collected_num - 1  # not found, discard the recorded number
                        break                   
                    if eta <= bound and flag == 0:
                        at_least_1 = True
                        break
                    # AdamBA procudure
                    if flag == 1 and decrease_flag == 0:
#unsafe control, keep strech out to find the boundary
                        # outreach
                        NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                        eta = eta * 2
                        continue
                    # monitor for 1st reaching out boundary
                    if flag == 0 and decrease_flag == 0:
#safe control found, but the vector is streching, need to set it to decay
                        decrease_flag = 1
                        eta = eta * 0.25  # make sure decrease step start at 0.5 of last increasing step
                        continue
                    # decrease eta
                    if flag == 1 and decrease_flag == 1 :
#the vector has gone too far decaying that exceed the safe control boundary, need to going streching again to land it inside the boundrary
                        NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                        eta = eta * 0.5
                        continue
                    if flag == 0 and decrease_flag == 1 :
#found the safe control set, now going back to find boundary
                        NP_vec_tmp_i = NP_vec_tmp_i - eta * NP_vec_dir_tmp_i
                        eta = eta * 0.5
                        continue
                NP_vec_tmp[v] = NP_vec_tmp_i
                phi_temp_all.append(phi_new)
                phi_dot_temp.append(phi_dot)
        NP_vec_tmp_new = []
        for vnum in range(0, len(NP_vec_tmp)):
            cnt += 1
            if outofbound(limits, NP_vec_tmp[vnum]):
                out += 1
                continue
            if NP_vec_tmp[vnum][0] == u[0] and NP_vec_tmp[vnum][1] == u[1]:
                yes += 1
                continue
            valid += 1
            NP_vec_tmp_new.append(NP_vec_tmp[vnum]) 
            good_phi_temp.append(phi_temp_all[vnum])
            good_phi_dot_temp.append(phi_dot_temp[vnum])
        NP_vec[n] = NP_vec_tmp_new
    end_time = time.time()
    NP_vec_tmp = NP_vec[0]
    if valid > 0:
        valid_adamba = "adamba success"
    elif valid == 0 and yes==vec_num:
        valid_adamba = "itself satisfy"
    elif valid == 0 and out==vec_num:
        valid_adamba = "all out"
    else:
        valid_adamba = "exception"
    if len(NP_vec_tmp_new) > 0:  # at least we have one sampled action satisfying the safety index 
#want to use the action with the smallest phi instead of using the action that is closest to the original action
#Pure AdamBA
        # norm_list = np.linalg.norm(NP_vec_tmp_new, axis=1)
        # optimal_action_index = np.where(norm_list == np.amin(norm_list))[0][0]
        # print("out = ", out)
        # print("valid = ", valid)
        # phi_best = good_phi_temp[optimal_action_index]
        # action_best = NP_vec_tmp_new[optimal_action_index]
        # print(action_best)
        # print(phi_best)
        # return action_best, valid_adamba, NP_vec_tmp, out, yes, valid, phi_best    
#Modified AdamBA
        phi_col = []
        phi_smallest = 100
        for i, action_temp in enumerate(NP_vec_tmp):
            local_new_robot = local_step(s, action_temp)
            s_local_new = [local_new_robot.x,local_new_robot.y,local_new_robot.v_x,local_new_robot.v_y]
            s_local_new = np.array(s_local_new)
            #s_new, reward, done, info = env.step(action, is_safe, unsafe_obstacle_ids) 
#When all obstacles are stationary
            phi_new = local_phi_stationary(s_local_new, obstacle, phi_params)
#When all obstacles are dynamic            
            #phi_new, obs_t2_temp, obs_t2_ids_temp = local_phi_dynamic(s_local_new, all_obsStates, phi_params)
            phi_col.append(phi_new)
            if phi_smallest > phi_new:
                phi_smallest = phi_new
                optimal_action = action_temp
        smallest_phi_dot = phi_smallest - phi_old
        print("out = ", out)
        print("valid = ", valid)
        print("phi_smallest=", phi_smallest)
        print("smallest_phi_dot=",smallest_phi_dot)
        print("Step -------------------------------")
        return smallest_phi_dot
    else:
        print(phi_old)
        print("yes = ", yes)
        smallest_phi_dot = 0
        return smallest_phi_dot


def AdamBASpreadPoints(s, u, env, threshold, phi_old, 
        obstacle, all_obsStates, phi_params, ctrlrange=0.006, max_tral_num=1, vec_num=None): #(limit is robot maximum speed)
    infSet = []
    u = np.clip(u, -ctrlrange, ctrlrange)
    np.random.seed(0)
#can change this to number of actions(controls)
    action_space_num = 2
    action = np.array(u).reshape(-1, action_space_num)
    limits = [[-ctrlrange, ctrlrange]] * action_space_num  # each row define the limits for one dimensional action
    NP = action
    start_time = time.time()
    NP_vec = []
    NP_vec_tmp = []
    phi_temp_all = []
    good_phi_temp = []
    phi_dot_temp = []
    good_phi_dot_temp = []
    NP_vec_tmp_new = []
    vec_num = 5000
    valid = 0
    cnt = 0
    out = 0
    yes = 0
    dt = 1
    max_trials = max_tral_num

    row_num = math.floor(np.sqrt(vec_num))
    x_temp = -ctrlrange
    y_temp = -ctrlrange
    # num of actions input, default as 1
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
                flag, phi_new, phi_dot = chk_unsafe(s, NP_vec_tmp_i, env, dt, threshold, phi_old, obstacle, all_obsStates, phi_params)
                if flag == 0:
                    NP_vec_tmp.append(NP_vec_tmp_i)
                    phi_temp_all.append(phi_new)
                    phi_dot_temp.append(phi_dot)
        for vnum in range(0, len(NP_vec_tmp)):
            cnt += 1
            if outofbound(limits, NP_vec_tmp[vnum]):
                # print("out")
                out += 1
                continue
            if NP_vec_tmp[vnum][0] == u[0] and NP_vec_tmp[vnum][1] == u[1]:
                # print("yes")
                yes += 1
                continue
            valid += 1
            NP_vec_tmp_new.append(NP_vec_tmp[vnum]) #This is the valid sets
            good_phi_temp.append(phi_temp_all[vnum])
            good_phi_dot_temp.append(phi_dot_temp[vnum])
        NP_vec[n] = NP_vec_tmp_new
    end_time = time.time()
    if len(NP_vec_tmp_new) > 0:  # at least we have one sampled action satisfying the safety index   
#Modified AdamBA
        phi_col = []
        phi_smallest = 100
#In time step t+1, this is the concerning obstacle, choose the action based on making this most dangerous obstacle safest
        obs_t2_temp = []
        obs_t2_ids_temp = 0
        obs_state_t2 = []
        obs_t2_ids = 0
        for i, action_temp in enumerate(NP_vec_tmp_new):
            local_new_robot = local_step(s, action_temp)
            s_local_new = [local_new_robot.x,local_new_robot.y,local_new_robot.v_x,local_new_robot.v_y]
            s_local_new = np.array(s_local_new)
            #s_new, reward, done, info = env.step(action, is_safe, unsafe_obstacle_ids) 
#When all obstacles are stationary
            #phi_new = local_phi_stationary(s_local_new, obstacle, phi_params)
#When all obstacles are dynamic            
            phi_new, obs_t2_temp, obs_t2_ids_temp = local_phi_dynamic(s_local_new, all_obsStates, phi_params)
            phi_col.append(phi_new)
            if phi_smallest > phi_new:
                phi_smallest = phi_new
                optimal_action = action_temp
        return NP_vec_tmp_new
    else:
        return NP_vec_tmp_new


def quadprog(H, f, A=None, b=None,
             initvals=None, verbose=False):
    qp_P = sparse.csc_matrix(H)
    qp_f = np.array(f)
    qp_l = -np.inf * np.ones(len(b))
    qp_A = sparse.csc_matrix(A)
    qp_u = np.array(b)

    model = osqp.OSQP()
    model.setup(P=qp_P, q=qp_f,
                A=qp_A, l=qp_l, u=qp_u, verbose=verbose)
    if initvals is not None:
        model.warm_start(x=initvals)
    results = model.solve()
    return results.x, results.info.status

