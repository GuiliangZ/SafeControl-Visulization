from __future__ import print_function
from __future__ import absolute_import

######################################################################
# This file copyright the Georgia Institute of Technology
#
# Permission is given to students to use or modify this file (only)
# to work on their assignments.
#
# You may NOT publish this file or make it available to others not in
# the course.
#
######################################################################

# python modules
import argparse
from dataclasses import field
import importlib  
import math
import random  
import numpy as np
import os.path
import sys
import collections
from itertools import cycle

# project files
import dynamic_obstacle
import bounds
import robot # double integrator robot
import simu_env
import runner
import param
from turtle_display import TurtleRunnerDisplay
from ssa import SafeSetAlgorithm
import ISSA_AdamBA as IA
import SafeSet_Com as SSC 


#zgl
import ISSA_AdamBA
import evaluator
from cma_es import SafeLearning
import yaml

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



def main(display_name, exploration, qp, enable_ssa_buffer, ISSA):
#Starts to initialize the parameters
    episode_limit = 1
      # testing env
    try:
        params = param.params
    except Exception as e:
        print(e)
        return
    display = display_for_name(display_name)
    env_params = run_kwargs(params)
    env = simu_env.Env(display, **(env_params))
    # ssa
    safe_controller = SafeSetAlgorithm(max_speed = env.robot_state.max_speed, is_qp = qp)

    is_meet_requirement = False
    reward_records = []#del
    robot_xs = []
    robot_ys = []
    obs_xs = []
    obs_ys = []
    safe_obs_xs = []
    safe_obs_ys = []
    xs_qp = []
    ys_qp = []
    obs_xs_qp = []
    obs_ys_qp = []
    out_s = []
    yes_s = []
    valid_s = []
#Newly added parameters:
    s_next_new = []
    phi = 0
#common safe sets numbers for different phi params
    safeset_com_num_list = []
#Collect number of times that collide    
    collision_num = 0
    MO_SIndex_largest = 0
    phi_ori_col = [] 
    phi_AdamBA_col = []
    most_dangerObs_ids = -1
    obs_t2_ids = -1
    # parameters
    max_steps = int(1e6)
    start_timesteps = 2e3
    episode_reward = 0
    episode_num = 0
    last_episode_reward = 0
    teacher_forcing_rate = 0
    total_rewards = []
    total_steps = 0
    # dynamic model parameters
    fx = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
    gx = np.array([[1,0],[0,1],[1,0],[0,1]])
    state, done = env.reset(), False
    collision_num = 0
    failure_num = 0
    success_num = 0
    # rl policy
    robot_state_size = 4 #(x,y,v_x,v_y)
    robot_action_size = 2
    nearest_obstacle_state_size = 2 #(delta_x, delta_y)
    state_dim = robot_state_size + nearest_obstacle_state_size
    safeset_com_num_list_temp = []
    flag = 1
    unsafe_obstacles = []


    for t in range(max_steps):
      #use different phi_params to run in the same env
      # phi_params = choices_phi(episode_num)
      #phi_params = [0.09999241612294944, 3.548114187928693, 0.1453982037959684]
      #phi_params = [0.013116, 0.96088, 0.0]
      phi_params = [0,2,1]
      #phi_params = [0.16435518, 3.32194647, 5.55730166]
      #phi_params = [0.3034039933606632, 1.8266185321204726, 2.5760077153076226]
      # phi_params = [0.1398, 4.2405, 8.8942]
      #phi_params = [0.12596910411874115, 5.0, 10.0]

      
      action = nominal_action(env.max_acc)
      original_action = action
      env.display_start()
      # ssa parameters
      unsafe_obstacle_ids, unsafe_obstacles, unsafe_obstacles_c = env.find_unsafe_obstacles(env.min_dist * 6)
      #safe_action = cautious_control(env.field, env.robot_state, unsafe_obstacle_ids, unsafe_obstacles, env.cur_step, env.min_dist)
      #action, is_safe = cbf_controller.get_safe_control(state[:4], unsafe_obstacles, fx, gx, action)
      #action, is_safe = shield_controller.probshield_control(state[:4], unsafe_obstacles, fx, gx, action, env.field, unsafe_obstacle_ids, unsafe_obstacles, env.cur_step)
      action_ssa, is_safe, is_unavoidable, danger_obs, phis_unsafe_temp_old, most_dangerObs_state, most_dangerObs_ids, MO_SIndex_old = safe_controller.get_safe_control(state[:4], unsafe_obstacles, fx, gx, action, phi_params, unsafe_obstacle_ids)
      # print("most_dangerObs_state=", most_dangerObs_state)
      # print("phis_unsafe_temp_old=", phis_unsafe_temp_old)
      # print("MO_SIndex_old=", MO_SIndex_old)
      #all_obs_states = env.find_all_obstacle_loc(state[0],state[1])

# u could be original_action
# u could be action(original action being filtered by the "get safe control")

# #Use the AdamBA to change the control action geneated by either vanilla SSA or adapated SSA:å

      # phi_ori_col.append(phis_unsafe_temp_old)
      # phi_AdamBA_col.append(phis_unsafe_temp_old) 

      if ISSA:
        #need to know phi in the ssa(future phi and current phi)
        if phis_unsafe_temp_old>0:
          # action, valid_adamba, NP_vec_tmp, out, yes, valid= IA.AdamBA(s = state[:4], u = original_action, env = env, threshold=0, phi_old = phis_unsafe_temp_old,
          #                                 unsafe_obstacles=unsafe_obstacles)
          
          # action, valid_adamba, NP_vec_tmp, out, yes, valid, phi_AdamBA, smallest_phi_dot, Safe_ControlSet_AdamBA, Safe_ControlSet_bound, obs_t2_ids = IA.AdamBA(s = state[:4], u = original_action, env = env, threshold=0, phi_old = phis_unsafe_temp_old,
          #                         obstacle=most_dangerObs_state, all_obsStates = unsafe_obstacles, phi_params = phi_params, unsafe_obstacle_ids = unsafe_obstacle_ids,most_dangerObs_ids=most_dangerObs_ids)
          
          # Safe_ControlSet_large = IA.AdamBASpreadPoints(s = state[:4], u = original_action, env = env, threshold=0, phi_old = phis_unsafe_temp_old,
          #               obstacle=most_dangerObs_state, all_obsStates = unsafe_obstacles, phi_params = phi_params)

          true_common_safeset, best_common_action, safeset_com_bound_num, best_action_phi_dot = SSC.safeset_common(s = state[:4], u = original_action, env = env,
                                                   all_obsStates = unsafe_obstacles, unsafe_obstacles_c=unsafe_obstacles_c, phi_params = phi_params, max_acc = env.max_acc)
          action = best_common_action
          # print(safeset_com_bound_num)

          if safeset_com_num_list_temp:
#collect the numbers of safesets for each different safe set parameters
            safeset_com_num_list_temp.append(safeset_com_bound_num)
          else:
            safeset_com_num_list_temp.append(phi_params[0])
            safeset_com_num_list_temp.append(phi_params[1])
            safeset_com_num_list_temp.append(phi_params[2])


          #plot_ControlSpace(true_common_safeset, original_action, action, env.max_acc)

# #Plot the control space graph
          

          if MO_SIndex_largest< MO_SIndex_old:
          # if flag:
            MO_SIndex_largest = MO_SIndex_old
            s_worst = state[:4]
            env_worst = env
            u_CMAES = original_action
            phi_old_CMAES = phis_unsafe_temp_old
            obstacle_CMAES = most_dangerObs_state
            all_obsStates_CMAES = unsafe_obstacles
            flag = 0
          
          # if phi_AdamBA != phis_unsafe_temp_old:
          #   phi_ori_col.append(phis_unsafe_temp_old)
          #   phi_AdamBA_col.append(phis_unsafe_temp_old)
          #   phi_AdamBA_col.pop()
          #   phi_AdamBA_col.append(phi_AdamBA)                        
      s_new, reward, done, info = env.step(action, obs_t2_ids, most_dangerObs_ids, is_safe, unsafe_obstacle_ids) 
      env.display_end()
      if phis_unsafe_temp_old>0:
        plot_ControlSpace(true_common_safeset, original_action, action, env.max_acc)
        if len(unsafe_obstacles)>1:
          SSC.plot_Obs_SafeComSet(state[:4], original_action, unsafe_obstacles_c, unsafe_obstacles, env, phi_params)
      old_state = state
      state = s_new
      original_reward = reward
      episode_reward += original_reward

      #Record data 
      #'''
      if (len(danger_obs) > 0):
        for obs in danger_obs:
          obs_xs.append(obs[0])
          obs_ys.append(obs[1])
      for obs in env.field.obstacles:
        safe_obs_xs.append(obs.c_x)
        safe_obs_ys.append(obs.c_y)
      robot_xs.append(state[0])
      robot_ys.append(state[1])
      #'''

      if (done and original_reward == -500):          
        print("collision")
        collision_num += 1      
        safe_controller.plot_control_subspace(old_state[:4], unsafe_obstacles, fx, gx, original_action)
      elif (done and original_reward == 2000):
        success_num += 1
      elif (done):
        failure_num += 1
      if (done):
        safeset_com_num_list.append(safeset_com_num_list_temp)
        safeset_com_num_list_temp = []   
        total_steps += env.cur_step
        total_rewards.append(episode_reward)
        episode_reward = 0
        episode_num += 1
        state, done = env.reset(), False 
        if (episode_num >= episode_limit):
          # print(collision_num)
          # print(success_num)
#could plot all the true common safe sets for different phi parameters
          # plt.close('all')
          # plot_NumSafeComSet(safeset_com_num_list)      
          break

        np.save('record/xs.npy', np.array(robot_xs))
        np.save('record/ys.npy', np.array(robot_ys))
        np.save('record/obs_xs.npy', np.array(obs_xs))
        np.save('record/obs_ys.npy', np.array(obs_ys))
        np.save('record/safe_obs_xs.npy', np.array(safe_obs_xs))
        np.save('record/safe_obs_ys.npy', np.array(safe_obs_ys))
        np.save('record/xs_qp.npy', np.array(xs_qp))
        np.save('record/ys_qp.npy', np.array(ys_qp))
        np.save('record/obs_xs_qp.npy', np.array(obs_xs_qp))
        np.save('record/obs_ys_qp.npy', np.array(obs_ys_qp))
        print(f"Train: episode_num {episode_num}, total_steps {total_steps}, reward {episode_reward}, is_qp {qp}, exploration {exploration}, last state {state[:2]}")
    #collecting phi with and without AdamBA
      np.save('record/phi_ori_col.npy', np.array(phi_ori_col))
      np.save('record/phi_AdamBA_col.npy', np.array(phi_AdamBA_col))

    print("CMA-ES Optimization Starts!!!!!!!!!")
    print(MO_SIndex_largest)
    #implement CMA-ES to optimize three parameters of phi
    with open('configs/SafetyIndex_config.yaml', 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            config["evaluator"] = evaluator.SafetyIndexEvaluator(s_worst, env_worst, u_CMAES, phi_old_CMAES, obstacle_CMAES, all_obsStates_CMAES)
            learner = SafeLearning(config)
            alpha, n, k = learner.learn()
            print("alpha={}, n={}, k = {}",alpha,n,k) 
        except yaml.YAMLError as exc:
            print(exc)
        #alpha, n, k = SafeLearning(self.dmin, d_pos, d_dot)
    return reward_records

def choices_phi(episode_num):
  if episode_num == 0:
    phi_params = [0,2,1]
  elif episode_num == 1:
    phi_params = [0, 0.4437, 2.9941]
  elif episode_num == 2:
    phi_params = [0.0087, 0.2426, 5]
  elif episode_num == 3:
    phi_params = [0.0255,0.3329, 6.9676]
  elif episode_num == 4:
    phi_params = [0.2356, 0.4146, 8.4251]
  elif episode_num == 5:
    phi_params = [0.0777, 0.9804, 9.9068]
  elif episode_num == 6:
    phi_params = [0.0560, 0.4174, 99.9183]
  elif episode_num == 7:
    phi_params = [0.4218, 1.4221, 0.2444]
  elif episode_num == 8:
    phi_params = [1.3989, 0.3752, 0.3991]
  elif episode_num == 9:
    phi_params = [0.3034, 1.8266, 2.5760]
  return phi_params
    

def plot_NumSafeComSet(safeset_com_num_list):
  #could plot all the true common safe sets for different phi parameters
  cycol = cycle('bgrcmkyw')
  for safeset_com_num in safeset_com_num_list:
    x = list(range(len(safeset_com_num)))
    legend_num_alpha = safeset_com_num[0]
    legend_num_n = safeset_com_num[1]
    legend_num_k = safeset_com_num[2]
    plt.plot(x,safeset_com_num, '-.', c=np.random.rand(3), linewidth = 1, label = f'alpha ={legend_num_alpha},n ={legend_num_n},k ={legend_num_k} ')
  plt.title('Numbers of available common safe control sets for different parameters')
  plt.legend()  
  plt.show()
  plt.savefig("../V3/SafeComSets.png", dpi=600, format='png')

def plot_ControlSpace(true_common_safeset, original_action, action, max_acc):
  plt.close('all')
  plt.clf()
  x_plot = []
  y_plot = []
  x_plot_common = []
  y_plot_common = []
  x_plot_AdamBA = []
  y_plot_AdamBA = []
  x_plot_bound = []
  y_plot_bound = []
  # for i,action_temp in enumerate(Safe_ControlSet_large):
  #   x_plot.append(action_temp[0]) 
  #   y_plot.append(action_temp[1])
  for i,action_temp in enumerate(true_common_safeset):
    x_plot_common.append(action_temp[0]) 
    y_plot_common.append(action_temp[1])
  # for i,action_temp in enumerate(Safe_ControlSet_AdamBA):
  #   x_plot_AdamBA.append(action_temp[0]) 
  #   y_plot_AdamBA.append(action_temp[1])
  # for i,action_temp in enumerate(Safe_ControlSet_bound):
  #   x_plot_bound.append(action_temp[0]) 
  #   y_plot_bound.append(action_temp[1])
  plt.scatter(original_action[0],original_action[1], c = "black", marker = 'o', label = 'original_action')
  # plt.scatter(x_plot,y_plot, c = "b", alpha = 0.2)
  plt.scatter(x_plot_common,y_plot_common, c = "orange", alpha = 0.5)            
  # plt.scatter(x_plot_bound,y_plot_bound, c = "purple")  
  # plt.scatter(x_plot_AdamBA,y_plot_AdamBA, c = "yellow")  
  plt.scatter(action[0],action[1],c = "r", marker = '*', label = 'original_action')
  rect = mpatches.Rectangle((-max_acc,-max_acc), max_acc*2, max_acc*2, 
                    fill = True, alpha = 0.2, color = "green")  
  plt.gca().add_patch(rect)
  # plt.show()
  plt.draw()
  plt.pause(1)
  plt.close('all')

def nominal_action(max_acc):
  x_acc = 0
  y_acc = 0.0025
  expl_action = []
  expl_action.append(x_acc)
  expl_action.append(y_acc)
  expl_action[0] = max(min(expl_action[0], max_acc), -max_acc)
  expl_action[1] = max(min(expl_action[1], max_acc), -max_acc)
  return expl_action

def display_for_name( dname ):
    # choose none display or visual display
    if dname == 'turtle':
        return TurtleRunnerDisplay(800,800)
    else:
        return runner.BaseRunnerDisplay()

def run_kwargs( params ):
    in_bounds = bounds.BoundsRectangle( **params['in_bounds'] )
    goal_bounds = bounds.BoundsRectangle( **params['goal_bounds'] )
    min_dist = params['min_dist']
    ret = { 'field': dynamic_obstacle.ObstacleField(),
            'robot_state': robot.DoubleIntegratorRobot( **( params['initial_robot_state'] ) ),
            'in_bounds': in_bounds,
            'goal_bounds': goal_bounds,
            'noise_sigma': params['noise_sigma'],
            'min_dist': min_dist,
            'nsteps': 1000 }
    return ret

def parser():
    prsr = argparse.ArgumentParser()
    prsr.add_argument( '--display',
                       choices=('turtle','text','none'),
                       default='none' )
    prsr.add_argument( '--explore',
                   choices=('psn','rnd','none'),
                   default='none' )
    prsr.add_argument( '--qp',dest='is_qp', action='store_true')
    prsr.add_argument( '--no-qp',dest='is_qp', action='store_false')
    prsr.add_argument( '--ssa-buffer',dest='enable_ssa_buffer', action='store_true')
    prsr.add_argument( '--no-ssa-buffer',dest='enable_ssa_buffer', action='store_false')
    return prsr





if __name__ == '__main__':
    args = parser().parse_args()
    all_reward_records = []
    for i in range(100):
      all_reward_records.append([])
    for i in range(1):
      reward_records = main(display_name = 'turtle',#args.display, 
          exploration = 'none',#args.explore,
          qp = False,#args.is_qp,
          enable_ssa_buffer = False,
          ISSA = True
          )#args.enable_ssa_buffer)s
      # for j, n in enumerate(reward_records):
      #   all_reward_records[j].append(n)
      #print(all_reward_records)
    #np.save('plot_result/ssa_rl.npy', np.array(all_reward_records))






#Stationary Obstacle Parameters
    #phi_params = [0,0.51790572,2.74636061]
    #phi_params = [0, 0.44375845, 2.99418662]
    #phi_params = [0.05622761319, 0.2953935145, 4.971045]
    #phi_params = [0.53438056, 0.35061571, 9.97233949]
    #phi_params = [0.3280583193647232, 0.4335922132226212, 19.973202193301056]

#Doesn't work
    #phi_params = [0.00879517, 0.24262109, 5]
    # phi_params = [0.0255215, 0.33291034, 6.96764781]
    #phi_params = [0.23564849, 0.41465331, 8.425172504511929]
    #phi_params = [0.04696510193635237, 1.4697965686401715, 9.794249025046284]
  #Learnt in a predefined stationary obstacle configuration
    #phi_params = [0.05637753968015462, 0.3894856054255201, 18.93207872380855]
    #phi_params = [0.8613649791427497, 0.5693015935701027, 21.91269530490127]
    #phi_params = [0.7183482393797289, 0.3188821426139298, 29.903551439815622]
    #phi_params = [0.056012173165720026, 0.41745273536412925, 99.91836222870572]
    #phi_params = [0.01634973967819753, 0.33084468193801175, 100.0]


#Dynamic Obstacle Parameters (Doesn't work on predefined stationary obs)
    #phi_params = [0.00879517, 0.24262109, 5]
    #phi_params = [0.0255215,0.33291034, 6.96764781]
    #phi_params = [0.01990758, 1.42645702,19.53025317]
    #phi_params = [0.41791110149587374, 2.0458143923072054, 94.19758442914573]
    #phi_params = [0.04355618561495937, 0.22942225587256523, 99.89409942310293]
#Dynamically learnt that work with stationary obstacle
    #phi_params = [0.07772311, 0.98041412, 9.90680764]
    #phi_params = [0.14015266533912432, 4.899243775227541, 9.974246079962754]
    #phi_params = [0.0007356653481057922, 1.5145270315525288, 0.0]
    #phi_params = [0,2,1]

    #phi_params = [0.0052729196189541154, 0.2254118294394611, 9.903872730865913]
    #phi_params = [7.45216468e-05, 3.60548843e+00, 3.37579849e+01]

#CMA-ES optimized in a set dynamic env-V2, safe common number reward fnc
    #Use phi_dot to judge whether its safe or not instead of using phi_0
    #phi_params = [0.09999241612294944, 3.548114187928693, 0.1453982037959684]
    #Use phi_0 to judge whether its safe or not
    #phi_params = [0.3034039933606632, 1.8266185321204726, 2.5760077153076226]
