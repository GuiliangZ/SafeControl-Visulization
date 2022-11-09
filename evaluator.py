import numpy as np
import time
import subprocess
import sys, os
import copy
import matplotlib.pyplot as plt

import ISSA_AdamBA as IA
import SafeSet_Com as SSC

class Evaluator(object):
    # Each evaluator must contain two major functions: set_params(params) and evaluate(params)
    def set_params(self, params):
        # Set the params to the desired component
        raise NotImplementedError
    
    def evaluate(self, params):
        # Execute and collect reward.
        raise NotImplementedError

    def visualize(self, final_params):
        # visualize the learned final params.
        pass

    @property
    def log(self):
        return "" 



class SafetyIndexEvaluator(Evaluator):

    def __init__(self, s_worst, env_worst, u_CMAES, phi_old_CMAES, obstacle_CMAES, all_obsStates_CMAES):
        self.alpha = 0
        self.n = 0
        self.k = 0
        self.s_worst = s_worst
        self.env_worst = env_worst
        self.u_CMAES = u_CMAES
        self.phi_old_CMAES = phi_old_CMAES
        self.obstacle_CMAES = obstacle_CMAES
        self.all_obsStates_CMAES = all_obsStates_CMAES  





    def evaluate(self, params):
#evaluate the validity of each set of parameters
#return the reward
        self.alpha = params[0]
        self.n = params[1]
        self.k = params[2]


        # smallest_phi_dot = IA.CMAES_AdamBA(s = self.s_worst, u = self.u_CMAES, env = self.env_worst, threshold=0, 
        # phi_old = self.phi_old_CMAES, obstacle=self.obstacle_CMAES, all_obsStates = self.all_obsStates_CMAES, 
        # phi_params=params)

#maximize the number of true common sets as the reward function
        true_common_safeset, best_common_action, safeset_com_bound_num, best_action_phi_dot = SSC.safeset_common(s = self.s_worst, u = self.u_CMAES, env = self.env_worst,
                                        all_obsStates = self.all_obsStates_CMAES, phi_params=params, max_acc = 0.005)

        return best_action_phi_dot



    def visualize(self, final_params, all_phi, best_phi, x_loc):
        xs = np.linspace(0, 99, 100)
        ys = all_phi
        plt.scatter(xs, ys)
        plt.scatter(x_loc, best_phi,c = 'r',marker = 'x')
        #plt.vlines(x=final_params[0], ymin=-15, ymax=3, label="Learned", colors="g")
        #plt.vlines(x=1, ymin=-15, ymax=3, label="Optima", colors="r", linestyles=":")
        plt.legend()
        plt.show()


    @property
    def log(self):
        return "k, n, alpha = " + str(self.k) + " " + str(self.n) + " " + str(self.alpha)












class TestEvaluator(Evaluator):
    # Each evaluator must contain two major functions: set_params(params) and evaluate(params)
    def __init__(self):
        self.x = 0

    def evaluate(self, params):
        # Execute and collect reward.
        self.x = params[0]
        return -(self.x - 1)** 2

    def visualize(self, final_params):
        xs = np.linspace(-3, 3, 100)
        ys = -(xs - 1) ** 2
        plt.plot(xs, ys)
        plt.vlines(x=final_params[0], ymin=-15, ymax=3, label="Learned", colors="g")
        plt.vlines(x=1, ymin=-15, ymax=3, label="Optima", colors="r", linestyles=":")
        plt.legend()
        plt.show()
 
    @property
    def log(self):
        return "x = " + str(self.x)

class PDEvaluator(Evaluator):
    # Each evaluator must contain two major functions: set_params(params) and evaluate(params)
    def __init__(self):
        self.xf = [10,0]
        self.dt = 0.01
    
    def simulation(self, kp, kd):
        x = np.zeros(2)
        e_prev = np.zeros(2)
        e = np.zeros(2)
        de = np.zeros(2)
        num_step = 10000
        self.xs = []
        for i in range(num_step):
            de = e - e_prev
            e_prev = e
            e = self.xf - x
            u = kp @ e.T + kd @ de.T / self.dt
            x[0] += 0.5 * u * self.dt ** 2 + x[1] * self.dt
            x[1] += u * self.dt
            self.xs.append(copy.copy(x))

            if np.linalg.norm(e) < 1e-3:
                return num_step - i

        return -np.linalg.norm(e) - np.linalg.norm(de) / self.dt

    def evaluate(self, params):
        # Execute and collect reward.
        self.kp = params[[0,1]]
        self.kd = params[[2,3]]
        return self.simulation(self.kp, self.kd)

    def visualize(self, params):
        self.evaluate(params)
        plt.hlines(y=self.xf[0], xmin=0, xmax=len(self.xs), label="reference")
        xs = [x[0] for x in self.xs]
        plt.plot(xs, label="traj")
        plt.legend()
        plt.show()

    @property
    def log(self):
        return "kp, kd = " + str(self.kp) + " " + str(self.kd)