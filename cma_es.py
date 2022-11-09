import numpy as np
import time
import subprocess
import evaluator
import os, sys
from datetime import datetime
import yaml

class SafeLearning(object):
    def __init__(self, CMAES_args):
        """
        ================================================================================
        Initialize the learning module. Create learning log.
        """
        self.cmaes_args = CMAES_args

        now = datetime.now()

        timestamp = now.strftime("%m-%d_%H:%M")
        log_name = CMAES_args["exp_prefix"] + "_epoch:" + str(CMAES_args["epoch"]) + "_populate_num:" + str(CMAES_args["populate_num"]) + "_elite_ratio:" + str(CMAES_args["elite_ratio"]) + "_init_sigma_ratio:" + str(CMAES_args["init_sigma_ratio"]) + "_noise_ratio:" + str(CMAES_args["noise_ratio"]) + "_date:" + timestamp 
        #os.path.dirname(os.path.abspath(__file__)) + "/logs/" + log_name + ".txt","w")
        self.log = open("data.txt","w")

        self.evaluator = self.cmaes_args["evaluator"]








    def regulate_params(self, params):
        """
        ================================================================================
        Regulate params by upper bound and lower bound. And convert params to integer if required by the user.
        """
        params = np.maximum(params, self.cmaes_args["lower_bound"]) # lower bound
        params = np.minimum(params, self.cmaes_args["upper_bound"]) # upper bound
#make all population point as integer if  "param_is_int" in the config file, but in our case, we don't have this param
#This will no be executed
        if "param_is_int" in self.cmaes_args:
            for i in range(params.shape[1]):
                if self.cmaes_args["param_is_int"][i]:
                    params[:,[i]] = np.vstack([int(round(x)) for x in params[:,i]])
            # params = [ int(params[i]) if self.cmaes_args["param_is_int"][i] else params[i] ]
        return params








    def populate(self, mu, sigma):
        """
        ================================================================================
        Populate n members using the current estimates of mu and S
        """
        self.population = np.random.multivariate_normal(mu, sigma, self.cmaes_args["populate_num"])
        self.population = self.regulate_params(self.population)






    def evaluate(self, mu, log=True):
        """
        ===============================================================================
        Evaluate a set of weights (a mu) by interacting with the environment and
        return the average total reward over multiple repeats.
        """

        rewards = []
        repeat_times = 1  # test multiple times to reduce randomness
        for i in range(repeat_times):
            reward = self.evaluator.evaluate(mu) #evaluate from the evaluator function
            rewards.append(reward)

        #print('Rewards: {}'.format(rewards))

        reward = np.mean(rewards)
        if log:
            self.log.write("{} {}".format(str(mu), reward))
            self.log.write(self.evaluator.log+"\n")
            self.log.flush()
        return reward






    def step(self, mu, sigma):
        """
        ===============================================================================
        Perform an iteration of CMA-ES by evaluating all members of the current 
        population and then updating mu and S with the top self.cmaes_args["elite_ratio"] proportion of members 
        and updateing the weights of the policy networks.
        """
#popluate new populations based on previous mu and sigma
        self.populate(mu, sigma)
        rewards = np.array(list(map(self.evaluate, self.population)))
#将函数从小到大排序以后，提取对应的索引index，然后输出到indexes，-reward 就是最大的reward，这是要找出elite sets
        indexes = np.argsort(rewards) 
        """
        ===============================================================================
        best members are the top self.cmaes_args["elite_ratio"] proportion of members with the highest 
        evaluation rewards.
        """
        best_members = self.population[indexes[0:int(self.cmaes_args["elite_ratio"] * self.cmaes_args["populate_num"])]]
#这里通过求那10个elite set的mean 来更新 mu 和sigma        
        mu = np.mean(best_members, axis=0)
        sigma = np.cov(best_members.T) + self.noise
        best_reward = rewards[indexes[0]]
        x_loc = indexes[0]
        #print("avg best mu in this epoch:")
        #print(mu)
        return mu, sigma, rewards, best_reward, x_loc






#每三个epoch得到一个best mu
    def learn(self):
        mu = self.cmaes_args["init_params"]
        bound_range = np.array(self.cmaes_args["upper_bound"]) - np.array(self.cmaes_args["lower_bound"])
        sigma = np.diag((self.cmaes_args["init_sigma_ratio"] * bound_range)**2)
        self.noise = np.diag((self.cmaes_args["noise_ratio"] * bound_range)**2)
        
        for i in range(self.cmaes_args["epoch"]):
            self.log.write("epoch {}\n".format(i))
            mu, sigma, rewards, best_phi, x_loc = self.step(mu, sigma)
            #print("learning")
        print("Final best param:", mu)
        #print("Final reward:")
        #print(self.evaluate(mu, log=False))       
#Visulize the Phi
        #self.evaluator.visualize(mu, rewards, best_phi, x_loc)
        return mu



'''
def main(config_path):

    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            config["evaluator"] = eval("evaluator."+config["evaluator"]+"()")
            learner = SafeLearning(config)
            learner.learn()
        except yaml.YAMLError as exc:
            print(exc)
        

if __name__ == "__main__":
    
    sys.argv[0] = 'cma_es.py'
    sys.argv.append('SafetyIndex_config.yaml') 
    if len(sys.argv) < 2:
        print("===============================================================================")
        print("Please pass in the learning config file path. Pre-defined files are in config")
        print("===============================================================================")
    main(sys.argv[1])


'''


