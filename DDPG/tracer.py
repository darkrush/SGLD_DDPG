import numpy as np
from copy import deepcopy
import itertools
import scipy.io as sio


class DDPG_tracer(object):
    def __init__(self, nb_rollout_steps, nb_train_steps, nb_state_step):
        self.nb_rollout_steps = nb_rollout_steps
        self.nb_train_steps = nb_train_steps
        self.nb_state_step = nb_state_step
        
    def setup(self, env, agent):
        self.env = env
        self.agent = agent
        self.action_scale = (env.action_space.high - env.action_space.low)/2.0
        self.action_bias = (env.action_space.high + env.action_space.low)/2.0
        self.action_space = self.env.action_space.shape[0]
        
        
        self.state_point = []
        for i in itertools.product(range(0,self.nb_state_step), repeat = env.observation_space.low.shape[0]):
            #(0.5,1.5,2.5...,N-2+0.5,N-1+0.5)/N
            self.state_point.append((np.array(i)+0.5)/self.nb_state_step)
        self.state_point = np.array(self.state_point)*(env.observation_space.high-env.observation_space.low)+env.observation_space.low
        
        
        self.reset()
        
    def reset(self):
        self.last_observation = deepcopy(self.env.reset())
        self.agent.reset_noise()
    
    def run_Qsurf(self):
        self.Q_value = []
        for idx_point in range(self.state_point.shape[0]):
            action = np.array([1,])
            self.Q_value.append(self.agent.calc_critic(self,obs,action))
        self.Q_value = np.array(self.Q_value)
        
    def save(self):
        sio.savemat('q_state.mat', {'q': np.array(self.q_surf_list), 'data':self.state_point,'obs':self.obs_list})
    def rollout(self):
        self.obs_list = []
        for t_rollout in range(self.nb_rollout_steps):
        
            #pick action by actor
            #action = self.agent.select_action(s_t = [self.last_observation], if_noise = True)
            action = np.array([1,])
            #apply the action to environment and get next state, reawrd and other information
            obs, reward, done, info = self.env.step(action * self.action_scale + self.action_bias)
            self.obs_list.append(obs)
            obs = deepcopy(obs)
            
            #store the transition into agent's replay buffer and update last observation
            self.agent.store_transition(self.last_observation, action,np.array([reward,]), obs, np.array([done,],dtype = np.float32))
            self.last_observation = deepcopy(obs)
                                
            #if current episode is done
            if done:
                self.last_observation = deepcopy(self.env.reset())
                self.agent.reset_noise()
        self.obs_list = np.array(self.obs_list) 
    def train(self):
        self.agent.update_num_pseudo_batches()
        cl_list = []
        
        for t_train in range(self.nb_train_steps):
            cl = self.agent.update_critic()
            cl_list.append(cl)
        print(cl_list)
        self.q_surf_list = []
        for t_train in range(30):
            self.agent.update_critic()
            q_surf = []
            for idx_obs in range(self.state_point.shape[0]):
                q = self.agent.calc_critic(np.expand_dims(self.state_point[idx_obs], axis=0))
                q_surf.append(q)
            q_surf = np.array(q_surf)
            self.q_surf_list.append(q_surf)
        
        