import numpy as np
import os
import pickle
import torch
import gym
import itertools
import scipy.io as sio
from copy import deepcopy

from arguments import Singleton_arger
from logger import Singleton_logger
from evaluator import Singleton_evaluator

from ddpg import DDPG
from AN_ddpg import action_noise_DDPG
from PN_ddpg import parameter_noise_DDPG
from SGLD_ddpg import SGLD_DDPG

class DDPG_tracer(object):
    def __init__(self):
        train_args = Singleton_arger()['train']
        self.nb_epoch = train_args['nb_epoch']
        self.nb_cycles_per_epoch = train_args['nb_cycles_per_epoch']
        self.nb_rollout_steps = train_args['nb_rollout_steps']
        self.nb_train_steps = train_args['nb_train_steps']
        self.nb_warmup_steps = train_args['nb_warmup_steps']
        self.train_mode = train_args['train_mode']
        
    def setup(self):
        main_args = Singleton_arger()['main']
        Singleton_logger.setup(main_args['result_dir'],multi_process = main_args['multi_process'])

        Singleton_evaluator.setup(main_args['env'], logger = Singleton_logger, num_episodes = 10, model_dir = main_args['result_dir'], multi_process = main_args['multi_process'], visualize = False, rand_seed = main_args['rand_seed'])
        
        env_name_list = main_args['env'].split('_')
        if len(env_name_list)>1:
            self.env = gym.make(env_name_list[0])
            self.env.env.change_coef = float(env_name_list[1])
        else:
            self.env = gym.make(main_args['env'])
        if main_args['rand_seed']>= 0:
            self.env.seed(main_args['rand_seed'])
        nb_actions = self.env.action_space.shape[0]
        nb_states = self.env.observation_space.shape[0]
        
        exploration_args = Singleton_arger()['exploration']
        if exploration_args['action_noise']:
            self.agent = action_noise_DDPG()
        elif exploration_args['parameter_noise']:
            self.agent = parameter_noise_DDPG()
        elif exploration_args['SGLD_mode'] > 0:
            self.agent = SGLD_DDPG()
        else:
            self.agent = DDPG()
            
        self.agent.setup(nb_states, nb_actions)
        self.action_scale = (self.env.action_space.high - self.env.action_space.low)/2.0
        self.action_bias = (self.env.action_space.high + self.env.action_space.low)/2.0
        self.action_space = self.env.action_space.shape[0]
        self.result_dir = main_args['result_dir']
        
        self.nb_state_step = 20
        self.state_point = []
        for i in itertools.product(range(0,self.nb_state_step), repeat = self.env.observation_space.low.shape[0]):
            #(0.5,1.5,2.5...,N-2+0.5,N-1+0.5)/N
            self.state_point.append((np.array(i)+0.5)/self.nb_state_step)
        self.state_point = np.array(self.state_point)*(self.env.observation_space.high-self.env.observation_space.low)+self.env.observation_space.low
        
        
        self.reset()
        
    def reset(self):
        self.last_episode_length = 0
        self.current_episode_length = 0
        self.current_episode_reward = 0.
        self.last_episode_reward = 0.
        self.total_step = 0
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
        
    def warmup(self):
        self.obs_list = []
        for t_warmup in range(self.nb_warmup_steps):
            #pick action by actor randomly
            self.apply_action(action = np.array([1,]))
            
        self.obs_list = np.array(self.obs_list) 
                
    def train(self):
        self.agent.before_epoch()
        self.agent.apply_lr_decay()
        self.agent.before_cycle()
        cl_mean,al_mean = self.apply_train()
        self.q_surf_list = []
        for t_train in range(2000):
            self.agent.update_critic()
        for t_train in range(30):
            self.agent.update_critic()
            q_surf = []
            for idx_obs in range(self.state_point.shape[0]):
                q = self.agent.calc_critic(np.expand_dims(self.state_point[idx_obs], axis=0))
                q_surf.append(q)
            q_surf = np.array(q_surf)
            self.q_surf_list.append(q_surf)
            
    def apply_train(self):
        #update agent for nb_train_steps times
        cl_list = []
        al_list = []
        if self.train_mode == 0:
            for t_train in range(self.nb_train_steps):
                cl = self.agent.update_critic()
                al = self.agent.update_actor()
                self.agent.update_critic_target()
                self.agent.update_actor_target()
                cl_list.append(cl)
                al_list.append(al)
        elif self.train_mode == 1:
            for t_train in range(self.nb_train_steps):
                cl = self.agent.update_critic()
                cl_list.append(cl)
                al = self.agent.update_actor()
                al_list.append(al)
            self.agent.update_critic_target(soft_update = False)
            self.agent.update_actor_target (soft_update = False)
        return np.mean(cl_list),np.mean(al_list)
        
        
    def apply_action(self, action):
        #apply the action to environment and get next state, reawrd and other information
        obs, reward, done, time_done, info = self.env.step(action * self.action_scale + self.action_bias)
        self.current_episode_reward += reward
        self.current_episode_length += 1
        obs = deepcopy(obs)
        self.obs_list.append(obs)
        #store the transition into agent's replay buffer and update last observation
        self.agent.store_transition(self.last_observation, action,np.array([reward,]), obs, np.array([done,],dtype = np.float32))
        self.last_observation = deepcopy(obs)
        
        #if current episode is done
        if done or time_done:
            self.last_observation = deepcopy(self.env.reset())
            self.agent.reset_noise()
            self.last_episode_reward = self.current_episode_reward
            self.last_episode_length = self.current_episode_length
            self.current_episode_reward = 0.
            self.current_episode_length = 0
            
    def __del__(self):
        Singleton_evaluator.trigger_close()
        Singleton_logger.trigger_close()

if __name__ == "__main__":
    trainer = DDPG_tracer()
    trainer.setup()
    trainer.warmup()
    trainer.train()
    trainer.save()