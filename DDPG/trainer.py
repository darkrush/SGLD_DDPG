import numpy as np
import os
import pickle
import torch
import gym
from copy import deepcopy

from arguments import Singleton_arger
from logger import Singleton_logger
from evaluator import Singleton_evaluator

class DDPG_trainer(object):
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

        self.env = gym.make(main_args['env'])
        if main_args['rand_seed']>= 0:
            self.env.seed(main_args['rand_seed'])
        nb_actions = self.env.action_space.shape[0]
        nb_states = self.env.observation_space.shape[0]
        
        exploration_args = Singleton_arger()['exploration']
        if exploration_args['action_noise']:
            from AN_ddpg import action_noise_DDPG
            self.agent = action_noise_DDPG()
        elif exploration_args['parameter_noise']:
            from PN_ddpg import parameter_noise_DDPG
            self.agent = parameter_noise_DDPG()
        elif exploration_args['SGLD_mode'] > 0:
            from SGLD_ddpg import SGLD_DDPG
            self.agent = SGLD_DDPG()
        else:
            from ddpg import DDPG
            self.agent = DDPG()
            
        self.agent.setup(nb_states, nb_actions)
        self.action_scale = (self.env.action_space.high - self.env.action_space.low)/2.0
        self.action_bias = (self.env.action_space.high + self.env.action_space.low)/2.0
        self.action_space = self.env.action_space.shape[0]
        self.result_dir = main_args['result_dir']
        self.reset()
        
    def reset(self):
        self.last_episode_length = 0
        self.current_episode_length = 0
        self.current_episode_reward = 0.
        self.last_episode_reward = 0.
        self.total_step = 0
        self.last_observation = deepcopy(self.env.reset())
        self.agent.reset_noise()
        
    def warmup(self):
        for t_warmup in range(self.nb_warmup_steps):
            #pick action by actor randomly
            self.apply_action(np.random.uniform(-1.,1.,self.action_space))
                
    def train(self):
        for epoch in range(self.nb_epoch):
            #apply hyperparameter decay
            self.agent.before_epoch()
            self.agent.apply_lr_decay()
            for cycle in range(self.nb_cycles_per_epoch):
                self.agent.before_cycle()
                
                cl_mean,al_mean = self.apply_train()
                
                for t_rollout in range(self.nb_rollout_steps):
                    #pick action by actor in state "last_observation"
                    self.apply_action(self.agent.select_action(s_t = [self.last_observation], apply_noise = True))
                    self.total_step += 1
                #End Rollout
                
                #trigger log events
                last_error = self.agent.calc_last_error()
                Singleton_logger.trigger_log('last_error', last_error,self.total_step)
                Singleton_logger.trigger_log('train_episode_length', self.last_episode_length,self.total_step)
                Singleton_logger.trigger_log('train_episode_reward', self.last_episode_reward,self.total_step)
                Singleton_logger.trigger_log('critic_loss_mean', cl_mean, self.total_step)
                Singleton_logger.trigger_log('actor_loss_mean', al_mean, self.total_step)
                
            #End Cycle 
            #save agent to disk
            self.agent.save_model(self.result_dir)
            
            #trigger evaluation and log_save
            Singleton_evaluator.trigger_load_from_file(actor_dir = self.result_dir)
            Singleton_evaluator.trigger_eval_process(total_cycle = self.total_step)
            Singleton_logger.trigger_save()
        #End Epoch
        
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
            if isinstance(self.agent, SGLD_DDPG)
                for t_train in range(self.nb_train_steps):
                    cl = self.agent.update_critic(last_step = self.nb_train_steps-t_train-1)
                    cl_list.append(cl)
                for t_train in range(int(self.nb_train_steps)):
                    al = self.agent.update_actor()
                    al_list.append(al)
                self.agent.update_critic_target(soft_update = False)
                self.agent.update_actor_target (soft_update = False)
            else:
                for t_train in range(self.nb_train_steps):
                    cl = self.agent.update_critic()
                    cl_list.append(cl)
                for t_train in range(self.nb_train_steps):
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
    trainer = DDPG_trainer()
    trainer.setup()
    trainer.warmup()
    trainer.train()