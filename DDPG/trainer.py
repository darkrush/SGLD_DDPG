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
        self.args_dict = {}
        for key in ('nb_epoch','nb_cycles_per_epoch','nb_rollout_steps','nb_train_steps','nb_warmup_steps',
                    'rand_seed','resume','train_mode','with_cuda','env','result_dir',
                    'multi_process','action_noise','parameter_noise','SGLD_mode'):
            self.args_dict[key] = Singleton_arger()[key]
        
    def setup(self):
        #main_args = Singleton_arger()['main']
        self.env = gym.make(self.args_dict['env'])
        
        if self.args_dict['resume']:
            pass
        else:
            if self.args_dict['rand_seed'] >= 0 :
                if self.args_dict['with_cuda']:
                    torch.backends.cudnn.deterministic = True
                    torch.cuda.manual_seed_all(self.args_dict['rand_seed'])
                torch.manual_seed(self.args_dict['rand_seed'])
                np.random.seed(self.args_dict['rand_seed'])
                
        
        Singleton_logger.setup(self.args_dict['result_dir'],multi_process = self.args_dict['multi_process'])

        Singleton_evaluator.setup(self.args_dict['env'], logger = Singleton_logger, num_episodes = 10, model_dir = self.args_dict['result_dir'], multi_process = self.args_dict['multi_process'], visualize = False, rand_seed = self.args_dict['rand_seed'])

        nb_actions = self.env.action_space.shape[0]
        nb_states = self.env.observation_space.shape[0]
        
        
        if self.args_dict['action_noise']:
            from AN_ddpg import action_noise_DDPG
            self.agent = action_noise_DDPG()
        elif self.args_dict['parameter_noise']:
            from PN_ddpg import parameter_noise_DDPG
            self.agent = parameter_noise_DDPG()
        elif self.args_dict['SGLD_mode'] > 0:
            from SGLD_ddpg import SGLD_DDPG
            self.agent = SGLD_DDPG()
        else:
            from ddpg import DDPG
            self.agent = DDPG()
            
        self.agent.setup(nb_states, nb_actions)
        self.args_dict['action_scale'] = (self.env.action_space.high - self.env.action_space.low)/2.0
        self.args_dict['action_bias'] = (self.env.action_space.high + self.env.action_space.low)/2.0
        self.args_dict['action_space'] = self.env.action_space.shape[0]
        self.state_dict = {}
        self.reset()
        
    def reset(self): 
        self.state_dict['current_epoch'] =0
        self.state_dict['last_episode_length'] = 0
        self.state_dict['current_episode_length'] = 0
        self.state_dict['current_episode_reward'] = 0
        self.state_dict['last_episode_reward'] = 0.
        self.state_dict['total_step'] = 0
        

        if self.args_dict['rand_seed'] >= 0:
            self.env.seed(np.random.randint(4294967295))
        self.state_dict['last_observation'] = deepcopy(self.env.reset())
        self.agent.reset_noise()
        
    def warmup(self):
        for t_warmup in range(self.args_dict['nb_warmup_steps']):
            #pick action by actor randomly
            self.apply_action(np.random.uniform(-1.,1.,self.args_dict['action_space']))
                
    def train(self):
        while self.state_dict['current_epoch'] < self.args_dict['nb_epoch']:
            #apply hyperparameter decay
            self.agent.before_epoch()
            self.agent.apply_lr_decay()
            for cycle in range(self.args_dict['nb_cycles_per_epoch']):
                self.agent.before_cycle()
                
                cl_mean,al_mean = self.apply_train()
                
                for t_rollout in range(self.args_dict['nb_rollout_steps']):
                    #pick action by actor in state "last_observation"
                    self.apply_action(self.agent.select_action(s_t = [self.state_dict['last_observation']], apply_noise = True))
                    self.state_dict['total_step'] += 1
                #End Rollout
                
                #trigger log events
                last_error = self.agent.calc_last_error()
                Singleton_logger.trigger_log('last_error', last_error,self.state_dict['total_step'])
                Singleton_logger.trigger_log('train_episode_length', self.state_dict['last_episode_length'],self.state_dict['total_step'])
                Singleton_logger.trigger_log('train_episode_reward', self.state_dict['last_episode_reward'],self.state_dict['total_step'])
                Singleton_logger.trigger_log('critic_loss_mean', cl_mean, self.state_dict['total_step'])
                Singleton_logger.trigger_log('actor_loss_mean', al_mean, self.state_dict['total_step'])
                
            #End Cycle 
            #save agent to disk
            self.agent.save_model(self.args_dict['result_dir'])
            
            #trigger evaluation and log_save
            Singleton_evaluator.trigger_load_from_file(actor_dir = self.args_dict['result_dir'])
            Singleton_evaluator.trigger_eval_process(total_cycle = self.state_dict['total_step'])
            Singleton_logger.trigger_save()
            self.state_dict['current_epoch']+=1
        #End Epoch
        
    def apply_train(self):
        #update agent for nb_train_steps times
        cl_list = []
        al_list = []
        if self.args_dict['train_mode'] == 0:
            for t_train in range(self.args_dict['nb_train_steps']):
                cl = self.agent.update_critic()
                al = self.agent.update_actor()
                self.agent.update_critic_target()
                self.agent.update_actor_target()
                cl_list.append(cl)
                al_list.append(al)
        elif self.args_dict['train_mode'] == 1:
            for t_train in range(self.args_dict['nb_train_steps']):
                cl = self.agent.update_critic()
                cl_list.append(cl)
            for t_train in range(self.args_dict['nb_train_steps']):
                al = self.agent.update_actor()
                al_list.append(al)
            self.agent.update_critic_target(soft_update = False)
            self.agent.update_actor_target (soft_update = False)
        return np.mean(cl_list),np.mean(al_list)
        
    def apply_action(self, action):
        #apply the action to environment and get next state, reawrd and other information
        obs, reward, done, time_done, info = self.env.step(action * self.args_dict['action_scale'] + self.args_dict['action_bias'])
        self.state_dict['current_episode_reward'] += reward
        self.state_dict['current_episode_length'] += 1
        obs = deepcopy(obs)
        
        #store the transition into agent's replay buffer and update last observation
        self.agent.store_transition(self.state_dict['last_observation'], action,np.array([reward,]), obs, np.array([done,],dtype = np.float32))
        self.state_dict['last_observation'] = deepcopy(obs)
        
        #if current episode is done
        if done or time_done:
            if self.args_dict['rand_seed'] >= 0:
                self.env.seed(np.random.randint(4294967295))
            self.state_dict['last_observation'] = deepcopy(self.env.reset())
            self.agent.reset_noise()
            self.state_dict['last_episode_reward'] = self.state_dict['current_episode_reward']
            self.state_dict['last_episode_length'] = self.state_dict['current_episode_length']
            self.state_dict['current_episode_reward'] = 0.
            self.state_dict['current_episode_length'] = 0
    def dump(self):
        pass
    def __del__(self):
        Singleton_evaluator.trigger_close()
        Singleton_logger.trigger_close()

if __name__ == "__main__":
    trainer = DDPG_trainer()
    trainer.setup()
    trainer.warmup()
    trainer.train()