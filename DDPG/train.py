import numpy as np
from copy import deepcopy

class DDPG_trainer(object):
    def __init__(self, nb_epoch, nb_cycles_per_epoch, nb_rollout_steps, nb_train_steps, nb_warmup_steps):
        self.nb_epoch = nb_epoch
        self.nb_cycles_per_epoch = nb_cycles_per_epoch
        self.nb_rollout_steps = nb_rollout_steps
        self.nb_train_steps = nb_train_steps
        self.nb_warmup_steps = nb_warmup_steps
        
    def setup(self, env, agent, evaluator, logger):
        self.env = env
        self.agent = agent
        self.evaluator = evaluator
        self.logger = logger
        self.action_scale = (env.action_space.high - env.action_space.low)/2.0
        self.action_bias = (env.action_space.high + env.action_space.low)/2.0
        self.action_space = self.env.action_space.shape[0]
        self.reset()
        
    def reset(self):
        self.current_episode_reward = 0.
        self.last_episode_reward = 0.
        self.total_cycle = 0
        self.last_observation = deepcopy(self.env.reset())
        self.agent.reset_noise()
        
    def warmup(self):
        for t_warmup in range(self.nb_warmup_steps):
        
            #pick action by actor randomly
            action = np.random.uniform(-1.,1.,self.action_space)
            #apply the action to environment and get next state, reawrd and other information
            obs, reward, done, info = self.env.step(action * self.action_scale + self.action_bias)
            self.current_episode_reward += reward
            obs = deepcopy(obs)
            
            #store the transition into agent's replay buffer and update last observation
            self.agent.store_transition(self.last_observation, action,np.array([reward,]), obs, np.array([done,],dtype = np.float32))
            self.last_observation = deepcopy(obs)
                                
            #if current episode is done
            if done:
                self.last_observation = deepcopy(self.env.reset())
                self.agent.reset_noise()
                self.last_episode_reward = self.current_episode_reward
                self.current_episode_reward = 0.
                
    def train(self):
        for epoch in range(self.nb_epoch):
            for cycle in range(self.nb_cycles_per_epoch):
                self.total_cycle += 1
                
                for t_rollout in range(self.nb_rollout_steps):
                
                    #pick action by actor in state "last_observation"
                    action = self.agent.select_action(s_t = [self.last_observation], if_noise = True)
                    
                    #apply the action to environment and get next state, reawrd and other information
                    obs, reward, done, info = self.env.step(action * self.action_scale + self.action_bias)
                    self.current_episode_reward += reward
                    obs = deepcopy(obs)
                    
                    #store the transition into agent's replay buffer and update last observation
                    self.agent.store_transition(self.last_observation, action, np.array([reward,]), obs, np.array([done,],dtype = np.float32))
                    self.last_observation = deepcopy(obs)
                    
                    #if current episode is done
                    if done:
                        self.last_observation = deepcopy(self.env.reset())
                        self.agent.reset_noise()
                        self.last_episode_reward = self.current_episode_reward
                        self.current_episode_reward = 0.
                
                #update agent for nb_train_steps times
                cl_list = []
                al_list = []
                for t_train in range(self.nb_train_steps):
                    cl,al = self.agent.update()
                    cl_list.append(cl)
                    al_list.append(al)
                al_mean = np.mean(al_list)
                cl_mean = np.mean(cl_list)
                
                #trigger log events
                self.logger.trigger_log('train_episode_reward', self.last_episode_reward,self.total_cycle)
                self.logger.trigger_log('actor_loss_mean', al_mean, self.total_cycle)
                self.logger.trigger_log('critic_loss_mean', cl_mean, self.total_cycle)
                
            #apply hyperparameter decay
            self.agent.apply_noise_decay()
            self.agent.apply_lr_decay()
            
            #save agent to disk
            self.agent.save_model(self.logger.get_dir())
            
            #trigger evaluation and log_save
            self.evaluator.trigger_load_from_file(actor_dir = self.logger.get_dir())
            self.evaluator.trigger_eval_process(total_cycle = self.total_cycle)
            self.logger.trigger_save()