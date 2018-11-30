import numpy as np
from copy import deepcopy

class DDPG_trainer(object):
    def __init__(self, nb_epoch, nb_cycles_per_epoch, nb_rollout_steps, nb_train_steps, nb_warmup_steps, train_mode = 0):
        self.nb_epoch = nb_epoch
        self.nb_cycles_per_epoch = nb_cycles_per_epoch
        self.nb_rollout_steps = nb_rollout_steps
        self.nb_train_steps = nb_train_steps
        self.nb_warmup_steps = nb_warmup_steps
        self.train_mode = train_mode
        
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
        self.last_episode_length = 0
        self.current_episode_length = 0
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
                self.last_episode_length = self.current_episode_length *0.1 + self.last_episode_length *0.9
                self.current_episode_reward = 0.
                self.current_episode_length = 0
                
    def train(self):
        self.agent.append_agent()
        for epoch in range(self.nb_epoch):
            
            for cycle in range(self.nb_cycles_per_epoch):
                self.total_cycle += 1
                self.agent.pick_agent()
                
                for t_rollout in range(self.nb_rollout_steps):
                
                    #pick action by actor in state "last_observation"
                    action = self.agent.select_action(s_t = [self.last_observation], if_noise = True)
                    
                    #apply the action to environment and get next state, reawrd and other information
                    obs, reward, done, time_done, info = self.env.step(action * self.action_scale + self.action_bias)
                    self.current_episode_reward += reward
                    self.current_episode_length += 1
                    obs = deepcopy(obs)
                    
                    #store the transition into agent's replay buffer and update last observation
                    self.agent.store_transition(self.last_observation, action, np.array([reward,]), obs, np.array([done,],dtype = np.float32))
                    self.last_observation = deepcopy(obs)
                    
                    #if current episode is done
                    if done or time_done:
                        self.last_observation = deepcopy(self.env.reset())
                        self.agent.reset_noise()
                        self.last_episode_reward = self.current_episode_reward
                        self.last_episode_length = self.current_episode_length *0.1 + self.last_episode_length *0.9
                        self.current_episode_reward = 0.
                        self.current_episode_length = 0
                
                #update agent for nb_train_steps times
                self.agent.update_num_pseudo_batches()
                if self.train_mode == 0:
                    cl_list = []
                    al_list = []
                    self.agent.adapt_param_noise()
                    for t_train in range(self.nb_train_steps):
                    
                        cl = self.agent.update_critic()
                        al = self.agent.update_actor()
                        self.agent.update_critic_target()
                        self.agent.update_actor_target()
                        
                        cl_list.append(cl)
                        al_list.append(al)
                    al_mean = np.mean(al_list)
                    cl_mean = np.mean(cl_list)
                #trigger log events
                last_error = self.agent.calc_last_error()
                self.logger.trigger_log('last_error', last_error,self.total_cycle)
                self.logger.trigger_log('train_episode_length', self.last_episode_length,self.total_cycle)
                self.logger.trigger_log('train_episode_reward', self.last_episode_reward,self.total_cycle)
                self.logger.trigger_log('actor_loss_mean', al_mean, self.total_cycle)
                self.logger.trigger_log('critic_loss_mean', cl_mean, self.total_cycle)
                self.agent.append_agent()    
            #apply hyperparameter decay
            self.agent.apply_noise_decay()
            self.agent.apply_lr_decay()
            
            #save agent to disk
            self.agent.save_model(self.logger.get_dir())
            
            #trigger evaluation and log_save
            self.evaluator.trigger_load_from_file(actor_dir = self.logger.get_dir())
            self.evaluator.trigger_eval_process(total_cycle = self.total_cycle)
            self.logger.trigger_save()