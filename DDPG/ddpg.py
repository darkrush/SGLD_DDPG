import io
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam,SGD


#from agent_pool import Agent_pool
from memory import Memory
from sgld import SGLD

class DDPG(object):
    def __init__(self, actor_lr, critic_lr, lr_decay,
                 l2_critic, batch_size, discount, tau,
                 action_noise, noise_decay, 
                 SGLD_mode, num_pseudo_batches, 
                 pool_mode, pool_size, with_cuda):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.lr_decay = lr_decay
        self.l2_critic = l2_critic
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.lr_coef = 1
        self.noise_coef = 1
        
        self.action_noise = action_noise
        self.noise_decay = noise_decay
        
        self.SGLD_mode = SGLD_mode
        self.num_pseudo_batches = num_pseudo_batches
        
        self.pool_mode = pool_mode
        self.pool_size = pool_size
        
        self.with_cuda = with_cuda
        
    def setup(self, actor, critic, memory):
        #if self.pool_size>0:
        #    self.agent_pool = Agent_pool(self.pool_size)
            
        self.memory = memory
    
        self.actor         = copy.deepcopy(actor)
        self.actor_target  = copy.deepcopy(actor)
        self.critic        = copy.deepcopy(critic)
        self.critic_target = copy.deepcopy(critic)
        if self.with_cuda:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()
        if (self.SGLD_mode == 1)or(self.SGLD_mode == 3):
            self.actor_optim  = SGLD(self.actor.parameters(),
                                     lr=self.actor_lr/self.num_pseudo_batches,
                                     num_pseudo_batches = self.num_pseudo_batches,
                                     num_burn_in_steps = 1000)
        else :
            self.actor_optim  = Adam(self.actor.parameters(), lr=self.actor_lr)
        if (self.SGLD_mode == 2)or(self.SGLD_mode == 3):
            self.critic_optim  = SGLD(self.critic.parameters(),
                                      lr=self.critic_lr/self.num_pseudo_batches,
                                      num_pseudo_batches = self.num_pseudo_batches,
                                      num_burn_in_steps = 1000)
        else:
            self.critic_optim  = Adam(self.critic.parameters(), lr=self.critic_lr)
            
    def reset_noise(self):
        if self.action_noise is not None:
            self.action_noise.reset()
            
    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):
        self.memory.append(s_t, a_t, r_t, s_t1, done_t)
        
    def update(self):
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        tensor_obs0 = batch['obs0']
        tensor_obs1 = batch['obs1']

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                tensor_obs1,
                self.actor_target(tensor_obs1),
            ])
        
            target_q_batch = batch['rewards'] + \
                self.discount*(1-batch['terminals1'])*next_q_values
        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([tensor_obs0, batch['actions']])
        value_loss = nn.functional.mse_loss(q_batch, target_q_batch)
        
        l2_coef = torch.tensor(self.l2_critic).cuda()
        l2_reg = torch.tensor(0.).cuda()
        for name,param in self.critic.named_parameters():
            if 'LN' not in name:
                l2_reg += torch.norm(param)
        value_loss += l2_coef*l2_reg
        value_loss.backward()
        self.critic_optim.step()
        
        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            tensor_obs0,
            self.actor(tensor_obs0)
        ])
        
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()
        
        # Target update
        for target,source in ((self.actor_target, self.actor),(self.critic_target, self.critic)):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return value_loss.item(),policy_loss.item()
        

    #TODO SGLD??
    def apply_lr_decay(self):
        if self.lr_decay > 0:
            self.lr_coef = self.lr_decay*self.lr_coef/(self.lr_coef+self.lr_decay)
            self.critic_optim.param_groups[0]['lr'] = self.critic_lr * self.lr_coef
            self.actor_optim.param_groups[0]['lr'] = self.actor_lr * self.lr_coef
        
    def apply_noise_decay(self):
        if self.noise_decay > 0:
            self.noise_coef = self.noise_decay*self.noise_coef/(self.noise_coef+self.noise_decay)
    
        
    def select_action(self, s_t, if_noise = True):
        s_t = torch.tensor(s_t,dtype = torch.float32,requires_grad = False)
        if self.with_cuda:
            s_t = s_t.cuda()
        with torch.no_grad():
            action = self.actor(s_t).cpu().numpy().squeeze(0)
        if if_noise & (self.action_noise is not None):
            action += max(self.noise_coef, 0)*self.action_noise()
        action = np.clip(action, -1., 1.)
        return action
        
    def load_weights(self, output): 
        self.actor  = torch.load('{}/actor.pkl'.format(output) )
        self.critic = torch.load('{}/critic.pkl'.format(output))
            
    def save_model(self, output):
        torch.save(self.actor ,'{}/actor.pkl'.format(output) )
        torch.save(self.critic,'{}/critic.pkl'.format(output))
        
    def get_actor_buffer(self):
        buffer = io.BytesIO()
        torch.save(self.actor, buffer)
        return buffer

        
    '''
    #TODO recode agent pool
    def append_actor(self):
        self.agent_pool.actor_append(self.actor.state_dict(),self.actor_target.state_dict())
        
    def pick_actor(self):
        actor,actor_target = self.agent_pool.get_actor()
        self.actor.load_state_dict(actor)
        self.actor_target.load_state_dict(actor_target)
    
    def append_critic(self):
        self.agent_pool.critic_append(self.critic.state_dict(),self.critic_target.state_dict())
        
    def pick_critic(self):
        critic,critic_target = self.agent_pool.get_critic()
        self.critic.load_state_dict(critic)
        self.critic_target.load_state_dict(critic_target)
    
    def append_actor_critic(self):
        self.agent_pool.actor_append(self.actor.state_dict(),self.actor_target.state_dict())
        self.agent_pool.critic_append(self.critic.state_dict(),self.critic_target.state_dict())
        
    def pick_actor_critic(self):
        actor,actor_target,critic,critic_target = self.agent_pool.get_agent()
        self.actor.load_state_dict(actor)
        self.actor_target.load_state_dict(actor_target)
        self.critic.load_state_dict(critic)
        self.critic_target.load_state_dict(critic_target)
    
    def append_agent(self):
        if self.pool_mode == 1:
            self.append_actor()
        elif self.pool_mode ==2:
            self.append_critic()
        elif self.pool_mode ==3:
            self.append_actor_critic()

    def pick_agent(self):
        if self.pool_mode == 1:
            self.pick_actor()
        elif self.pool_mode ==2:
            self.pick_critic()
        elif self.pool_mode ==3:
            self.pick_actor_critic()
    '''