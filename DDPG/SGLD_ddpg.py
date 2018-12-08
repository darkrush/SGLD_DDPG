import copy
import numpy as np
import torch
import torch.nn as nn
from sgld import SGLD
from ddpg import DDPG
from torch.optim import Adam
from arguments import Singleton_arger

class SGLD_DDPG(DDPG):
    def __init__(self):
        super(SGLD_DDPG, self).__init__()
        exploration_args = Singleton_arger()['exploration']

        self.SGLD_noise = exploration_args['SGLD_noise']
        self.SGLD_mode = exploration_args['SGLD_mode']
        #self.adapt_pseudo_batches = num_pseudo_batches is 0
        self.num_pseudo_batches = exploration_args['num_pseudo_batches']
        
    def setup(self, nb_states, nb_actions):
        super(SGLD_DDPG, self).setup(nb_states, nb_actions)
        self.rollout_actor   = copy.deepcopy(self.actor)
        self.rollout_critic  = copy.deepcopy(self.critic)
        if self.with_cuda:
            for net in (self.rollout_actor,self.rollout_critic):
                if net is not None:
                    net.cuda()
                    
        self.rollout_actor_optim  = Adam(self.rollout_actor.parameters(), lr=self.actor_lr)
        p_groups = [{'params': [param,],
                     'noise_switch': self.SGLD_noise and (True if ('LN' not in name) else False),
                     'weight_decay': self.l2_critic if ('weight' in name) and ('LN' not in name) else 0
                    } for name,param in self.rollout_critic.named_parameters() ]
        self.rollout_critic_optim  = SGLD(params = p_groups,
                                  lr = self.critic_lr,
                                  num_pseudo_batches = self.num_pseudo_batches,
                                  num_burn_in_steps = 1000)

    def update_rollout_critic(self, batch = None, pass_batch = False):
        # Sample batch
        if batch is None:
            batch = self.memory.sample(self.batch_size)
        assert batch is not None
        tensor_obs0 = batch['obs0']
        tensor_obs1 = batch['obs1']
        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                tensor_obs1,
                self.actor_target(tensor_obs1),
            ])
        
            target_q_batch = batch['rewards'] + self.discount*(1-batch['terminals1'])*next_q_values
        # Critic update
        self.rollout_critic.zero_grad()

        q_batch = self.rollout_critic([tensor_obs0, batch['actions']])
        value_loss = nn.functional.mse_loss(q_batch, target_q_batch)
        value_loss.backward()
        self.rollout_critic_optim.step()
        if pass_batch :
            return value_loss.item(), batch
        else:
            return value_loss.item()
        
    def update_rollout_actor(self, batch = None, pass_batch = False):
        if batch is None:
            batch = self.memory.sample(self.batch_size)
        assert batch is not None  
        tensor_obs0 = batch['obs0']
        # Actor update
        self.rollout_actor.zero_grad()

        policy_loss = -self.rollout_critic([
            tensor_obs0,
            self.rollout_actor(tensor_obs0)
        ])
        
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.rollout_actor_optim.step()  
        if pass_batch :
            return policy_loss.item(), batch
        else:
            return policy_loss.item()


    def reset_noise(self):
        if self.memory.nb_entries<self.batch_size:
           return
        for target_param, param in zip(self.rollout_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.rollout_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for _ in range(50):
            self.update_rollout_critic()
        for _ in range(50):
            self.update_rollout_actor()
        
    def before_cycle(self):
        self.update_num_pseudo_batches()
    
    def update_num_pseudo_batches(self):
        if self.num_pseudo_batches is not 0:
            return
        for opt in (self.rollout_actor_optim,self.rollout_critic_optim):
            if isinstance(opt,SGLD):
                for group in opt.param_groups:
                    group['num_pseudo_batches'] = self.memory.nb_entries

    def select_action(self, s_t, apply_noise):
        s_t = torch.tensor(s_t,dtype = torch.float32,requires_grad = False)
        if self.with_cuda:
            s_t = s_t.cuda()
        with torch.no_grad():
            if apply_noise:
                action = self.rollout_actor(s_t).cpu().numpy().squeeze(0)
            else:
                action = self.actor(s_t).cpu().numpy().squeeze(0)
        action = np.clip(action, -1., 1.)
        return action    