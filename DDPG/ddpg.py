import io
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from arguments import Singleton_arger

from model import Actor,Critic
from memory import Memory

class DDPG(object):
    def __init__(self):
        self.args_dict = {}
        for key in ('actor_lr','critic_lr','lr_decay','l2_critic','batch_size',
                    'discount','tau','with_cuda','buffer_size','hidden1','hidden2','layer_norm'):
            self.args_dict[key] = Singleton_arger()[key]
            
    def setup(self, nb_states, nb_actions):
        self.state_dict = {}
        self.state_dict['lr_coef'] = 1
        
        actor  = Actor (nb_states, nb_actions, hidden1 = self.args_dict['hidden1'], hidden2 = self.args_dict['hidden2'] , layer_norm = self.args_dict['layer_norm'])
        critic = Critic(nb_states, nb_actions, hidden1 = self.args_dict['hidden1'], hidden2 = self.args_dict['hidden2'] , layer_norm = self.args_dict['layer_norm'])
        
        self.model_dict = {}
        self.model_dict['actor'] = copy.deepcopy(actor)
        self.model_dict['actor_target'] = copy.deepcopy(actor)
        self.model_dict['critic'] = copy.deepcopy(critic)
        self.model_dict['critic_target'] = copy.deepcopy(critic)
        
        self.memory = Memory(int(self.args_dict['buffer_size']), (nb_actions,), (nb_states,), self.args_dict['with_cuda'])
        
        if self.args_dict['with_cuda']:
            for net in self.model_dict.values():
                if net is not None:
                    net.cuda()
        
        p_groups = [{'params': [param,],
                     'weight_decay': self.args_dict['l2_critic'] if ('weight' in name) and ('LN' not in name) else 0
                    } for name,param in self.model_dict['critic'].named_parameters() ]
        self.optim_dict = {}
        self.optim_dict['critic_optim'] = Adam(params = p_groups, lr=self.args_dict['critic_lr'], weight_decay = self.args_dict['l2_critic'])
        self.optim_dict['actor_optim'] = Adam(self.model_dict['actor'].parameters(), lr=self.args_dict['actor_lr'])

    def reset_noise(self):
        pass
        
    def before_epoch(self):
        pass
    
    def before_cycle(self):
        pass
        
    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):
        s_t = torch.tensor(s_t,dtype = torch.float32,requires_grad = False)
        if self.args_dict['with_cuda']:
            s_t = s_t.cuda()
        self.memory.append(s_t, a_t, r_t, s_t1, done_t)

    def update_critic(self, batch = None, pass_batch = False):
        # Sample batch
        if batch is None:
            batch = self.memory.sample(self.args_dict['batch_size'])
        assert batch is not None
        tensor_obs0 = batch['obs0']
        tensor_obs1 = batch['obs1']
        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.model_dict['critic_target']([
                tensor_obs1,
                self.model_dict['actor_target'](tensor_obs1),
            ])
        
            target_q_batch = batch['rewards'] + self.args_dict['discount']*(1-batch['terminals1'])*next_q_values
        # Critic update
        self.model_dict['critic'].zero_grad()

        q_batch = self.model_dict['critic']([tensor_obs0, batch['actions']])
        value_loss = nn.functional.mse_loss(q_batch, target_q_batch)
        value_loss.backward()
        self.optim_dict['critic_optim'].step()
        if pass_batch :
            return value_loss.item(), batch
        else:
            return value_loss.item()
        
    def update_actor(self, batch = None, pass_batch = False):
        if batch is None:
            batch = self.memory.sample(self.args_dict['batch_size'])
        assert batch is not None  
        tensor_obs0 = batch['obs0']
        # Actor update
        self.model_dict['actor'].zero_grad()

        policy_loss = -self.model_dict['critic']([
            tensor_obs0,
            self.model_dict['actor'](tensor_obs0)
        ])
        
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.optim_dict['actor_optim'].step()  
        if pass_batch :
            return policy_loss.item(), batch
        else:
            return policy_loss.item()

    def update_critic_target(self,soft_update = True):
        for target_param, param in zip(self.model_dict['critic_target'].parameters(), self.model_dict['critic'].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.args_dict['tau']) + param.data * self.args_dict['tau'] \
                                    if soft_update else param.data)

    def update_actor_target(self,soft_update = True):
        for target_param, param in zip(self.model_dict['actor_target'].parameters(), self.model_dict['actor'].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.args_dict['tau']) + param.data * self.args_dict['tau'] \
                                    if soft_update else param.data)
                                    
    def apply_lr_decay(self):
        if self.args_dict['lr_decay'] > 0:
            self.state_dict['lr_coef'] = self.args_dict['lr_decay']*self.state_dict['lr_coef']/(self.state_dict['lr_coef']+self.args_dict['lr_decay'])
            for (opt,base_lr) in ((self.optim_dict['actor_optim'],self.args_dict['actor_lr']),(self.optim_dict['critic_optim'],self.args_dict['critic_lr'])):
                for group in opt.param_groups:
                    group['lr'] = base_lr * self.state_dict['lr_coef']
            
    def calc_last_error(self):
        # Sample batch
        batch = self.memory.sample_last(self.args_dict['batch_size'])
        tensor_obs0 = batch['obs0']
        tensor_obs1 = batch['obs1']
        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.model_dict['critic_target']([
                tensor_obs1,
                self.model_dict['actor_target'](tensor_obs1),
            ])
            target_q_batch = batch['rewards'] + self.args_dict['discount']*(1-batch['terminals1'])*next_q_values
            q_batch = self.model_dict['critic_target']([tensor_obs0, batch['actions']])
            value_loss = nn.functional.mse_loss(q_batch, target_q_batch)
        return value_loss.item()
        
    def select_action(self, s_t, apply_noise):
        s_t = torch.tensor(s_t,dtype = torch.float32,requires_grad = False)
        if self.args_dict['with_cuda']:
            s_t = s_t.cuda()
        with torch.no_grad():
            action = self.model_dict['actor'](s_t).cpu().numpy().squeeze(0)
        action = np.clip(action, -1., 1.)
        return action
        
    def load_weights(self, output): 
        self.model_dict['actor']  = torch.load('{}/actor.pkl'.format(output) )
        self.model_dict['critic'] = torch.load('{}/critic.pkl'.format(output))
            
    def save_model(self, output):
        torch.save(self.model_dict['actor'] ,'{}/actor.pkl'.format(output) )
        torch.save(self.model_dict['critic'],'{}/critic.pkl'.format(output))
            
    def get_actor_buffer(self):
        actor_buffer = io.BytesIO()
        torch.save(self.model_dict['actor'], actor_buffer)
        return actor_buffer