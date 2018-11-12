import io
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam,SGD


from model_pool import Model_pool
from memory import Memory
from sgld import SGLD

class DDPG(object):
    def __init__(self, actor_lr, critic_lr, lr_decay,
                 l2_critic, batch_size, discount, tau,
                 action_noise, noise_decay,
                 parameter_noise,
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
        self.parameter_noise = parameter_noise
        self.SGLD_mode = SGLD_mode
        self.num_pseudo_batches = num_pseudo_batches
        
        exploration_method = (self.action_noise is not None) + (self.parameter_noise is not None) + (self.SGLD_mode is not 0)
        
        #pool_mode 0: no model pool, 1: actor only, 2: critic only, 3: both A&C
        self.pool_mode = pool_mode
        self.pool_size = pool_size

        self.with_cuda = with_cuda
        
    def setup(self, actor, critic, memory):
        self.actor         = copy.deepcopy(actor)
        self.actor_target  = copy.deepcopy(actor)
        self.noise_actor = None
        self.measure_actor = None
        if self.SGLD_mode is not 0:
            self.noise_actor   = copy.deepcopy(actor)
        if self.parameter_noise is not None:
            self.noise_actor   = copy.deepcopy(actor)
            self.measure_actor = copy.deepcopy(actor)
            
        self.critic        = copy.deepcopy(critic)
        self.critic_target = copy.deepcopy(critic)
        
        if self.with_cuda:
            for net in (self.actor, self.actor_target, self.noise_actor, self.measure_actor, self.critic, self.critic_target):
                if net is not None:
                    net.cuda()
                    
            
        if (self.SGLD_mode == 1)or(self.SGLD_mode == 3):
            self.actor_optim  = SGLD(self.actor.parameters(),
                                     lr=self.actor_lr,
                                     num_pseudo_batches = self.num_pseudo_batches,
                                     num_burn_in_steps = 1000)
        else :
            self.actor_optim  = Adam(self.actor.parameters(), lr=self.actor_lr)
        
        if (self.SGLD_mode == 2)or(self.SGLD_mode == 3):
            p_groups = [{'params': [param,],
                         'noise_switch': True if ('LN' not in name) else False,
                         'weight_decay': self.l2_critic if ('weight' in name) and ('LN' not in name) else 0
                        } for name,param in self.critic.named_parameters() ]
            self.critic_optim  = SGLD(params = p_groups,
                                      lr = self.critic_lr,
                                      num_pseudo_batches = self.num_pseudo_batches,
                                      num_burn_in_steps = 1000,
                                      weight_decay = self.l2_critic)
        else:
            p_groups = [{'params': [param,],
                         'weight_decay': self.l2_critic if ('weight' in name) and ('LN' not in name) else 0
                        } for name,param in self.critic.named_parameters() ]
            self.critic_optim  = Adam(params = p_groups, lr=self.critic_lr, weight_decay = self.l2_critic)
        
        self.memory = memory
        
        if self.pool_mode>0:
            assert self.pool_size>0
            self.agent_pool = Model_pool(self.pool_size)
                        
    def reset_noise(self):
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.parameter_noise is not None:
            for target_param, param in zip(self.noise_actor.parameters(), self.actor.named_parameters()):
                name, param = param
                if 'LN' not in name:
                    target_param.data.copy_(param.data + torch.normal(mean=torch.zeros_like(param),std=torch.full_like(param,self.parameter_noise.current_stddev)))
                else:
                    target_param.data.copy_(param.data)
        if self.SGLD_mode is not 0:
            self.noise_actor.load_state_dict(copy.deepcopy(self.actor.state_dict()))
            
            
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
        
            target_q_batch = batch['rewards'] + self.discount*(1-batch['terminals1'])*next_q_values
        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([tensor_obs0, batch['actions']])
        value_loss = nn.functional.mse_loss(q_batch, target_q_batch)
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
    
    def update_num_pseudo_batches(self):
        if isinstance(self.actor_optim,SGLD):
            self.actor_optim.param_groups[0]['num_pseudo_batches'] = self.memory.nb_entries
        if isinstance(self.critic_optim,SGLD):
            self.critic_optim.param_groups[0]['num_pseudo_batches'] = self.memory.nb_entries
            

    def apply_lr_decay(self):
        if self.lr_decay > 0:
            self.lr_coef = self.lr_decay*self.lr_coef/(self.lr_coef+self.lr_decay)
            self.critic_optim.param_groups[0]['lr'] = self.critic_lr * self.lr_coef
            self.actor_optim.param_groups[0]['lr'] = self.actor_lr * self.lr_coef
        
    def apply_noise_decay(self):
        if self.noise_decay > 0:
            self.noise_coef = self.noise_decay*self.noise_coef/(self.noise_coef+self.noise_decay)
            
    def adapt_param_noise(self):
        if self.parameter_noise is None:
            return
        
        for target_param, param in zip(self.measure_actor.parameters(), self.actor.named_parameters()):
            name, param = param
            if 'LN' not in name:
                target_param.data.copy_(param.data + torch.normal(mean=torch.zeros_like(param),std=torch.full_like(param,self.parameter_noise.current_stddev)))
            else:
                target_param.data.copy_(param.data)
        
        batch = self.memory.sample(self.batch_size)
        tensor_obs0 = batch['obs0']
        with torch.no_grad():
            distance = torch.mean(torch.sqrt(torch.sum((self.actor(tensor_obs0)- self.measure_actor(tensor_obs0))**2,1)))
        self.parameter_noise.adapt(distance)
        
        
    def select_action(self, s_t, if_noise = True):
        s_t = torch.tensor(s_t,dtype = torch.float32,requires_grad = False)
        if self.with_cuda:
            s_t = s_t.cuda()
        with torch.no_grad():
            if not if_noise :
                action = self.actor(s_t).cpu().numpy().squeeze(0)
            else:
                if (self.parameter_noise is not None) or (self.SGLD_mode is not 0):
                    action = self.noise_actor(s_t).cpu().numpy().squeeze(0)
                else:
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
        
    def append_agent(self):
        if self.pool_mode is 0:
            return
        model_dict = {}
        if (self.pool_mode==1) or (self.pool_mode==3):
            model_dict['actor'] = copy.deepcopy(self.actor.state_dict())
            #model_dict['actor_target'] = copy.deepcopy(self.actor_target.state_dict())
        if (self.pool_mode==2) or (self.pool_mode==3):
            model_dict['critic'] = copy.deepcopy(self.critic.state_dict())
            #model_dict['critic_target'] = copy.deepcopy(self.critic_target.state_dict())
        self.agent_pool.model_append(model_dict)

    def pick_agent(self, id = None):
        # id: -1:last agent; None: random agent; 0~pool_size-1: specific agent
        if self.pool_mode is 0:
            return
        model_dict = self.agent_pool.get_model(id)
        if (self.pool_mode==1) or (self.pool_mode==3):
            self.actor.load_state_dict(model_dict['actor'])
            #self.actor_target.load_state_dict(model_dict['actor_target'])
        if (self.pool_mode==2) or (self.pool_mode==3):
            self.critic.load_state_dict(model_dict['critic'])
            #self.critic_target.load_state_dict(model_dict['critic_target'])

    
