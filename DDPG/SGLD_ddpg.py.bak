import copy
import numpy as np
import torch
from sgld import SGLD
from ddpg import DDPG
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
        if self.with_cuda:
            for net in (self.rollout_actor,):
                if net is not None:
                    net.cuda()
                    
            
        if (self.SGLD_mode == 1)or(self.SGLD_mode == 3):
            self.actor_optim  = SGLD(self.actor.parameters(),
                                     lr=self.actor_lr,
                                     num_pseudo_batches = self.num_pseudo_batches,
                                     num_burn_in_steps = 1000)
        if (self.SGLD_mode == 2)or(self.SGLD_mode == 3):
            p_groups = [{'params': [param,],
                         'noise_switch': self.SGLD_noise and (True if ('LN' not in name) else False),
                         'weight_decay': self.l2_critic if ('weight' in name) and ('LN' not in name) else 0
                        } for name,param in self.critic.named_parameters() ]
            self.critic_optim  = SGLD(params = p_groups,
                                      lr = self.critic_lr,
                                      num_pseudo_batches = self.num_pseudo_batches,
                                      num_burn_in_steps = 1000)
                                      
    def reset_noise(self):
        if self.SGLD_mode is not 0:
            self.rollout_actor.load_state_dict(copy.deepcopy(self.actor.state_dict()))
    
    def before_cycle(self):
        self.update_num_pseudo_batches()
    
    def update_num_pseudo_batches(self):
        if self.num_pseudo_batches is not 0:
            return
        for opt in (self.actor_optim,self.critic_optim):
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