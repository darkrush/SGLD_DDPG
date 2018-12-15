import torch

class Memory(object):
    def __init__(self, limit, action_shape, observation_shape, with_cuda):
        self.args_dict = {'limit': limit, 'with_cuda':with_cuda}
        self.state_dict = {}
        
        self.state_dict['_next_entry'] = 0
        self.state_dict['_nb_entries'] = 0
        
        self.data_buffer = {}
        self.data_buffer['obs0'      ] = torch.zeros((self.args_dict['limit'],) + observation_shape)
        self.data_buffer['obs1'      ] = torch.zeros((self.args_dict['limit'],) + observation_shape)
        self.data_buffer['actions'   ] = torch.zeros((self.args_dict['limit'],) + action_shape     )
        self.data_buffer['rewards'   ] = torch.zeros((self.args_dict['limit'],1)                   )
        self.data_buffer['terminals1'] = torch.zeros((self.args_dict['limit'],1)                   )
        if self.args_dict['with_cuda']:
            for key,value in self.data_buffer.items():
                self.data_buffer[key] = self.data_buffer[key].cuda()
        
    def __getitem(self, idx):
        return {key: value[idx] for key,value in self.data_buffer.items()}
    
    def sample_last(self, batch_size):
        batch_idxs = torch.arange(self.state_dict['_next_entry'] - batch_size ,self.state_dict['_next_entry'])%self.state_dict['_nb_entries']
        if self.args_dict['with_cuda']:
            batch_idxs = batch_idxs.cuda()
        return {key: torch.index_select(value,0,batch_idxs) for key,value in self.data_buffer.items()}
    
    
    def sample(self, batch_size):
        batch_idxs = torch.randint(0,self.state_dict['_nb_entries'], (batch_size,),dtype = torch.long)
        if self.args_dict['with_cuda']:
            batch_idxs = batch_idxs.cuda()
        return {key: torch.index_select(value,0,batch_idxs) for key,value in self.data_buffer.items()}
    
    @property
    def nb_entries(self):
        return self.state_dict['_nb_entries']
    
    def reset(self):
        self.state_dict['_next_entry'] = 0
        self.state_dict['_nb_entries'] = 0
        
    def append(self, obs0, action, reward, obs1, terminal1):
        self.data_buffer['obs0'][self.state_dict['_next_entry']] = torch.as_tensor(obs0)
        self.data_buffer['obs1'][self.state_dict['_next_entry']] = torch.as_tensor(obs1)
        self.data_buffer['actions'][self.state_dict['_next_entry']] = torch.as_tensor(action)
        self.data_buffer['rewards'][self.state_dict['_next_entry']] = torch.as_tensor(reward)
        self.data_buffer['terminals1'][self.state_dict['_next_entry']] = torch.as_tensor(terminal1)
        
        if self.state_dict['_nb_entries'] < self.args_dict['limit']:
            self.state_dict['_nb_entries'] += 1
            
        self.state_dict['_next_entry'] = (self.state_dict['_next_entry'] + 1)%self.args_dict['limit']