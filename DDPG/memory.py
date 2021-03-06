import torch

class Memory(object):
    def __init__(self, limit, action_shape, observation_shape, with_cuda):
        self.limit = limit
        self._next_entry = 0
        self._nb_entries = 0
        self.with_cuda = with_cuda
        
        self.data_buffer = {}
        self.data_buffer['obs0'      ] = torch.zeros((limit,) + observation_shape)
        self.data_buffer['obs1'      ] = torch.zeros((limit,) + observation_shape)
        self.data_buffer['actions'   ] = torch.zeros((limit,) + action_shape     )
        self.data_buffer['rewards'   ] = torch.zeros((limit,1)                   )
        self.data_buffer['terminals1'] = torch.zeros((limit,1)                   )
        if self.with_cuda:
            for key,value in self.data_buffer.items():
                self.data_buffer[key] = self.data_buffer[key].cuda()

    def __getitem(self, idx):
        return {key: value[idx] for key,value in self.data_buffer.items()}
    
    def sample_last(self, batch_size):
        batch_idxs = torch.arange(self._next_entry - batch_size ,self._next_entry)%self._nb_entries
        if self.with_cuda:
            batch_idxs = batch_idxs.cuda()
        return {key: torch.index_select(value,0,batch_idxs) for key,value in self.data_buffer.items()}
    
    
    def sample(self, batch_size):
        batch_idxs = torch.randint(0,self._nb_entries, (batch_size,),dtype = torch.long)
        if self.with_cuda:
            batch_idxs = batch_idxs.cuda()
        return {key: torch.index_select(value,0,batch_idxs) for key,value in self.data_buffer.items()}
    
    @property
    def nb_entries(self):
        return self._nb_entries
    
    def reset(self):
        self._next_entry = 0
        self._nb_entries = 0
        
    def append(self, obs0, action, reward, obs1, terminal1):
        self.data_buffer['obs0'][self._next_entry] = torch.as_tensor(obs0)
        self.data_buffer['obs1'][self._next_entry] = torch.as_tensor(obs1)
        self.data_buffer['actions'][self._next_entry] = torch.as_tensor(action)
        self.data_buffer['rewards'][self._next_entry] = torch.as_tensor(reward)
        self.data_buffer['terminals1'][self._next_entry] = torch.as_tensor(terminal1)
        
        if self._nb_entries < self.limit:
            self._nb_entries += 1
            
        self._next_entry = (self._next_entry + 1)%self.limit