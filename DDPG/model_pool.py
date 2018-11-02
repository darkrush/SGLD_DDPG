import torch

class Model_pool(object):
    def __init__(self, size = 10):
        self.size = size
        self.next_id = 0
        self.item_nb = 0
        self.buffer = tuple({} for _ in range(self.size))
        
    def reset(self):
        self.next_id = 0
        self.item_nb = 0
        
    def model_append(self,model_dict):
        self.buffer[self.next_id ].update(model_dict)
        self.next_id = (self.next_id+1) % self.size
        if self.item_nb < self.size:
            self.item_nb = self.item_nb+1
        return self.next_id

    def get_model(self, id = None):
        if id is None:
            id = int(torch.randint(low = 0, high = self.item_nb, size = (1,)).item())
        if id is -1:
            id = (self.next_id-1)%self.size
        assert self.item_nb > id >= 0
        return self.buffer[id]
        

        
        
#test model_pool
if __name__ == "__main__":
    from model import Actor,Critic
    from copy import deepcopy
    pool_size = 10
    append_time = 20
    
    actor  = Actor (3, 3, layer_norm = True)
    critic = Critic(3, 3, layer_norm = True)
    
    target_actor = Actor (3, 3, layer_norm = True)
    target_critic = Critic(3, 3, layer_norm = True)
    
    pool = Model_pool(pool_size)
    assert len(pool.buffer) == pool_size
    
    for i in range(append_time):
        assert pool.next_id == i % pool_size
        assert pool.item_nb == min(i,pool_size)
        
        actor.state_dict()['LN2.bias'].data.copy_(actor.state_dict()['LN2.bias']*0 + i)
        critic.state_dict()['LN2.bias'].data.copy_(critic.state_dict()['LN2.bias']*0 - i)
        
        pool.model_append({'actor':deepcopy(actor),'critic':deepcopy(critic)})
        
        target_actor.load_state_dict(pool.get_model()['actor'].state_dict())
        target_critic.load_state_dict(pool.get_model()['critic'].state_dict())

        assert target_actor.state_dict()['LN2.bias'][0].item() == i
        assert target_critic.state_dict()['LN2.bias'][0].item() == -i