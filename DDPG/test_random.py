import gym
import numpy as np
import torch
import random
import copy
from snapshot_wrap import WithSnapshots
rand_seed = 314
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(rand_seed)
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)


env = gym.make('Reacher-v2')
env.seed(rand_seed)

action_scale = (env.action_space.high - env.action_space.low)/2.0
action_bias = (env.action_space.high + env.action_space.low)/2.0
action_space = env.action_space.shape[0]

#################################################################
np_random1 = np.random.rand(action_space)
torch_random1 = torch.rand(action_space)
torch_cuda_random1 = torch.rand(action_space,device = torch.device('cuda'))
np_random2 = np.random.rand(action_space)
torch_random2 = torch.rand(action_space)
torch_cuda_random2 = torch.rand(action_space,device = torch.device('cuda'))



obs = env.reset()
obs1, reward1, done1, time_done1, info1 = env.step(torch_random1.numpy() * action_scale + action_bias)
obs2, reward2, done2, time_done2, info2 = env.step(torch_random2.numpy() * action_scale + action_bias)
#################################################################
env.seed(rand_seed)
obs = env.reset()
torch_RNG_state = torch.get_rng_state()
torch_CUDA_RNG_state = torch.cuda.get_rng_state()
numpy_RNG_state = np.random.get_state()
random.seed(rand_seed)
#################################################################
np_random1 = np.random.rand(action_space)
torch_random1 = torch.rand(action_space)
torch_cuda_random1 = torch.rand(action_space,device = torch.device('cuda'))
np_random2 = np.random.rand(action_space)
torch_random2 = torch.rand(action_space)
torch_cuda_random2 = torch.rand(action_space,device = torch.device('cuda'))




obs1, reward1, done1, time_done1, info1 = env.step(torch_random1.numpy() * action_scale + action_bias)
obs2, reward2, done2, time_done2, info2 = env.step(torch_random2.numpy() * action_scale + action_bias)
#################################################################
env.seed(rand_seed)
new_obs = env.reset()
torch.set_rng_state(torch_RNG_state)
torch.cuda.set_rng_state(torch_CUDA_RNG_state)
np.random.set_state(numpy_RNG_state)
random.seed(rand_seed)
#################################################################
new_np_random1 = np.random.rand(action_space)
new_torch_random1 = torch.rand(action_space)
new_torch_cuda_random1 = torch.rand(action_space,device = torch.device('cuda'))
new_np_random2 = np.random.rand(action_space)
new_torch_random2 = torch.rand(action_space)
new_torch_cuda_random2 = torch.rand(action_space,device = torch.device('cuda'))

new_obs1, new_reward1, new_done1, new_time_done1, new_info1 = env.step(new_torch_random1.numpy() * action_scale + action_bias)
new_obs2, new_reward2, new_done2, new_time_done2, new_info2 = env.step(new_torch_random2.numpy() * action_scale + action_bias)

print('##################')
print(np_random1,new_np_random1)
print('##################')
print(torch_random1,new_torch_random1)
print('##################')
print(torch_cuda_random1,new_torch_cuda_random1)
print('##################')
print(np_random2,new_np_random2)
print('##################')
print(torch_random2,new_torch_random2)
print('##################')
print(torch_cuda_random2,new_torch_cuda_random2)
print('##################')
print(obs,new_obs)
print('##################')
print(obs1,new_obs1)
print('##################')
print(obs2,new_obs2)
print('##################')



