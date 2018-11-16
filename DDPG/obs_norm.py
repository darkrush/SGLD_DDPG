import torch
import torch.nn as nn


class Run_Normalizer(nn.Module):
    def __init__(self, size):
        super(Run_Normalizer, self).__init__()
        self.n = torch.nn.Parameter(torch.zeros(size,dtype = torch.float),requires_grad=False)
        self.mean = torch.nn.Parameter(torch.zeros(size,dtype = torch.float),requires_grad=False)
        self.mean_diff = torch.nn.Parameter(torch.zeros(size,dtype = torch.float),requires_grad=False)
        self.var = torch.nn.Parameter(torch.zeros(size,dtype = torch.float),requires_grad=False)


    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var.copy_(torch.clamp(self.mean_diff/self.n, min=1e-2))

    def forward(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean)/obs_std

          
        
if __name__ == "__main__":
    size = 2
    step = 1000
    norm = Run_Normalizer(size = (size,))
    mean = torch.rand(size = (size,),dtype = torch.float)
    sigma = torch.rand(size = (size,),dtype = torch.float)+1
    data = torch.zeros(size = (size,step),dtype = torch.float)
    for i in range(step):
         sample = torch.normal(mean,sigma)
         data[:,i] = sample
         norm.observe(sample)
         assert norm.n[0].item() == i+1
    
    norm_mean = torch.reshape(norm.mean , (-1,))
    norm_std = torch.sqrt(torch.reshape(norm.var , (-1,)))
    data_mean = torch.mean(data,dim  = 1)
    data_std  = torch.std(data,dim  = 1 , unbiased = True)
    for i in range(size):
        assert -1e-6<norm_mean[i].item()-data_mean[i].item()<1e-6
        assert -1e-3<norm_std[i].item()-data_std[i].item()<1e-3
    print('pass test')
    