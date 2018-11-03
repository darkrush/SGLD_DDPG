import torch

class Run_Normalizer():
    def __init__(self, size, if_cuda):
        self.n = torch.zeros(size,dtype = torch.float)
        self.mean = torch.zeros(size,dtype = torch.float)
        self.mean_diff = torch.zeros(size,dtype = torch.float)
        self.var = torch.zeros(size,dtype = torch.float)
        self.if_cuda = if_cuda
        if if_cuda:
            self.cuda()

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean)/obs_std
        
        
    def cuda(self):
        self.n=self.n.cuda()
        self.mean=self.mean.cuda()
        self.mean_diff=self.mean_diff.cuda()
        self.var=self.var.cuda()
          
        
if __name__ == "__main__":
    size = 2
    step = 1000
    norm = Run_Normalizer(size = (size,),if_cuda = False)
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
    