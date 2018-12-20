import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='DDPG on pytorch')
parser.add_argument('--logdir', type=str, help='result dir')
parser.add_argument('--low',default = 1,type=int, help='first run number')
parser.add_argument('--high',default = 4,type=int, help='last run number')
parser.add_argument('--log-file',default='log_data_dict.pkl', type=str, help='log pkl file')
parser.add_argument('--arg-file',default='args.pkl', type=str, help='args txt file')
parser.add_argument('--item',default='', type=str, help='args txt file')
parser.add_argument('--smooth',default=0.0, type=float, help='curve smooth coef')

args = parser.parse_args()
plt.ion()

itemlist = ['eval_reward_mean','actor_loss_mean','critic_loss_mean','train_episode_reward','train_episode_length','last_error']
for i in range(len(itemlist)):
    for run_num in range(args.low,args.high+1):
        with open(os.path.join(args.logdir+'{}'.format(run_num), args.arg_file),'rb') as f:
            exp_args = pickle.load(f)
        with open(os.path.join(args.logdir+'{}'.format(run_num), args.log_file),'rb') as f:
            m=pickle.load(f)
        if itemlist[i] not in m:
            continue
        reward_mean,step = zip(*m[itemlist[i]])
        smooth_reward = [reward_mean[0],]
        for r in reward_mean:
            smooth_reward.append(smooth_reward[-1]*args.smooth + r*(1-args.smooth))
        smooth_reward=smooth_reward[1:]

        plt.figure(i)
        plt.title(exp_args.env +'\n'+ itemlist[i])
        
        label = None
        if exp_args.SGLD_mode is not 0:
            label = 'SGLD'
        if exp_args.action_noise:
            label = 'action-noise'
        if exp_args.parameter_noise:
            label = 'parameter-noise'
        if label is None:
            label = 'No exploration'
        label = exp_args.exp_name + ' ' + label
        plt.plot(step,smooth_reward,label= label)
        plt.legend()

plt.ioff()     
plt.show()
