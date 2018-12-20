import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import argparse
import os

def plot_item(figure_id,item_name):
    



parser = argparse.ArgumentParser(description='DDPG on pytorch')
parser.add_argument('--logdir', type=str, help='result dir')
parser.add_argument('--low',default = 1,type=int, help='first run number')
parser.add_argument('--high',default = 4,type=int, help='last run number')
parser.add_argument('--nb-seed',default = 4,type=int, help='number of seed')
parser.add_argument('--log-file',default='log_data_dict.pkl', type=str, help='log pkl file')
parser.add_argument('--arg-file',default='args.pkl', type=str, help='args txt file')
parser.add_argument('--item',default='', type=str, help='args txt file')
parser.add_argument('--smooth',default=0.0, type=float, help='curve smooth coef')

args = parser.parse_args()
plt.ion()

itemlist = ['eval_reward_mean','actor_loss_mean','critic_loss_mean','train_episode_reward','train_episode_length']
for item_id in range(len(itemlist)):
    for run_num in range(args.low,args.high+1):
        with open(os.path.join(args.logdir+'{}_1'.format(run_num), args.arg_file),'rb') as f:
            exp_args = pickle.load(f)
        value_dict = {}
        least_length = None
        for seed_id in range(1,args.nb_seed+1):
            with open(os.path.join(args.logdir+'{}_{}'.format(run_num,seed_id), args.log_file),'rb') as f:
                m=pickle.load(f)
            if itemlist[item_id] not in m:
                continue
            value,step = zip(*m[itemlist[item_id]])
            value_dict[seed_id] = value
            least_length = len(value) if (least_length is None) or (least_length > len(value)) else least_length
        for seed_id in range(1,args.nb_seed+1):
            value_dict[seed_id] =value_dict[seed_id][:least_length]
        value_dict['step'] = step[:least_length]
        df = pd.DataFrame(value_dict).melt(id_vars = ['step'])
        plt.figure(item_id)
        plt.title(exp_args.env +'\n'+ itemlist[item_id])
        
        label = None
        if exp_args.SGLD_mode is not 0:
            label = 'SGLD'
        if exp_args.action_noise:
            label = 'action-noise {}'.format(exp_args.stddev)
        if exp_args.parameter_noise:
            label = 'parameter-noise {}'.format(exp_args.stddev)
        if label is None:
            label = 'No exploration'
#        label = exp_args.exp_name + ' ' + label
        sns.lineplot(x='step',y = 'value' , data = df, ci = 'sd', label = label)
        #plt.plot(step,mean_array,label= label)
        plt.legend()

plt.ioff()     
plt.show()
