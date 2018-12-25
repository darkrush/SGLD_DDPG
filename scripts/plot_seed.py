import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='DDPG on pytorch')
parser.add_argument('--logdir', default = None, type=str, help='result dir')
parser.add_argument('--log-file',default='log_data_dict.pkl', type=str, help='log pkl file')
parser.add_argument('--arg-file',default='args.pkl', type=str, help='args txt file')
args = parser.parse_args()



item_list = ['eval_reward_mean','actor_loss_mean','critic_loss_mean','train_episode_reward','train_episode_length']

exp_list = os.listdir(os.path.join(args.logdir))
dataframe = {}
for exp_name in exp_list:
    dataframe[exp_name] = {}

for exp_name in exp_list:
    seed_list = os.listdir(os.path.join(args.logdir,exp_name))
    value_dict = {}
    for item_name in item_list:
        value_dict[item_name] = {}

    least_length = {}
    for item_name in item_list:
        least_length[item_name] = None
    for seed_name in seed_list:
        with open(os.path.join(args.logdir,exp_name,seed_name, args.log_file),'rb') as f:
            log_dict=pickle.load(f)
        for item_name in item_list:
            value,step = zip(*log_dict[item_name])
            value_dict[item_name][seed_name] = value
            least_length[item_name] = len(value) if (least_length[item_name] is None) or (least_length[item_name] > len(value)) else least_length[item_name]
            value_dict[item_name]['step'] = step
            print(exp_name,seed_name,item_name, len(step))
    for seed_name in seed_list:
        for item_name in item_list:
            value_dict[item_name][seed_name] =value_dict[item_name][seed_name][:least_length[item_name]]
            value_dict[item_name]['step'] = value_dict[item_name]['step'][:least_length[item_name]]
    for item_name in item_list:
        dataframe[exp_name][item_name]= pd.DataFrame(value_dict[item_name]).melt(id_vars = ['step'])
    #for item_name in item_list:
    #    print(item_name , exp_name, least_length[item_name])
plt.ion()
for item_name in item_list:
    plt.figure(item_name)
    plt.title(args.logdir +'\n'+ item_name)
    for exp_name in exp_list:
        sns.lineplot(x='step',y = 'value' , data = dataframe[exp_name][item_name], ci = 'sd', label = exp_name)
    plt.legend()
plt.ioff()     
plt.show()