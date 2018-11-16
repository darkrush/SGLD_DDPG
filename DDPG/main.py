import numpy as np
import argparse

import os
import pickle
import torch
import gym

from logger import Singleton_logger
from evaluator import Singleton_evaluator
from model import Actor,Critic
from memory import Memory
from train import DDPG_trainer
from ddpg import DDPG
from obs_norm import Run_Normalizer
from noise import *



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='DDPG on pytorch')
    
    #Exp & Dir name 
    parser.add_argument('--output', default='results/', type=str, help='result output dir')
    parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--exp-name', default='0', type=str, help='exp dir name')
    parser.add_argument('--result-dir',default=None, type=str, help='whole result dir name')
    
    #Training args
    parser.add_argument('--nb-epoch', default=500, type=int, help='number of epochs')
    parser.add_argument('--nb-cycles-per-epoch', default=20, type=int, help='number of cycles per epoch')
    parser.add_argument('--nb-rollout-steps', default=100, type=int, help='number rollout steps')
    parser.add_argument('--nb-train-steps', default=50, type=int, help='number train steps')
    parser.add_argument('--max-episode-length', default=1000, type=int, help='max steps in one episode')
    parser.add_argument('--nb-warmup-steps', default=100, type=int, help='time without training but only filling the replay memory')
    
    #Model args
    parser.add_argument('--hidden1', default=64, type=int, help='number of hidden1')
    parser.add_argument('--hidden2', default=64, type=int, help='number of hidden2')
    parser.add_argument('--not-LN', dest='layer_norm', action='store_false',help='model without LayerNorm')
    parser.set_defaults(layer_norm=True)
    
    #DDPG args
    parser.add_argument('--actor-lr', default=0.0001, type=float, help='actor net learning rate')
    parser.add_argument('--critic-lr', default=0.001, type=float, help='critic net learning rate')
    parser.add_argument('--lr-decay', default=0, type=float, help='critic lr decay')
    parser.add_argument('--l2-critic', default=0.01, type=float, help='critic l2 regularization')
    parser.add_argument('--batch-size', default=128, type=int, help='minibatch size')
    parser.add_argument('--discount', default=0.99, type=float, help='reward discout')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--action-noise', dest='action_noise', action='store_true',help='enable action space noise')
    parser.set_defaults(action_noise=False)
    parser.add_argument('--parameter-noise', dest='parameter_noise', action='store_true',help='enable parameter space noise')
    parser.set_defaults(parameter_noise=False)
    parser.add_argument('--stddev', default=0.2, type=float, help='action noise stddev')
    parser.add_argument('--noise-decay', default=0, type=float, help='action noise decay')
    parser.add_argument('--SGLD-mode', default=0, type=int, help='SGLD mode, 0: no SGLD, 1: actor sgld only, 2: critic sgld only, 3: both actor & critic')
    parser.add_argument('--num-pseudo-batches', default=0, type=int, help='SGLD pseude batch number')
    parser.add_argument('--pool-mode', default=0, type=int, help='agent pool mode, 0: no pool, 1: actor pool only, 2: critic pool only, 3: both actor & critic')
    parser.add_argument('--pool-size', default=0, type=int, help='agent pool size, 0 means no agent pool')
    parser.add_argument('--obs-norm', dest='obs_norm', action='store_true',help='enable observation normalization')
    parser.set_defaults(obs_norm=False)
    parser.add_argument('--buffer-size', default=1e6, type=int, help='memory buffer size')
    
    #Other args
    parser.add_argument('--eval-visualize', dest='eval_visualize', action='store_true',help='enable render in evaluation progress')
    parser.set_defaults(eval_visualize=False)
    parser.add_argument('--rand_seed', default=314, type=int, help='random_seed')
    parser.add_argument('--nocuda', dest='with_cuda', action='store_false',help='disable cuda')
    parser.set_defaults(with_cuda=True)
    parser.add_argument('--mp', dest='multi_process', action='store_true',help='enable multi process')
    parser.set_defaults(multi_process=False)
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        args.with_cuda = False
        
    if args.rand_seed >= 0 :
        if args.with_cuda:
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(args.rand_seed)
        torch.manual_seed(args.rand_seed)
        np.random.seed(args.rand_seed)
        
    if args.result_dir is None:
        args.result_dir = os.path.join(args.output, args.env+args.exp_name)
    os.makedirs(args.result_dir, exist_ok=True)
        
    Singleton_logger.setup(args.result_dir,multi_process = args.multi_process)
    Singleton_evaluator.setup(args.env, logger = Singleton_logger, obs_norm = args.obs_norm, num_episodes = 10, max_episode_length=args.max_episode_length, model_dir = args.result_dir, multi_process = args.multi_process, visualize = args.eval_visualize, rand_seed = args.rand_seed)

    
    with open(os.path.join(args.result_dir,'args.pkl'),'wb') as f:
        pickle.dump(args, file = f)  
    with open(os.path.join(args.result_dir,'args.txt'),'w') as f:
        print(args,file = f)
        
    env = gym.make(args.env)
    if args.rand_seed >= 0 :
        env.seed(args.rand_seed)
        
    nb_actions = env.action_space.shape[0]
    nb_states = env.observation_space.shape[0]
    
    action_noise = None
    if args.action_noise:
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(args.stddev) * np.ones(nb_actions))
    parameter_noise = None
    if args.parameter_noise:
        parameter_noise = AdaptiveParamNoiseSpec( initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01)
    
    actor  = Actor (nb_states, nb_actions, hidden1 = args.hidden1, hidden2 = args.hidden2 , layer_norm = args.layer_norm)
    critic = Critic(nb_states, nb_actions, hidden1 = args.hidden1, hidden2 = args.hidden2 , layer_norm = args.layer_norm)
    memory = Memory(int(args.buffer_size), (nb_actions,), (nb_states,), args.with_cuda)
    obs_norm = None
    if args.obs_norm :
        obs_norm = Run_Normalizer(size = (nb_states,))
        
    agent = DDPG(actor_lr = args.actor_lr, critic_lr = args.critic_lr, lr_decay = args.lr_decay,
                 l2_critic = args.l2_critic, batch_size = args.batch_size, discount = args.discount, tau = args.tau,
                 action_noise = action_noise, noise_decay = args.noise_decay, 
                 parameter_noise = parameter_noise,
                 SGLD_mode = args.SGLD_mode, num_pseudo_batches = args.num_pseudo_batches, 
                 pool_mode = args.pool_mode, pool_size = args.pool_size, with_cuda = args.with_cuda)
    agent.setup(actor, critic, memory, obs_norm)
    
    trainer = DDPG_trainer(nb_epoch = args.nb_epoch, nb_cycles_per_epoch = args.nb_cycles_per_epoch,
                         nb_rollout_steps = args.nb_rollout_steps, nb_train_steps = args.nb_train_steps,
                         nb_warmup_steps = args.nb_warmup_steps)
    trainer.setup(env = env, agent = agent, evaluator = Singleton_evaluator, logger = Singleton_logger)
    
    trainer.warmup()
    trainer.train()
    
    Singleton_evaluator.trigger_close()
    Singleton_logger.trigger_close()
