# imports
import datetime
import os
from os import walk, listdir
import numpy as np
import gym
import time
from stable_baselines3 import PPO
import pandas as pd
import json
import argparse
import pprint as pp
import yaml
import pprint as pp

#local modules
from utils.utils import argparse_train_model2_mpi, plot_args, argparse_train_model, random_states, predict_actions, scatter_plot, CM, create_dir, argparse_plot_trajectories, argparse_train_model2
import torch as th

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def load_config(filename):
    """Load and return a config file."""

    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config
config_path = "config.yml"

if __name__ == '__main__':
    start_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    directory = "results/" + start_time + '/'
    Theta = {0: 113.92, 1: 87.15, 2: 107.97, 3: 113.67, 4: 113.58, 5: 95.68}
    parser = argparse.ArgumentParser(description='provide arguments finding optimal policy of the SEIR model')

    # general parameters
    parser.add_argument('--summary_dir', help='directory for saving and loading model and other data', default=directory)
    parser.add_argument('--Senario', help='Scenarios that needs to run', choices=[0,1,2,3,4,5], type=int, default=0)
    parser.add_argument('--seed', help='seed for random number generator', type=int, default=2222)
    parser.add_argument('--num_cpu', help='seed for random number generator', type=int, default=30)

    # RL agent hyperparameters
    parser.add_argument('--n_timesteps', help='Total number of training steps for training', type = int, default=int(1e1))
    parser.add_argument('--check_freq', help='frequency of upating the model ', type = int, default=int(1000/30))
    parser.add_argument('--policy_kwargs', help='policy kwargs for the agent NN model', type=json.loads, default=dict(activation_fn=th.nn.ReLU, net_arch=[128, dict(pi=[512, 512], vf=[512, 512])]))
    parser.add_argument('--rl_algo', help='0:PPO, 1:A2C, 2:DQN', choices=[0,1,2], type=int, default=0)
    parser.add_argument('--learning_rate', help='Control the Learning rate', type=float, default=0.0003)
    parser.add_argument('--clip_range', help='Controls the clip parameter of PPO algorithm', type=float, default=0.1)


    # SEIR model hyperparameters
    parser.add_argument('--env_id', help='gym environment id ', default='gym_seir:seir-b-v0')
    parser.add_argument('--theta', help=' theta of the selected scenarios', type=float, default=None)
    parser.add_argument('--weight', help='weight factor in reward', type = float, default=0.5)
    parser.add_argument('--health_cost_scale', help='weight factor in reward', type = float, default=581.)
    parser.add_argument('--rho_per_week', help='weight factor in reward', type = float, default=0.02)
    parser.add_argument('--hospital_beds_ratio', help='weight factor in reward', type = float, default=0.005)
    parser.add_argument('--max_hospital_cost', help='weight factor in reward', type = float, default=10.)


    # plotting hyperparameters
    parser.add_argument('--N', help='Number of samples used to plot', type = int, default=5000)
    parser.add_argument('--plot_inital_states', help='Initial states for plotting', type = json.loads, default=[[99666., 81., 138., 115.]])

    args = vars(parser.parse_args())
    args['theta'] = Theta[args["Senario"]] if args['theta'] == None else args['theta']
    # if args['theta']==0 or args['theta']==None:
    #     args['theta'] = Theta[args["Senario"]]
    pp.pprint(args)

    np.random.seed(args['seed'])
    args['summary_dir'] = args['summary_dir'] + '/'
    try:
        create_dir(args['summary_dir'])
    except:
        pass

    states = random_states(args['N'])
    # model = argparse_train_model2(args)
    model = argparse_train_model2_mpi(args)
    print("plotting")
    # df, actions = predict_actions(states, model, df=True)
    # scatter_plot(df=df, save_fig=True, fig_name=args['summary_dir']+"scatter.jpg")
    # scatter_plot(df=df, save_fig=True, fig_name=args['summary_dir']+"scatter.pdf")

    # for plotting we always plot learned model on baseline scenario,
    # so setting theta to the baseline scenario - value
    if len(args['plot_inital_states'])==0:
        argparse_plot_trajectories(model, args, inital_state=None, eval=False)
    else:
        for init_state in args['plot_inital_states']:
            argparse_plot_trajectories(model,args, inital_state=init_state, eval=False)

    # for plotting we always plot learned model on baseline scenario,
    # so setting theta to the baseline scenario - value
    args['theta'] = Theta[0]
    if len(args['plot_inital_states'])==0:
        argparse_plot_trajectories(model, args, inital_state=None)
    else:
        for init_state in args['plot_inital_states']:
            argparse_plot_trajectories(model,args, inital_state=init_state)

    args_path = args['summary_dir'] + 'args.csv'
    # with open(args_path, 'w') as file:
    #     file.write(json.dumps(args)) # use `json.loads` to do the reverse

    cost_train, cost_eval = plot_args(args, eval=None), plot_args(args, eval=True)

    end_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    args['start_time'] = start_time
    args['end_time'] = end_time
    args['cost_train'] = cost_train
    args['cost_eval'] = cost_eval
    HP = pd.DataFrame(list(args.items()),columns = ['hyperparameter','value'])
    HP.to_csv(args_path)

    model.save(args['summary_dir'] + "model")
