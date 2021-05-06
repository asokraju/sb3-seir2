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
#local modules
from utils.utils import train_model, random_states, predict_actions, scatter_plot, CM, create_dir
import torch as th

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == '__main__':

    start_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    directory = "results/" + start_time + '/'
    create_dir("results/")
    create_dir(directory)
    # try:
    #     os.mkdir("results/")
    # except:
    #     pass
    # try:
    #     os.mkdir(directory)
    # except:
    #     pass
    args = {
        'n_timesteps' : int(0.5e5), # No of RL training steps
        'check_freq' : 1000, # frequency of upating the model
        'env_id' : 'gym_seir:seir-v0', # gym environment id
        'N' : 5000, # number of samples to plot
        'theta':{0: 113.92, 1: 87.15, 2: 107.97},
        'w_all' : [0.0 , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ],
        'sel_w' : [0.1],  # the weight values to be used in the code
        'Senarios' : [ 'BaseLine', 'Senario_1', 'Senario_2'],
        'a_map' : {0:'LockDown', 1:'Social Distancing', 2:'Open'},
        'initial_state':{
            0:[99666., 81., 138., 115.], 
            1:[99962.0, 7.0, 14.0, 17.0],
            2:[99905.0, 22.0, 39.0, 34.0]
            },
        'seed': 2424, #random number generator seed
        'policy_kwargs': dict(activation_fn=th.nn.ReLU,
                     net_arch=[128, dict(pi=[512, 512], vf=[512, 512])])  #NN parameters
    }
    np.random.seed(args['seed'])
    states = random_states(args['N'])
    for w in args['sel_w']:
        dir_w = directory + str(w) +"/"
        create_dir(dir_w)
        # try:
        #     os.mkdir(dir_w)
        # except:
        #     pass
        Scenario_actions = []
        for i, senario in enumerate(args['Senarios']):
            dir_sen = dir_w + senario + "/"
            create_dir(dir_sen)
            # try:
            #     os.mkdir(dir_sen)
            # except:
            #     pass
            print(dir_sen)
            start_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
            
            # training the model
            model = train_model(w, i, args, log_dir=dir_sen, seed=args['seed'])
            print("plotting")
            df, actions = predict_actions(states, model, df=True)
            Scenario_actions.append(actions)
            print(len(actions), len(Scenario_actions))
            scatter_plot(df=df, save_fig=True, fig_name=dir_sen+"scatter.pdf") 

        baseline_model_dir = dir_w + args['Senarios'][0] + "/"

        Scenario_1_model_dir = dir_w + args['Senarios'][1] + "/"
        C, DF1 = CM(states, Scenario_actions[0], Scenario_actions[1], save_fig=True, fig_name=Scenario_1_model_dir+'confusion.pdf')
        C.to_csv(Scenario_1_model_dir+'C.csv')
        Scenario_2_model_dir = dir_w + args['Senarios'][2] + "/"
        C, DF2 = CM(states, Scenario_actions[0], Scenario_actions[2], save_fig=True, fig_name=Scenario_2_model_dir+'confusion.pdf')
        C.to_csv(Scenario_2_model_dir+'C.csv')

        df = pd.DataFrame(states, columns=['Susceptible', 'Exposed', 'Infected', 'Recovered'])
        df['Baseline'] = Scenario_actions[0]
        df['Scenario-1'] = Scenario_actions[1]
        df['Scenario-2'] = Scenario_actions[2]
        df.to_csv(dir_w+'data.csv')
    try:
        args_path = dir_w+'/args.txt'
        with open(args_path, 'w') as file:
            file.write(json.dumps(args)) # use `json.loads` to do the reverse
    except:
        pass 
        



