import json
import datetime
import os 
args = {
        'n_timesteps' : int(1e5), # No of RL training steps
        'check_freq' : 1000, # frequency of upating the model
        'env_id' : 'gym_seir:seir-b-v0', # gym environment id
        'N' : 5000, # number of samples to plot
        'theta':{0: 113.92, 1: 87.15, 2: 107.97},
        'w_all' : [0.0 , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ],
        'sel_w' : [0.5],
        'Senarios' : [ 'BaseLine', 'Senario_1', 'Senario_2'],
        'Selected_Senarios': ['BaseLine'],
        'a_map' : {0:'LockDown', 1:'Social Distancing', 2:'Open'},
        'initial_state':{
            0:[99666., 81., 138., 115.], 
            1:[99962.0, 7.0, 14.0, 17.0],
            2:[99905.0, 22.0, 39.0, 34.0]
            },
        'seed': 2424,
        # plots the trajectories for specified scenario and initial states
        'plot_inital_states' : {
            0 : [[99666., 81., 138., 115.]],
            1 : [[99666., 81., 138., 115.], [99962.0, 7.0, 14.0, 17.0]],
            2 : [[99666., 81., 138., 115.], [99905.0, 22.0, 39.0, 34.0]],
        },
        'seed': 2424, # random number generator seed
        # Total COst = w * Economic cost + (1-w)* Public Health  Cost/ health_cost_scale
        # So the cost is less sensitive to public health and may favour economic cost
        'health_cost_scale': 650.,
        # 'health_cost_scale': 1. ==> 0 po licy
        # 'health_cost_scale': 1000. ==> 1,
        'rho_per_week': 0.02,
        'hospital_beds_ratio': 0.01, # set this to 1.1 if we dont want to implement the hospital beds case.
        'max_hospital_cost':100.,
    }
def create_dir(dir_name:str):
    try:
        os.mkdir(dir_name)
    except:
        print("Sub directory {} is already available".format(dir_name))
        pass
start_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
directory = "results/" + start_time + '/'
print(directory)
create_dir(directory)
with open(directory+'args.txt', 'w') as file:
    file.write(json.dumps(args))