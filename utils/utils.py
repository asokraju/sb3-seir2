import numpy as np
from itertools import permutations
import pandas as pd
import os
import gym
from pandas.core.frame import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, A2C, DQN
from sklearn.metrics import confusion_matrix

from typing import Callable
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


def train_model(w:float, Senario:int, args:dict, log_dir:str, seed:int=None):
    env_id = args['env_id']
    n_timesteps = args['n_timesteps']
    check_freq = args['check_freq']
    tensorboard_log = log_dir + "board/"
    env_kwargs = {
        'validation':False,
        'theta':args['theta'][Senario],
        'weight' : w,
        'health_cost_scale' : args['health_cost_scale'],
        'rho_per_week': args['rho_per_week'],
        'hospital_beds_ratio': args['hospital_beds_ratio'],
        'max_hospital_cost':args['max_hospital_cost'],
        }
    env = gym.make(env_id,**env_kwargs)
    env = Monitor(env, log_dir)
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=0, 
        tensorboard_log=tensorboard_log, 
        seed = seed,
        policy_kwargs = args["policy_kwargs"])
    callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir)
    model.learn(n_timesteps, tb_log_name="test_1", callback=callback)
    print("Finished training")
    return model

def random_uniform_state(all:bool=False):
    popu=1e5
    X = np.random.uniform(low=0.0, high=popu)
    Y = np.random.uniform(low=0.0, high=popu-X)
    Z = np.random.uniform(low=0.0, high=popu-(X + Y))
    W = popu-(X+Y+Z)
    perms = permutations([X, Y, Z, W])
    if all:
        States = []
        for p in perms:
            States.append(list(p))
        return States
    else:
        return list(perms[np.random.choice(np.arange(len(perms)))])

def random_states(N:int=10000, df:bool=False):
    states = []
    for _ in range(N):
        s = random_uniform_state(all=True)
        states += s
    if df:
        return pd.DataFrame(states, columns=['Susceptible', 'Exposed', 'Infected', 'Recovered'])
    else:
        return states

def normalize_state(state):
    popu = 1e5
    S, E, I, R = state[0], state[1], state[2], state[3]
    S, E, I, R = S/popu, E/popu, I/popu, R/popu
    return np.array([S, E, I, R], dtype=float)

def predict_actions(states, model, df:bool=False):
    actions = []
    a_map = ['0:LockDown','1:Social Distancing', '2:Open']
    for s in states:
        a = model.predict(normalize_state(s), deterministic=True)[0]
        actions.append(a)
    POL = [a_map[a] for a in actions]
    if df:
        DF = pd.DataFrame(states, columns=['Susceptible', 'Exposed', 'Infected', 'Recovered'])
        DF['policy'] =  POL
        return DF, actions
    else:
        actions
    
def scatter_plot(df:DataFrame, save_fig:bool=False, fig_name:str=None):
    plot_kws={'alpha': 0.6}
    hue_order = ['0:LockDown','1:Social Distancing', '2:Open']
    pal = {'0:LockDown':"Red", '1:Social Distancing':"Green",'2:Open':'Blue'}
    # g = sns.pairplot(df, kind='scatter', alpha=0.1})
    g = sns.pairplot(df, hue="policy",  palette=pal, plot_kws = plot_kws, hue_order = hue_order) #,height=10, aspect=1.
    # g.set_title(title)
    # g.title(title)
    if save_fig:
        g.savefig(fig_name, bbox_inches='tight')
        plt.close()
    else:
        return g

def CM(states, baseline_actions, scenario_actions, save_fig:bool=False, fig_name:str=None):

    a_map = ['0:LockDown','1:Social Distancing', '2:Open']

    POL_B = [a_map[a] for a in baseline_actions]
    POL_S = [a_map[a] for a in scenario_actions]

    DF = pd.DataFrame(states, columns=['Susceptible', 'Exposed', 'Infected', 'Recovered'])
    DF['baseline_policy'] =  POL_B
    DF['scenario_policy'] =  POL_S
    C = confusion_matrix(y_true = DF['baseline_policy'], y_pred = DF['scenario_policy'], labels = a_map)
    if save_fig:
        sns.heatmap(C*(100/np.shape(states)[0]), annot=True)
        plt.savefig(fig_name, bbox_inches='tight')
    return pd.DataFrame(C), DF


def create_dir(dir_name:str):
    try:
        os.mkdir(dir_name)
    except:
        print("Sub directory {} is already available".format(dir_name))
        pass

def plot_trajectories(model, w:float, Senario:int, args:dict, log_dir:str, inital_state):
    env_id = args['env_id']
    env_kwargs = {
        'validation':True,
        'theta':args['theta'][Senario],
        'weight' : w,
        'inital_state':inital_state,
        'health_cost_scale': args['health_cost_scale'],
        'rho_per_week': args['rho_per_week'],
        'hospital_beds_ratio': args['hospital_beds_ratio'],
        'max_hospital_cost':args['max_hospital_cost'],
        }
    env = gym.make(env_id,**env_kwargs)
    env = Monitor(env, log_dir)
    actions, rewards = [], []
    done = False
    s = env.reset()
    while not done:
        a = model.predict(s, deterministic=True)[0] 
        s, r, done, _ = env.step(a)
        # states.append(env.state)
        rewards.append(r)
        actions.append(a)

    Rewards = [i for i in rewards for _ in range(env.time_steps)]
    Actions = [i for i in actions for _ in range(env.time_steps)]
    index = pd.date_range("2020-05-15 00:00:00", "2020-11-05 23:55:00", freq="5min")
    df = pd.DataFrame(np.array(env.state_trajectory)[:-1], columns=['Susceptible', 'Exposed', 'Infected', 'Recovered'], index=index)
    df['actions'] = Actions
    df['rewards'] = Rewards

    main_title = "weight = " + str(w) + ", " + "Scenario - " + str(Senario) + " - " + str(inital_state)
    ax = df[['Susceptible', 'Exposed', 'Infected', 'Recovered']].plot.line(subplots=True, figsize = (10,10), title = main_title + 'state')
    for axes in ax:
        axes.set_ylim([0, 1e5])
    plt.savefig(log_dir+main_title+"states.png")
    plt.close()
    
    ax = df['actions'].plot.line( figsize = (10,2.5), title = main_title + 'actions')
    ax.set_ylim([-0.1,2.2])
    plt.savefig(log_dir+main_title+"actions.png")
    plt.close()

    ax = df['rewards'].plot.line( figsize = (10,2.5), title = main_title + 'rewards')
    plt.savefig(log_dir+main_title+"rewards.png")
    plt.close()
    df.to_csv(log_dir+'sar.csv')
    return df



# C:\Users\kkris\Documents\GitHub\sb3-seir2\results\21-08-24-11-34\board\test_1_1
def argparse_train_model(args:dict):
    env_id = args['env_id']
    n_timesteps = args['n_timesteps']
    check_freq = args['check_freq']
    tensorboard_log = args['summary_dir'] + "board/"
    log_dir = args['summary_dir']
    env_kwargs = {
        'validation':False,
        'theta':args['theta'],
        'weight' : args['weight'],
        'health_cost_scale' : args['health_cost_scale'],
        'rho_per_week': args['rho_per_week'],
        'hospital_beds_ratio': args['hospital_beds_ratio'],
        'max_hospital_cost':args['max_hospital_cost']
        }
    env = gym.make(env_id,**env_kwargs)
    env = Monitor(env, log_dir)
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=0, 
        tensorboard_log=tensorboard_log, 
        seed = args['seed'],
        policy_kwargs = args["policy_kwargs"],
        learning_rate=args['learning_rate'],
        clip_range=args['clip_range']
        )
    callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir)
    model.learn(n_timesteps, tb_log_name="test_1", callback=callback)
    print("Finished training")
    return model

def make_env(args: dict, rank: int) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    env_id = args['env_id']
    seed = args['seed']
    env_kwargs = {
        'validation':False,
        'theta':args['theta'],
        'weight' : args['weight'],
        'health_cost_scale' : args['health_cost_scale'],
        'rho_per_week': args['rho_per_week'],
        'hospital_beds_ratio': args['hospital_beds_ratio'],
        'max_hospital_cost':args['max_hospital_cost']
        }
    def _init() -> gym.Env:
        env = gym.make(env_id,**env_kwargs)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def argparse_train_model_mpi(args:dict):
    env_id = args['env_id']
    n_timesteps = args['n_timesteps']
    check_freq = args['check_freq']
    tensorboard_log = args['summary_dir'] + "board/"
    log_dir = args['summary_dir']
    num_cpu = args['num_cpu']
    env = SubprocVecEnv([(make_env(args, i),  log_dir)for i in range(num_cpu)])
    # env = Monitor(env, log_dir)
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=0, 
        tensorboard_log=tensorboard_log, 
        seed = args['seed'],
        policy_kwargs = args["policy_kwargs"],
        learning_rate=args['learning_rate'],
        clip_range=args['clip_range']
        )
    callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir)
    model.learn(n_timesteps, tb_log_name="test_1", callback=callback)
    print("Finished training")
    return model

def argparse_plot_trajectories(model, args:dict, inital_state, eval=True):
    env_id = args['env_id']
    env_kwargs = {
        'validation':True,
        'theta':args['theta'],
        'weight' : args['weight'],
        'inital_state':inital_state,
        'health_cost_scale': args['health_cost_scale'],
        'rho_per_week': args['rho_per_week'],
        'hospital_beds_ratio': args['hospital_beds_ratio'],
        'max_hospital_cost':args['max_hospital_cost'],
        }
    log_dir = args['summary_dir']
    env = gym.make(env_id,**env_kwargs)
    env = Monitor(env, log_dir)
    actions, rewards = [], []
    done = False
    s = env.reset()
    while not done:
        a = model.predict(s, deterministic=True)[0] 
        s, r, done, _ = env.step(a)
        # states.append(env.state)
        rewards.append(r)
        actions.append(a)

    Rewards = [i for i in rewards for _ in range(env.time_steps)]
    Actions = [i for i in actions for _ in range(env.time_steps)]
    index = pd.date_range("2020-05-15 00:00:00", "2020-11-05 23:55:00", freq="5min")
    df = pd.DataFrame(np.array(env.state_trajectory)[:-1], columns=['Susceptible', 'Exposed', 'Infected', 'Recovered'], index=index)
    df['actions'] = Actions
    df['rewards'] = Rewards

    main_title = "weight = " + str(args['weight']) + ", " + "Scenario - " + str(args['Senario']) + " - " + str(inital_state)
    ax = df[['Susceptible', 'Exposed', 'Infected', 'Recovered']].plot.line(subplots=True, figsize = (10,10), title = main_title + 'state')
    for axes in ax:
        axes.set_ylim([0, 1e5])
    if eval == True:
        EVAL = "EVAL-"
    else:
        EVAL = ""

    plt.savefig(log_dir+main_title+EVAL+"states.png")
    plt.close()
    
    ax = df['actions'].plot.line( figsize = (10,2.5), title = main_title + 'actions')
    ax.set_ylim([-0.1,2.2])
    plt.savefig(log_dir+main_title+EVAL+"actions.png")
    plt.close()

    ax = df['rewards'].plot.line( figsize = (10,2.5), title = main_title + 'rewards')
    plt.savefig(log_dir+main_title+EVAL+"rewards.png")
    plt.close()
    df.to_csv(log_dir+EVAL+'sar.csv')
    return df



def argparse_train_model2(args:dict):
    env_id = args['env_id']
    n_timesteps = args['n_timesteps']
    check_freq = args['check_freq']
    tensorboard_log = args['summary_dir'] + "board/"
    log_dir = args['summary_dir']
    env_kwargs = {
        'validation':False,
        'theta':args['theta'],
        'weight' : args['weight'],
        'health_cost_scale' : args['health_cost_scale'],
        'rho_per_week': args['rho_per_week'],
        'hospital_beds_ratio': args['hospital_beds_ratio'],
        'max_hospital_cost':args['max_hospital_cost']
        }
    env = gym.make(env_id,**env_kwargs)
    env = Monitor(env, log_dir)
    if args['rl_algo'] == 0:
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=0, 
            tensorboard_log=tensorboard_log, 
            seed = args['seed'],
            policy_kwargs = args["policy_kwargs"],
            learning_rate=args['learning_rate'],
            clip_range=args['clip_range']
            )
    elif args['rl_algo'] == 1:
        model = A2C(
            'MlpPolicy', 
            env, 
            verbose=0, 
            tensorboard_log=tensorboard_log, 
            seed = args['seed'],
            policy_kwargs = args["policy_kwargs"],
            # learning_rate=args['learning_rate'],
            )
    elif args['rl_algo'] == 2:
        model = DQN(
            'MlpPolicy', 
            env, 
            verbose=0, 
            tensorboard_log=tensorboard_log, 
            seed = args['seed'],
            policy_kwargs = args["policy_kwargs"],
            # learning_rate=args['learning_rate'],
            )
 
    callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir)
    model.learn(n_timesteps, tb_log_name="test_1", callback=callback)
    print("Finished training")
    return model

def argparse_train_model2_mpi(args:dict):
    env_id = args['env_id']
    n_timesteps = args['n_timesteps']
    check_freq = args['check_freq']
    tensorboard_log = args['summary_dir'] + "board/"
    log_dir = args['summary_dir']
    num_cpu = args['num_cpu']
    env = SubprocVecEnv([make_env(args, i) for i in range(num_cpu)])
    # env = Monitor(env, log_dir)
    # model = PPO(
    #     'MlpPolicy', 
    #     env, 
    #     verbose=0, 
    #     tensorboard_log=tensorboard_log, 
    #     seed = args['seed'],
    #     policy_kwargs = args["policy_kwargs"],
    #     learning_rate=args['learning_rate'],
    #     clip_range=args['clip_range']
    #     )
    # env = Monitor(env, log_dir)
    if args['rl_algo'] == 0:
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=0, 
            tensorboard_log=tensorboard_log, 
            seed = args['seed'],
            policy_kwargs = args["policy_kwargs"],
            learning_rate=args['learning_rate'],
            clip_range=args['clip_range']
            )
    elif args['rl_algo'] == 1:
        model = A2C(
            'MlpPolicy', 
            env, 
            verbose=0, 
            tensorboard_log=tensorboard_log, 
            seed = args['seed'],
            policy_kwargs = args["policy_kwargs"],
            # learning_rate=args['learning_rate'],
            )
    elif args['rl_algo'] == 2:
        model = DQN(
            'MlpPolicy', 
            env, 
            verbose=0, 
            tensorboard_log=tensorboard_log, 
            seed = args['seed'],
            policy_kwargs = args["policy_kwargs"],
            # learning_rate=args['learning_rate'],
            )
 
    callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir)
    model.learn(n_timesteps, tb_log_name="test_1")
    print("Finished training")
    return model



def plot_path(path, eval=None):
    if eval == True:
        EVAL = "EVAL-"
    else:
        EVAL = ""
    idx = path.find("mhc=")
    mhc = float(path[idx+4:idx+7])
    idx = path.find("hbr=")
    hbr = float(path[idx+4:idx+9])
    idx = path.find("hcs=")
    hcs = float(path[idx+4:idx+9])
    idx = path.find("hcs=")
    scenario = path[idx-2:idx-1]
    # print(mhc, hbr, hcs, scenario)
    Scenarios = [ 'BaseLine', 'Senario_1', 'Senario_2']
    states = ['Susceptible', 'Exposed', 'Infected', 'Recovered']
    a_map = {0:'LockDown', 1:'Social Distancing', 2:'Open'}
    cols = []
    df_0 = pd.read_csv(path + "\\"+EVAL+"sar.csv",  index_col=0, infer_datetime_format=True)
    # print(df_0.iloc[0])
    df_0.index = pd.to_datetime(df_0.index)
    eco_costs    = np.array([1., 0.2, 0.0], dtype=float)
    df_0['EconomicCost'] = df_0['actions'].map(lambda x: eco_costs[x])*7.
    def Bed_cost(x):
        avail_hospital_beds = 0.005*1e5
        return 0. if x < avail_hospital_beds else mhc
    df_0['bedcost'] = df_0['Infected'].map(lambda x: Bed_cost(x))
    df_0.head()
    Susceptible = df_0['Susceptible'].values
    dt           = 5/(24*60)
    Ts           = 7
    time_steps   = int((Ts) / dt)
    def ds_cost(Susceptible):
        ds = []
        l            = np.shape(Susceptible)[0]
        weakly_sus = Susceptible[::time_steps]
        # print(len(weakly_sus))
        for i in range(24):
            for _ in range(time_steps):
                ds.append(weakly_sus[i]-weakly_sus[i+1])
        for _ in range(time_steps):
            ds.append(weakly_sus[i]-weakly_sus[i+1])
        return ds
    df_0['Ds_cost'] = np.array(ds_cost(Susceptible))/hcs
    df_0['PublicHealthCost'] = df_0['bedcost'] + df_0['Ds_cost']
    df_0['TotalCost'] = (1-0.5)*df_0['PublicHealthCost'] + (0.5)*df_0['EconomicCost']
    TotalCost = df_0['TotalCost'].values
    TotalWeeklyCosts = TotalCost[::time_steps]
    TotalWeeklyCumulativeCosts = discount_reward(TotalWeeklyCosts, GAMMA=0.99)
    TotalDialyCumulativeCosts = []
    for _ in TotalWeeklyCumulativeCosts:
        for t in range(time_steps):
            TotalDialyCumulativeCosts.append(_)
    df_0['CumulativeCost'] = TotalDialyCumulativeCosts
    index = pd.date_range("2020-05-15 00:00:00", "2020-11-05 23:55:00", freq="5min")
    main_title = "weight = " + str(0.5) + ", "
    ax = df_0[['Susceptible', 'Exposed', 'Infected', 'Recovered', 'actions']].plot.line(subplots=True, figsize = (5,5))
    ax[0].set_ylim([94000, 100000])
    ax[1].set_ylim([0, 500])
    ax[2].set_ylim([0, 1500])
    ax[2].axhline(y=hbr*1e5)
    ax[3].set_ylim([0, 5000])
    ax[4].set_ylim([-0.1, 2.1])
    plt.savefig(path + main_title+"Scenario - " + scenario + " - states.png")
    # plt.show()
    plt.close()

    index = pd.date_range("2020-05-15 00:00:00", "2020-11-05 23:55:00", freq="5min")
    main_title = "weight = " + str(0.5) + ", "
    ax = df_0[['PublicHealthCost', 'EconomicCost', 'TotalCost', 'CumulativeCost']].plot.line(subplots=True, figsize = (5,5))
    # ax[0].set_ylim([94000, 100000])
    # ax[1].set_ylim([0, 500])
    ax[2].set_ylim([-0.1, 7.5])
    # ax[3].set_ylim([-0.1, 2.1])
    plt.savefig(path + main_title+"Scenario - " + scenario + " - costs.png")
    # plt.show()
    plt.close()
    return TotalWeeklyCumulativeCosts[0]


def discount_reward(rewards, GAMMA=0.99):
    reward_sum = 0
    discounted_rewards = []
    for reward in rewards[::-1]:  # reverse buffer r
        reward_sum = reward + GAMMA * reward_sum
        discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()
    # discounted_rewards = np.array(discounted_rewards)
    # len = discounted_rewards.shape[0]
    return discounted_rewards

def plot_args(args, eval=None):
    if eval == True:
        EVAL = "EVAL-"
    else:
        EVAL = ""
    print(args['max_hospital_cost'])
    mhc = float(args['max_hospital_cost'])
    hbr = float(args['hospital_beds_ratio'])
    hcs = float(args['health_cost_scale'])
    scenario = int(args['Senario'])
    path =args['summary_dir']
    # print(mhc, hbr, hcs, scenario)
    Scenarios = [ 'BaseLine', 'Senario_1', 'Senario_2']
    states = ['Susceptible', 'Exposed', 'Infected', 'Recovered']
    a_map = {0:'LockDown', 1:'Social Distancing', 2:'Open'}
    cols = []
    df_0 = pd.read_csv(path +EVAL+"sar.csv",  index_col=0, infer_datetime_format=True)
    # print(df_0.iloc[0])
    df_0.index = pd.to_datetime(df_0.index)
    eco_costs    = np.array([1., 0.2, 0.0], dtype=float)
    df_0['EconomicCost'] = df_0['actions'].map(lambda x: eco_costs[x])*7.
    def Bed_cost(x):
        avail_hospital_beds = hbr*1e5
        return 0. if x < avail_hospital_beds else mhc
    df_0['bedcost'] = df_0['Infected'].map(lambda x: Bed_cost(x))
    df_0.head()
    Susceptible = df_0['Susceptible'].values
    dt           = 5/(24*60)
    Ts           = 7
    time_steps   = int((Ts) / dt)
    def ds_cost(Susceptible):
        ds = []
        l            = np.shape(Susceptible)[0]
        weakly_sus = Susceptible[::time_steps]
        # print(len(weakly_sus))
        for i in range(24):
            for _ in range(time_steps):
                ds.append(weakly_sus[i]-weakly_sus[i+1])
        for _ in range(time_steps):
            ds.append(weakly_sus[i]-weakly_sus[i+1])
        return ds
    df_0['Ds_cost'] = np.array(ds_cost(Susceptible))/hcs
    df_0['PublicHealthCost'] = df_0['bedcost'] + df_0['Ds_cost']
    df_0['TotalCost'] = (1-0.5)*df_0['PublicHealthCost'] + (0.5)*df_0['EconomicCost']
    TotalCost = df_0['TotalCost'].values
    TotalWeeklyCosts = TotalCost[::time_steps]
    TotalWeeklyCumulativeCosts = discount_reward(TotalWeeklyCosts, GAMMA=0.99)
    TotalDialyCumulativeCosts = []
    for _ in TotalWeeklyCumulativeCosts:
        for t in range(time_steps):
            TotalDialyCumulativeCosts.append(_)
    df_0['CumulativeCost'] = TotalDialyCumulativeCosts
    index = pd.date_range("2020-05-15 00:00:00", "2020-11-05 23:55:00", freq="5min")
    main_title = "weight = " + str(0.5) + ", "
    ax = df_0[['Susceptible', 'Exposed', 'Infected', 'Recovered', 'actions']].plot.line(subplots=True, figsize = (5,5))
    ax[0].set_ylim([94000, 100000])
    ax[1].set_ylim([0, 500])
    ax[2].set_ylim([0, 1500])
    ax[2].axhline(y=hbr*1e5)
    ax[3].set_ylim([0, 5000])
    ax[4].set_ylim([-0.1, 2.1])
    plt.savefig(path + main_title+"Scenario - " + str(scenario) + " - states.png")
    # plt.show()
    plt.close()

    index = pd.date_range("2020-05-15 00:00:00", "2020-11-05 23:55:00", freq="5min")
    main_title = "weight = " + str(0.5) + ", "
    ax = df_0[['PublicHealthCost', 'EconomicCost', 'TotalCost', 'CumulativeCost']].plot.line(subplots=True, figsize = (5,5))
    # ax[0].set_ylim([94000, 100000])
    # ax[1].set_ylim([0, 500])
    ax[2].set_ylim([-0.1, 7.5])
    # ax[3].set_ylim([-0.1, 2.1])
    plt.savefig(path + main_title+"Scenario - " + str(scenario) + " - costs.png")
    # plt.show()
    plt.close()
    return TotalWeeklyCumulativeCosts[0]