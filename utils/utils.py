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
from stable_baselines3 import PPO
from sklearn.metrics import confusion_matrix


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
        'weight' : w
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