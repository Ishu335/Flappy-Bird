import flappy_bird_gymnasium
import gymnasium as gym
import torch
from dqn import DQN
from experiance_replay  import ReplayMemory  # we have too use this at trining time
import itertools
import yaml
import torch.nn as nn
import torch.optimizer as optim

if torch.backend.mps.is_available():
    device="mps"
elif torch.backend.cuda.is_available():
    device="cuda"
else:
    device="cpu"

def run(self,is_training=True,render=False):

    env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

    #policy network
    new_state=env.observation_space.shape[0]    #input dim
    new_action=env.action_space.n #output dim

    policy_dqn=DQN(new_state,new_action)

    if is_training:
        memory=ReplayMemory(10000) # we take static size of memory  

    for episode in itertools.count():
        state, _ = env.reset()
        episode_rewards=0

        while not terminated: 
            # Next action:
            # (feed the observation to your agent here)
            action = env.action_space.sample()

            # Processing:
            obs, reward, terminated, _, _ = env.step(action)

            #store the experiance in Experiance Replay in training Mode 
            if is_training:
                memory.append(( state,action,new_state,reward,terminated))

            state=new_state
            episode_rewards+=reward   

        print(f"for Episode Rewards ={episode+1} with total rewards ={episode_rewards}")
    # env.close()