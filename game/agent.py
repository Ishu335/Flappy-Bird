import flappy_bird_gymnasium
import gymnasium as gym
import torch
from dqn import DQN
from experiance_replay  import ReplayMemory  # we have too use this at trining time
import itertools
import yaml
import random
import torch.nn as nn
import torch.optimizer as optim

if torch.backend.mps.is_available():
    device="mps"
elif torch.backend.cuda.is_available():
    device="cuda"
else:
    device="cpu"

class Agent():
    def __int__(self,param_set):
        self.param_set=param_set
        with open("parameter.yaml","r") as f:
            all_param_set=yaml.safe_load(f)
            param=all_param_set[param_set]
  
        self.epsilon_min=param["epsilon_min"]
        self.epsilon_init=param["epsilon_init"]
        self.epsilon_decay=param["epsilon_decay"]
        self.replay_memory_size=param["replay_memory_size"]
        self.min_batch_size=param["min_batch_size"]
        self.network_sync_rate=param["network_sync_rate"]
        self.alpha=param["alpha"]
        self.gamma=param["gamma"]
        self.reward_threshold=param["reward_threshold"]

        self.loss_fn=nn.MSELoss()
        self.optimizer=None
    
    def run(self,is_training=True,render=False):

        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        #policy network
        new_state=env.observation_space.shape[0]    #input dim
        new_action=env.action_space.n #output dim

        policy_dqn=DQN(new_state,new_action)

        if is_training:
            memory=ReplayMemory(self.replay_memory_size) # we take static size of memory  
            epsilon=self.epsilon_init
            
            #target network
            target_dqn=DQN(new_state,new_action).to(device)
            #copy wight and bies values from policy network => target_network
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps=0

            # Optimizer
            self.optimizer=optim.Adam(policy_dqn.parameters(),lr=self.alpha)

        for episode in itertools.count():
            state, _ = env.reset()
            state=torch.tensor(state,dtype=torch.float,device=device)

            episode_rewards=0
            terminated=False

            while not terminated: 
                if is_training and random.random()<epsilon:
                    # Next action:
                    # (feed the observation to your agent here)
                    action = env.action_space.sample() # exploration
                    action=torch.tensor(action,dtype=torch.long,device=device)
                else:
                    with torch.no_grad():
                        action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() #exploit

                # Processing:
                obs, reward, terminated, _, _ = env.step(action)

                #store the experiance in Experiance Replay in training Mode 
                if is_training:
                    memory.append(( state,action,new_state,reward,terminated))
                    steps+=1

                state=new_state
                episode_rewards+=reward

            print(f"for Episode Rewards ={episode+1} with total rewards ={episode_rewards} epsilon={epsilon}")
            
            if is_training:
                # epsilon decay is done when training part not at testing part
                epsilon.max(epsilon*self.epsilon_decay,self.epsilon_min)

            if is_training and len(memory)>self.min_batch_size:
                #get sample
                min_batch=memory.sample(self.min_batch_size)

                optimize(min_batch,policy_dqn,target_dqn)

                #sync the networks 
                if steps >self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    # re-initalize
                    steps=0
        # env.close()

    def optimize(self,min_batch,policy_dqn,target_dqn):
        #get Experiance =>mini batch
        for state,action,next_state,reward,terminated in min_batch:
            if terminated:
                target=reward
            else:
                with torch.no_grad():
                    target_q=reward+self.gamma*target_dqn(next_state).max()  # this Y-True
            current_q=policy_dqn(state) # this predict Y

            #loss =MSELoss
            loss=self.loss_fn(current_q,target_q)             

            self.optimizer.zero_grad()
            loss.backward() #back-propation
            self.optimizer.step() #update wight and bieas vlues