import flappy_bird_gymnasium
import gymnasium as gym
import torch
from dqn import DQN
from experiance_replay  import ReplayMemory  # we have too use this at trining time
import itertools
import yaml
import random
import torch.nn as nn
import torch.optim as optim
import os
import argparse

if torch.backends.mps.is_available():
    device="mps"
elif torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"


RUNS_DIR="runs"
os.makedirs(RUNS_DIR,exist_ok=True)


class Agent:
    def __init__(self,param_set):
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

        self.LOG_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.log") # storing parameter .log
        self.MODEL_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.pt") # storing model .pt when we store this  model when we get best rewards
    
    def run(self,is_training=True,render=False):

        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        #policy network
        next_state=env.observation_space.shape[0]    #input dim
        new_action=env.action_space.n #output dim

        policy_dqn=DQN(next_state,new_action)

        if is_training:
            memory=ReplayMemory(self.replay_memory_size) # we take static size of memory  
            epsilon=self.epsilon_init
            
            #target network
            target_dqn=DQN(next_state,new_action).to(device)
            #copy wight and bies values from policy network => target_network
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps=0

            # Optimizer
            self.optimizer=optim.Adam(policy_dqn.parameters(),lr=self.alpha)
            best_reward=float("-inf")
        else:
            # load best model means load best policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()


        for episode in itertools.count():
            state, _ = env.reset()
            state=torch.tensor(state,dtype=torch.float,device=device)

            episode_reward=0
            terminated=False

            while (not terminated and episode_reward <self.reward_threshold): 
                if is_training and random.random()<epsilon:
                    # Next action:
                    # (feed the observation to your agent here)
                    action = env.action_space.sample() # exploration
                    action=torch.tensor(action,dtype=torch.long,device=device)
                else:
                    with torch.no_grad():   
                        action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() #exploit

                # Processing:
                next_state, reward, terminated, _, _ = env.step(action.item())

                episode_reward+=reward

                #create tesnors
                reward=torch.tensor(reward,dtype=torch.float,device=device)
                next_state=torch.tensor(next_state,dtype=torch.float,device=device)

                if is_training:
                    memory.append(( state,action,next_state,reward,terminated))
                    steps+=1

                state=next_state
                

            print(f"for Episode Rewards ={episode+1} with total rewards ={episode_reward}  episode={episode}")
            
            if is_training:
                # epsilon decay is done when training part not at testing part
                epsilon=max(epsilon*self.epsilon_decay,self.epsilon_min)
                if episode_reward> best_reward:
                    log_msg=f"best rewards ={episode_reward} for episode={episode+1}"
                    with open(self.LOG_FILE,"a")as f:
                        f.write(log_msg+"\n")
                    torch.save(policy_dqn.state_dict(),self.MODEL_FILE)
                    best_reward=episode_reward

            if is_training and len(memory)>self.min_batch_size:
                #get sample
                min_batch=memory.sample(self.min_batch_size)

                self.optimize(min_batch,policy_dqn,target_dqn)

                #sync the networks 
                if steps >self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    # re-initalize
                    steps=0
        # env.close()

    def optimize(self,min_batch,policy_dqn,target_dqn):

        #get batch of experiiences
        states,actions,next_states,rewards,terminations=zip(*min_batch)

        states=torch.stack(states)
        actions=torch.stack(actions)
        next_states=torch.stack(next_states)
        rewards=torch.stack(rewards)
        terminations=torch.tensor(terminations).float().to(device)

       # calculate target Q-values if terminations=true > zero
        with torch.no_grad():
               target_q = rewards + (1-terminations) * self.gamma *target_dqn(next_states).max(dim=1)[0]
        # calculate y pred 1.e. Q-value from current policy
        current_q= policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        #compute loss
        loss = self.loss_fn(current_q, target_q)
        #optimize model
        self.optimizer.zero_grad()
        loss.backward()


if __name__ == "__main__":

    #Parse command line inputs
    parser= argparse.ArgumentParser (description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()
    dql = Agent (param_set=args.hyperparameters)
    if args.train:
         dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)