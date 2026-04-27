from collections import deque
import random

class   ReplayMemory():

    def __init__(self,maxlen,seed=None):
        self.memory=deque([],maxlen=maxlen)

    def append(self,new_exp):
        self.memory.append(new_exp)

    def sample(self,smaple_size):
        return random.sample(self.memory,smaple_size)
    
    #current duffere size
    def __len__(self):
        return len(self.memory)
     
