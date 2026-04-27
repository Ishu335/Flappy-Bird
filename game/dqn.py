import torch
import torch.nn as  nn

class DQN(nn.Module):        

    #input dim  ,output dim, hidden dim
    def __init__(self,state_din=12,action_din=12,   hidden_din=256):
        super(DQN,self).__init__()  
        self.model=nn.Sequential(

                    nn.Linear(state_din,hidden_din),
                    nn.ReLU(),

                    nn.Linear(hidden_din,action_din )
                )
        
    def forward(self,x):
        return  self.model(x)