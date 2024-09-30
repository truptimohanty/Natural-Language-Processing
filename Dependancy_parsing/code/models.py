import torch 
import torch.nn as nn 

class PARSE_MODEL_CONCAT(nn.Module):
    
    def __init__(self,demb,dpos,h,T,P,cwindow):
        """ 
        demb = embidding dimension after Glove 
        dpos = pos embedding dimension 
        h = hidden layer dimension = 200 
        T = total number of actions = 75 
        P = pos vocabulary dimension for pos = 18 
        """
        super().__init__()
        
        self.Embedding = nn.Embedding(P,dpos)
        self.Linear_pos = nn.Linear(2*cwindow*dpos,h,bias=True)
        self.Linear_wemb = nn.Linear(2*cwindow*demb,h,bias=True)
        self.Linear_final = nn.Linear(h,T,bias=True)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self,W,Pos):
        # print('pos shape',Pos.shape)
        # print('W shape',W.shape)

        pemb = self.Embedding(Pos)
        pemb = torch.flatten(pemb,start_dim=1)

        # print('after pos embedding',pemb.shape)
        # # print(pemb)

        hpemb = self.Linear_pos(pemb)

        # print('after hpemb',hpemb.shape)
        # # print(hpemb)

        hwemb = self.Linear_wemb(W)

        # print('after hWemb',hwemb.shape)
        # # print(hwemb)

        hrep1 = hwemb+hpemb

        # print('hrep1.shape',hrep1.shape)

        hrep = self.relu(hrep1)

        # print('after adding both and ReLu',hrep.shape)

        scores = self.Linear_final(hrep)
        
        # print('scores shape', scores.shape)
        # y = self.softmax(scores)
        # print(y.shape)
        return scores
    

class PARSE_MODEL_MEAN(nn.Module):
    
    def __init__(self,demb,dpos,h,T,P):
        """ 
        demb = embidding dimension after Glove 
        dpos = pos embedding dimension 
        h = hidden layer dimension = 200 
        T = total number of actions = 75 
        P = POS vocabulary dimension for pos = 18 
        """
        super().__init__()
        
        self.Embedding = nn.Embedding(P,dpos)
        self.Linear_pos = nn.Linear(dpos,h,bias=True)
        self.Linear_wemb = nn.Linear(demb,h,bias=True)
        self.Linear_final = nn.Linear(h,T,bias=True)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self,W,Pos):

        # print('pos shape',Pos.shape)
        # print('W shape',W.shape)
        
        pemb = self.Embedding(Pos)
        pemb = pemb.mean(1)

        # print('after pos embedding',pemb.shape)
        # # print(pemb)

        hpemb = self.Linear_pos(pemb)

        # print('after hpemb',hpemb.shape)
        # print(hpemb)

        hwemb = self.Linear_wemb(W)

        # print('after hWemb',hwemb.shape)
        # print(hwemb)

        hrep1 = hwemb+hpemb

        # print('hrep1.shape',hrep1.shape)

        hrep = self.relu(hrep1)
        # print('after adding both and ReLu',hrep.shape)
        
        scores = self.Linear_final(hrep)
        # print('scores shape', scores.shape)
        # y = self.softmax(scores)
        # print(y.shape)
        return scores
    
