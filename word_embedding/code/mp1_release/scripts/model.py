
import torch.nn as nn
import torch
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import os


class CBOW_MODEL(nn.Module):
    
    def __init__(self,V_size,E_size):
        super().__init__()
        self.Embedding = nn.Embedding(V_size,E_size)
        self.Linear = nn.Linear(E_size,V_size,bias=False)
        
    def forward(self,input):
        # print('input shape',input.shape)
        h = self.Embedding(input)
        # print('after embedding shape',h.shape)
        # print(h)
        h = h.mean(1)
        # print('after sum shape',h.shape)
        # print(h)
        out = self.Linear(h)
        # print('after classification',out.shape)
        # print(out)
        return(out)

