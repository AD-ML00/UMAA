import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import math

class PositionalEncoding(nn.Module):
    
    def __init__(self, seq_len, d_model, n, device):
        
        super(PositionalEncoding, self).__init__() # nn.Module 초기화
        
        # encoding : (seq_len, d_model)
        self.encoding = torch.zeros(seq_len, d_model, device=device)
        self.encoding.requires_grad = False
	
        # (seq_len, )
        pos = torch.arange(0, seq_len, device=device)
        # (seq_len, 1)         
        pos = pos.float().unsqueeze(dim=1) # int64 -> float32         
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        _2i2 = torch.arange(1, d_model, step=2, device=device).float()
        self.encoding[:, ::2] = torch.sin(pos / (n ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (n ** (_2i2 / d_model)))
        

    def forward(self, x):
        # x.shape : (batch, seq_len) or (batch, seq_len, d_model)
        seq_len = x.size()[1] 
        # return : (seq_len, d_model)
        # return matrix will be added to x by broadcasting
        return self.encoding[:seq_len, :]