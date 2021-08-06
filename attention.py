import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC


class Norm(nn.Module):
    def __init__(self, dim_seq, input_size, eps = 1e-6):
        super().__init__()
    
        self.size = input_size
        self.seq = dim_seq
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones((self.size, self.seq)))
        self.bias = nn.Parameter(torch.zeros((self.size, self.seq)))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True))/ (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class Attention(nn.Module):
    def __init__(self, dim_seq, input_size, dropout=0.1):
        super().__init__()
        self.dim_seq = dim_seq
        self.dk = input_size
        
        self.q_linear = nn.Linear(dim_seq, dim_seq)
        self.k_linear = nn.Linear(dim_seq, dim_seq)
        self.v_linear = nn.Linear(dim_seq, dim_seq)
        
        self.norm_1 = Norm(dim_seq, input_size)
        self.norm_2 = Norm(dim_seq, input_size)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, S):
        S = self.norm_1(S).float()
        q = self.k_linear(S)
        k = self.k_linear(S)
        v = self.v_linear(S)
        scores = torch.matmul(q, k.transpose(-2,-1)) / np.sqrt(self.dk)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_1(scores)
        output = torch.matmul(scores, v)
        S = self.norm_2(S + self.dropout_2(output))
        return S


class AttentionModel(nn.Module, ABC):
    def __init__(self, dim_seq, input_size, output_size, n_hid=128):
        super().__init__()
        self.attn_head = Attention(dim_seq, input_size)
        self.model = nn.Sequential(
            nn.Linear(dim_seq*input_size, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, output_size),
        )

    def forward(self, input_tensor):
        S = self.attn_head(input_tensor.clone().detach().requires_grad_(True))
        X = torch.flatten(S, start_dim=-2).float()
        return self.model(X)