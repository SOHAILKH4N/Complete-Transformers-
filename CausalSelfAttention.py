import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from dataclasses import dataclass


class LayerNorm(nn.Module):
    '''LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False'''
    def __init__(self , ndim , bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self , input):
        return F.layer_norm(input , self.weight.shape , self.weight , self.bias , 1e-5)




class CausalSelfAttention(nn.Module):
    def __init__(self , config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd , 3 * config.n_embd , bias = config.bias) 

        # Output Projection
        self.c_proj = nn.Linear(config.n_embd , config.n_embd , bias = config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash attention
        self.flash = hasattr(torch.nn.functional , 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias" , torch.tril(torch.ones(config.block_size , config.block_size)).view(1 , 1 , config.block_size , config.block_size))


    def forward(self , x ):
        B , T , C = x.size()

        #Calculate query , key , values for all heads in batch and move head forward to be the batch dim
        q , k , v = self.c_attn(x).split(self.n_embd , dim = 2)

        k = k.view( B , T , self.n_head , C // self.n_head).transpose(1 , 2)
        v = v.view( B , T , self.n_head , C // self.n_head).transpose(1 , 2)
        q = q.view( B , T , self.n_head , C // self.n_head).transpose(1 , 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(q , k , v , attn_mask = None if self.flash else self.bias[: , : , :T , :T] , dropout_p = self.dropout if self.training else 0.0 , is_causal = True)
        y = y.transpose(1 , 2).contiguous().view(B , T , C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y 