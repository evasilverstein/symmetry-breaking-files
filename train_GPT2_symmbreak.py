### ES: now trying 124M model, but changing some things like batch and context length to save memory

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import os
import time 
import wandb
import time
import numpy as np
from ECD_q1_scaled import ECD_q1_scaled
from torch.optim.optimizer import Optimizer, required

if torch.cuda.is_available():
    from flash_attn import flash_attn_func #this additionally requires the installation of flash attention: pip install flash-attn --no-build-isolation

import wandb

    # Log in to Weights & Biases
    # If WANDB_API_KEY environment variable is set, it will be used automatically.
    # Otherwise, you will be prompted to enter your API key in the console.
    #wandb.login()

    # Alternatively, you can pass the API key directly (less secure for production)
wandb.login(key="") 

terminate_pod_at_the_end = True #False #To kill the runpod pod at the end of the execution of this file

initial_time = time.time()
train = True                #If true trains the model, if False but "inference" below is "True" the code will load the saved model and perform inference

#corpus = 'sherlock'         #This is ~0.8M tokens, and it's already included in this repo
#corpus = 'shakespeare'     #This is a even smaller, and it's already included in this repo
corpus = 'fineweb'         #This is 10B tokens, and needs to be separately downloaded, it's on runpod already

valid_every = 50   #compute the validation loss every valid_every iterations
save_every = 10000000000000 #250    #save the model every save_every iterations
fraction_dataset = 0.06 #0.02  #fraction of the dataset to train over before exiting

inference = False #Perform inference on the saved model

small_model = False #True #If true uses a small model defined below

#Set the following to True if the GPU supporst bfloat16.
#Ampere class GPUs (the ones that start with A) support it
#Set it if it's available since it could allow up to an 8x speed-up
has_bfloat16 = True #Need to be set False on sherlock 

##Set to false if you are having errors with compilation
compile = True

#-------------------- Optimizer Config --------------------#
optimizer_type = 'ECD' #'adam'
learning_rate= 0.5 #1e-1
eta= 100  ###100 beat adam at both scales, now trying higher since we're doing larger model
F0=2 #-1   #5
nu=0

#-------------------- Defining the model --------------------#

##The numbers below are for the 124M GPT-2 model
if small_model == False:
    @dataclass
    class GPTConfig:
        context_length: int = 512 #1024      #Maximum context length
        vocab_size:     int = 50257     #Vocabulary dimension, depends on tokenizer
        n_layer:        int = 12        #Number of layers
        n_head:         int = 12        #Number of attention heads
        n_embd:         int = 768       #64*n_head , number of embedding dimension
else:
    @dataclass
    class GPTConfig:
        context_length: int = 256       #Maximum context length
        vocab_size:     int = 50257     #Vocabulary dimension, depends on tokenizer
        n_layer:        int = 2         #Number of layers
        n_head:         int = 2         #Number of attention heads
        n_embd:         int = 8         #4*n_head , number of embedding dimension





### Now putting in actual rotation-breaking element, also with causality

###This version is explicit breaking, now with disorder-inspired full rotation breaking via the biases


class DisorderedCausalSelfAttention(nn.Module):
    """
    Per-batch 'disorder' biases for Q and K that are non-trainable but re-sampled.
    Shapes:
      - mean_Q, mean_K, std_Q, std_K are 1-D tensors (d,)
      - bQ, bK are buffers (n_head, d), re-sampled each batch when training
    """
    def __init__(self, config,
                 mean_Q=None, mean_K=None,     # scalar or (d,)
                 std_Q=None,  std_K=None,      # scalar or (d,)
                 mode="per_batch", impl='eager'):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d = config.n_embd // config.n_head
        C = config.n_embd

        # fused qkv and output projection
        self.c_attn = nn.Linear(C, 3 * C)
        self.c_proj = nn.Linear(C, C)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.impl = impl

        # helper: coerce scalars/arrays to 1-D vectors (d,)
        def to_vec(x, default):
            if x is None:
                x = default
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            if x.ndim == 0:
                x = x.repeat(self.d)
            return x.to(torch.float32).view(self.d)

        # defaults chosen to break symmetry in expectation (anisotropic stds, two means)
        mean_Q = to_vec(mean_Q, 0.5)                                 # (d,)
        mean_K = to_vec(mean_K, 0.3)                                 # (d,)
        std_Q  = to_vec(std_Q,  torch.linspace(0.05, 0.15, self.d))  # (d,)
        std_K  = to_vec(std_K,  torch.linspace(0.12, 0.08,  self.d)) # (d,)
        # std_Q  = to_vec(std_Q,  torch.linspace(0.3, 0.7, self.d))  # (d,)
        # std_K  = to_vec(std_K,  torch.linspace(0.5, 0.2,  self.d)) # (d,)

        # register as fixed buffers (non-trainable, move with .to(device))
        self.register_buffer("mean_Q", mean_Q)
        self.register_buffer("mean_K", mean_K)
        self.register_buffer("std_Q",  std_Q.abs())
        self.register_buffer("std_K",  std_K.abs())

        # current draws (n_head, d) used this batch/step
        self.register_buffer("bQ", torch.zeros(self.n_head, self.d))
        self.register_buffer("bK", torch.zeros(self.n_head, self.d))
        self.mode = mode  # "per_batch" or "off" (no re-sampling)

        if self.impl=='eager':
            # precompute a maximal causal mask
            mask = torch.tril(torch.ones(config.context_length, config.context_length))
            
            self.register_buffer(
                "causal_mask",
                mask.view(1, 1, config.context_length, config.context_length)
            )

    @torch.no_grad()
    def _resample_biases(self, device):
        """
        Draw one 1-D vector (d,) for Q and K, then tile across heads -> (n_head, d).
        Called per batch in training (or call from trainer once per epoch if you prefer).
        """
        mQ, sQ = self.mean_Q.to(device), self.std_Q.to(device)
        mK, sK = self.mean_K.to(device), self.std_K.to(device)

        base_Q = mQ + sQ * torch.randn_like(mQ)  # (d,)
        base_K = mK + sK * torch.randn_like(mK)  # (d,)

        self.bQ.copy_(base_Q.unsqueeze(0).expand(self.n_head, -1).contiguous())
        self.bK.copy_(base_K.unsqueeze(0).expand(self.n_head, -1).contiguous())

    def forward(self, x):
        B, T, C = x.shape
        nh, d = self.n_head, self.d

        # re-sample once per forward in training mode if enabled
        if self.training and self.mode == "per_batch":
            # avoid tracing in-graph buffer mutation with torch.compile
            torch._dynamo.graph_break()
            self._resample_biases(x.device)

        # QKV projections
        qkv = self.c_attn(x)                       # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)              # each (B, T, C)

        

        if self.impl=='eager':
            # reshape to heads: (B, nh, T, d)
            q = q.view(B, T, nh, d).transpose(1, 2)
            k = k.view(B, T, nh, d).transpose(1, 2)
            v = v.view(B, T, nh, d).transpose(1, 2)
            


            # add disorder biases (broadcast over batch/time): (1, nh, 1, d)
            q = q + self.bQ.view(1, nh, 1, d)
            k = k + self.bK.view(1, nh, 1, d)

            # causal attention
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)  # (B, nh, T, T)
            scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(scores, dim=-1)

            y = att @ v                                      # (B, nh, T, d)
            y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        else:
            q, k, v = tuple(map(lambda t: t.to(torch.bfloat16),(q, k, v)))
            # reshape to heads: (B, T, nh, d) for flash attn
            q = q.view(B, T, nh, d)
            k = k.view(B, T, nh, d)
            v = v.view(B, T, nh, d)
            # add disorder biases (broadcast over batch/time): (1, 1 nh, d)
            q = q + self.bQ.view(1, nh, 1, d).transpose(1,2)
            k = k + self.bK.view(1, nh, 1, d).transpose(1,2)
            y = flash_attn_func(q, k, v, casual=True)
            y = y.contiguous().view(B, T, C).to(x.dtype) # (B, T, C)



        return self.c_proj(y)


class RotationallyAsymmetricCausalSelfAttention(nn.Module):
    def __init__(
        self,
        config,
        asym_mean: float = 1.0,
        asym_std:  float = 0.7 #0.5, #0.3,  ### 0.3 did better than SSB case
    ):
        """
        config:        your GPTConfig
        asym_mean:     the mean of the fixed Normal() you sample from
        asym_std:      the std-dev of that Normal()
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head   = config.n_head
        self.head_dim = config.n_embd // config.n_head
        C = config.n_embd

        # QKV & output proj
        self.c_attn = nn.Linear(C, 3*C)
        self.c_proj = nn.Linear(C, C)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # draw fixed, non-trainable asymmetry tensors
        q_s = torch.randn(self.n_head, self.head_dim) * asym_std + asym_mean
        q_b = torch.randn(self.n_head, self.head_dim) * asym_std + asym_mean
        k_s = torch.randn(self.n_head, self.head_dim) * asym_std + asym_mean
        k_b = torch.randn(self.n_head, self.head_dim) * asym_std + asym_mean

        # register as buffers so they move with .to(device) but do NOT train
        self.register_buffer("q_scale", q_s)
        self.register_buffer("q_bias",  q_b)
        self.register_buffer("k_scale", k_s)
        self.register_buffer("k_bias",  k_b)

        # precompute max causal mask
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.context_length, config.context_length)
        )

    def forward(self, x):
        B, T, C = x.size()
        nh, hd = self.n_head, self.head_dim

        # 1) project to QKV
        qkv = self.c_attn(x)                # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)       # each (B, T, C)

        # 2) to (B, nh, T, hd)
        q = q.view(B, T, nh, hd).transpose(1,2)
        k = k.view(B, T, nh, hd).transpose(1,2)
        v = v.view(B, T, nh, hd).transpose(1,2)

        # 3) apply fixed asymmetry
        qs = self.q_scale.view(1, nh, 1, hd)
        qb = self.q_bias .view(1, nh, 1, hd)
        ks = self.k_scale.view(1, nh, 1, hd)
        kb = self.k_bias .view(1, nh, 1, hd)
        q  = q * qs + qb
        k  = k * ks + kb

        # 4) causal scaled-dot-product
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(hd)  # (B, nh, T, T)
        mask   = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # 5) softmax & attend
        att = F.softmax(scores, dim=-1)
        y   = att @ v                                       # (B, nh, T, hd)

        # 6) reassemble & project out
        y = y.transpose(1,2).reshape(B, T, C)
        return self.c_proj(y)

###  The following was the old learned (trained) asymmetry, while the above is explicit breaking instead
# class RotationallyAsymmetricCausalSelfAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         assert config.n_embd % config.n_head == 0

#         self.n_head  = config.n_head
#         self.head_dim= config.n_embd // config.n_head
#         C = config.n_embd

#         # fused QKV projection
#         self.c_attn = nn.Linear(C, 3*C)
#         # output projection
#         self.c_proj = nn.Linear(C, C)
#         self.c_proj.NANOGPT_SCALE_INIT = 1

#         # per-head Q/K scale & bias (rotation‐breaking)
#         self.q_scale = nn.Parameter(torch.ones(self.n_head, self.head_dim))
#         self.q_bias  = nn.Parameter(torch.zeros(self.n_head, self.head_dim))
#         self.k_scale = nn.Parameter(torch.ones(self.n_head, self.head_dim))
#         self.k_bias  = nn.Parameter(torch.zeros(self.n_head, self.head_dim))

#         # precompute a maximal causal mask (context_length × context_length)
#         mask = torch.tril(torch.ones(config.context_length, config.context_length))
#         self.register_buffer('causal_mask',
#             mask.view(1, 1, config.context_length, config.context_length)
#         )

#     def forward(self, x):
#         B, T, C = x.size()   # batch, seq_len, emb_dim
#         nh, hd = self.n_head, self.head_dim

#         # 1) project and split QKV
#         qkv = self.c_attn(x)                   # (B, T, 3C)
#         q, k, v = qkv.split(C, dim=2)          # each (B, T, C)

#         # 2) reshape into heads: (B, nh, T, hd)
#         q = q.view(B, T, nh, hd).transpose(1,2)
#         k = k.view(B, T, nh, hd).transpose(1,2)
#         v = v.view(B, T, nh, hd).transpose(1,2)

#         # 3) rotation‐breaking on Q and K
#         #    broadcast scales & biases over batch & seq
#         qs = self.q_scale.view(1, nh, 1, hd)
#         qb = self.q_bias .view(1, nh, 1, hd)
#         ks = self.k_scale.view(1, nh, 1, hd)
#         kb = self.k_bias .view(1, nh, 1, hd)

#         q = q * qs + qb
#         k = k * ks + kb

#         # 4) compute causal attention scores
#         scores = (q @ k.transpose(-2,-1)) / math.sqrt(hd)  # (B, nh, T, T)
#         mask = self.causal_mask[:,:,:T,:T]                  # (1,1,T,T)
#         scores = scores.masked_fill(mask == 0, float('-inf'))

#         # 5) softmax & attend
#         att   = F.softmax(scores, dim=-1)
#         y     = att @ v                                      # (B, nh, T, hd)

#         # 6) reassemble & project out
#         y     = y.transpose(1,2).contiguous().view(B, T, C)
#         return self.c_proj(y)



# -------------------------------------------------------------------------
# 2) Replace CausalSelfAttention with this asymmetric version
# -------------------------------------------------------------------------
class AsymmetricCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head  = config.n_head
        self.head_dim= config.n_embd // config.n_head
        C = config.n_embd

        # QKV fused projection
        self.c_attn = nn.Linear(C, 3*C)
        # output projection
        self.c_proj = nn.Linear(C, C)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # ——— new asymmetry parameters ———
        # per-head softmax temperature (start at 1/√d)
        init_temp = 1.0 / math.sqrt(self.head_dim)
        self.head_temp  = nn.Parameter(torch.ones(self.n_head) * init_temp)
        # per-head output scale (start at 1.0)
        self.head_scale = nn.Parameter(torch.ones(self.n_head))

    def forward(self, x):
        B, T, C = x.size()
        # QKV
        qkv = self.c_attn(x)                    # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)           # each (B, T, C)

        # reshape into (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1,2)

        # scaled‐dot‐product
        scores = q @ k.transpose(-2,-1)         # (B, n_head, T, T)
        # apply per-head temperature
        temp   = self.head_temp.view(1, self.n_head, 1, 1)
        scores = scores * temp
        att    = F.softmax(scores, dim=-1)
        y      = att @ v                        # (B, n_head, T, head_dim)

        # per-head gating
        gate = self.head_scale.view(1, self.n_head, 1, 1)
        y    = y * gate

        # reassemble & project
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.c_proj(y)


# -------------------------------------------------------------------------
# 3) Hook it into your GPT2 Block
# -------------------------------------------------------------------------
# class Block(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         # …
#         self.ln_1 = nn.LayerNorm(config.n_embd)
#         self.attn = AsymmetricCausalSelfAttention(config)  # <- swapped in
#         self.ln_2 = nn.LayerNorm(config.n_embd)
#         self.mlp  = MLP(config)                           # your asymmetric MLP

#     def forward(self, x):
#         x = x + self.attn(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x


# # -------------------------------------------------------------------------
# # 4) Training loop using ECD_q1_scaled
# # -------------------------------------------------------------------------
# model = GPT2Model(config)      # however you instantiate your GPT2
# optimizer = ECD_q1_scaled(
#     model.parameters(),
#     lr=0.01,
#     eta=1.0,
#     weight_decay=5e-5,
#     F0=0.0,
#     consEn=True
# )

# for epoch in range(epochs):
#     for batch in dataloader:
#         def closure():
#             optimizer.zero_grad()
#             logits = model(batch.input_ids)
#             loss   = loss_fn(logits, batch.labels)
#             loss.backward()
#             return loss
#         loss = optimizer.step(closure)
#     print(f"Epoch {epoch}  loss {loss.item():.4f}")



class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0   #Make sure that the dimensions are correctly specified

        #n_head*(key, query, value) all put together in a single matrix for efficiency, will split the different elements below
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)

        #projection back to the residual stream
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 ##This is just an attribute we will use to initialize this layer in a proper way

        #The mask needed to block out the 'future' tokens.
        #The name 'bias' is misleading, but it's the one used in the GPT2 paper and we are following through so that we can load their weights
        #We don't need this anumore with flash attention
        #self.register_buffer("bias", torch.tril(torch.ones(config.context_length, config.context_length))
        #                             .view(1,1,config.context_length, config.context_length))
        self.n_embd = config.n_embd
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size() #batch size, sequence length, embedding dimensionality (n_embd)

        #Now extract the query, key, values from the big matrix in c_attn. 
        #nh is the number of heads, hs is the head size, and C = nh*ns
        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.n_embd, dim = 2) #split qkv, each of them already includes all the heads together

        ## Now I separate all the heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        #Now compute the attention matrix, in parallel for all the heads

        ## The 4 lines below are a manual implementation of the attention.
        ## They work as expected, but I am commenting them out and use pytorch's implemention
        # Pytorch's implementation is much faster since it uses Flash Attention
        #  
        #att = (q@k.transpose(-2,-1))/math.sqrt(k.size(-1)) # normalization so that the variance is preserved
        #att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # set as -infty all the elements in the upper triangular part (not causally connected)
        #att = F.softmax(att, dim = -1) # apply softmax to normalize the rows
        #y = att@v # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q,k,v, is_causal = True)
        
        y = y.transpose(1,2).contiguous().view(B, T, C) #Now reassemble all the head outputs side by side, to be fed to the MLP
        #final output projection
        y = self.c_proj(y)
        
        return y



# class AsymmetricCausalSelfAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         assert config.n_embd % config.n_head == 0

#         self.n_head = config.n_head
#         self.head_dim = config.n_embd // config.n_head

#         # fused QKV projection
#         self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
#         # output projection
#         self.c_proj = nn.Linear(config.n_embd, config.n_embd)
#         self.c_proj.NANOGPT_SCALE_INIT = 1

#         # --- learnable asymmetry parameters ---
#         # one temperature per head (initialized so that 1/√dk is the starting temp)
#         init_temp = 1.0 / math.sqrt(self.head_dim)
#         self.head_temp = nn.Parameter(torch.ones(self.n_head) * init_temp)

#         # one output‐scale per head (initialized to 1.0)
#         self.head_scale = nn.Parameter(torch.ones(self.n_head))

#     def forward(self, x):
#         B, T, C = x.size()

#         # 1) project to QKV and split
#         qkv = self.c_attn(x)                     # (B, T, 3*C)
#         q, k, v = qkv.split(C, dim=2)            # each (B, T, C)

#         # 2) reshape into heads
#         # (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
#         q = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)
#         k = k.view(B, T, self.n_head, self.head_dim).transpose(1,2)
#         v = v.view(B, T, self.n_head, self.head_dim).transpose(1,2)

#         # 3) scaled‐dot‐product with per‑head temperature
#         #    q @ kᵀ → (B, n_head, T, T)
#         scores = q @ k.transpose(-2, -1)
#         # apply each head’s temperature: broadcast head_temp over batch & positions
#         # head_temp: (n_head,) → (1, n_head, 1, 1)
#         temp = self.head_temp.view(1, self.n_head, 1, 1)
#         scores = scores * temp

#         # 4) causal mask + softmax
#         att = F.softmax(scores, dim=-1, dtype=scores.dtype)

#         # 5) attention output
#         y = att @ v  # (B, n_head, T, head_dim)

#         # 6) per‑head output gating
#         # head_scale: (n_head,) → (1, n_head, 1, 1)
#         gate = self.head_scale.view(1, self.n_head, 1, 1)
#         y = y * gate

#         # 7) re‑assemble and project
#         y = y.transpose(1,2).contiguous().view(B, T, C)
#         y = self.c_proj(y)

#         return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)      #Internal dimension is 4*n_embd
        self.gelu = nn.GELU(approximate = 'tanh')           #Using the approximate because it's what they use in the GPT2 paper, but it's not needed anymore
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)    #Projection back to the residual stream
        self.c_proj.NANOGPT_SCALE_INIT = 1 ##This is just an attribute we will use to initialize this layer in a proper way

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x



# -------------------------------------------------------------------------
# Option A: Custom channel‑wise LeakyReLU for 1D features
# -------------------------------------------------------------------------
class ChannelWiseLeakyReLU1d(nn.Module):
    def __init__(self, features, init_slope=0.2, slope_std=0.05):
        """
        features: number of hidden neurons (4 * n_embd)
        init_slope: mean of the initial leak slopes
        slope_std:  standard deviation for initializing different slopes
        """
        super().__init__()
        # one learnable slope per neuron
        self.slopes = nn.Parameter(
            torch.randn(features) * slope_std + init_slope
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        pos = F.relu(x)
        neg = torch.clamp(x, max=0)
        # reshape slopes to [1,1,features] so broadcasts over batch & seq
        s = self.slopes.view(1, 1, -1)
        return pos + s * neg


# -------------------------------------------------------------------------
# Option B: Use PyTorch’s PReLU but randomize each slope at init
# -------------------------------------------------------------------------
# class RandomPReLU1d(nn.PReLU):
#     def __init__(self, features, init_slope=0.2, slope_std=0.05):
#         # num_parameters=features => one weight per hidden neuron
#         super().__init__(num_parameters=features, init=init_slope)
#         # now randomly perturb each slope so they’re all distinct
#         with torch.no_grad():
#             self.weight.mul_(0)  # zero out the default
#             self.weight.add_(torch.randn_like(self.weight) * slope_std + init_slope)

class RandomPReLU1d(nn.Module):
    def __init__(self, features, init_slope=0.2, slope_std=0.05):
        super().__init__()
        # one slope per hidden neuron
        weight = torch.randn(features) * slope_std + init_slope
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        # x: [B, T, features]
        # permute so features→channels
        x2 = x.transpose(1, 2)    # now [B, features, T]
        y2 = F.prelu(x2, self.weight)
        return y2.transpose(1, 2) # back to [B, T, features]



# -------------------------------------------------------------------------
# Asymmetric MLP replacing the original GELU
# -------------------------------------------------------------------------
class AsymmetricMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd

        self.c_fc   = nn.Linear(config.n_embd, hidden)
        # pick one of the two:
        self.act    = ChannelWiseLeakyReLU1d(hidden, init_slope=0.2, slope_std=0.5)
        #self.act = RandomPReLU1d(hidden, init_slope=0.2, slope_std=0.5)

        self.c_proj = nn.Linear(hidden, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)      # [B, T, 4*n_embd]
        x = self.act(x)       # per‑neuron asymmetric activation
        x = self.c_proj(x)    # back to [B, T, n_embd]
        return x

class AsymmetricMLPPreLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden)
        self.act  = RandomPReLU1d(hidden, init_slope=0.2, slope_std=1.0)
        self.c_proj = nn.Linear(hidden, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x) 
        x = self.act(x)     # now works cleanly
        x = self.c_proj(x)
        return x



class Block(nn.Module):
    """ A transformer block: communication followed by computation """

    def __init__(self, config):                         
        super().__init__()
        #self.sa = CausalSelfAttention(config)           ##This is the multi head attention
        #self.sa = RotationallyAsymmetricCausalSelfAttention(config)  ## asymmetrized (now rotationally) attn
        self.sa = DisorderedCausalSelfAttention(config, impl='flash')  ## asymmetrized (now fully via disorder) rotationally attn
        #self.sa = AsymmetricCausalSelfAttention(config)            
        #self.mlp = MLP(config)
        #self.mlp = AsymmetricMLP(config)
        self.mlp = AsymmetricMLPPreLU(config)

        self.ln1 = torch.nn.LayerNorm(config.n_embd)    ##Layer norms: normalize the inputs to be dev std 1 and mean 0 at initialization
        self.ln2 = torch.nn.LayerNorm(config.n_embd)    ##Layer norms: normalize the inputs to be dev std 1 and mean 0 at initialization

    def forward(self, x):
        # A difference with the original transformer paper is that we are applying the layer norms _before_ the compuations
        x = x + self.sa(self.ln1(x))    ##Adding also residual connections
        x = x + self.mlp(self.ln2(x))   ##Adding also residual connections
        return x



class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),                   #Token embedding matrix
            wpe = nn.Embedding(config.context_length, config.n_embd),               #Position embedding matrix
            h = nn.ModuleList( [Block(config) for _ in range(config.n_layer)]),     #Here all the blocks repeated n_layer times
            ln_f = nn.LayerNorm(config.n_embd),                                     #Final Layer norm
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)    #Final linear layer that will produce the logits (with no bias as per the GPT paper) 
        
        #Weight sharing scheme: the performance improves if the final linear layer is exactly the same as the token embedding matrix
        #It's good because these are a lot of parameters
        self.transformer.wte.weight = self.lm_head.weight

        #Apply the custom initialization of some of the parameters
        self.apply(self._init_weights)

    #Custom initialization of some of the parameters, according to the original GPT2 paper and code
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 #roughly similar to the Xavier, but we are manually enforcing in this way to be close to the original GPT
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer)**(-0.5) #dividing by the sqrt of number of layers in the forked path, before merging in the residual stream 
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, targets = None):
        #idx is the current contex, with shape (B, T)
        B, T = idx.size()
        assert T <= self.config.context_length, f"Cannot process a sequence of length {T}, context length is {self.config.context_length}"
        
        #Position embedding
        pos = torch.arange(0, T, dtype = torch.long , device = idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_emb)

        #Token embedding
        tok_emb = self.transformer.wte(idx) #(B, T, n_emb)

        # put information of the token and of the position together
        x = tok_emb+pos_emb  #(B, T, n_embd), broadcasting happened

        #now process through the various blocks
        for block in self.transformer.h:
            x = block(x)
        
        #final layer norm and linear transformation
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)

        loss = None
        #The views are just collapsing (B,T) to B*T, since cross_entropy only accepts a single batch dimension
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1) )
        return logits, loss




#-----------------------------Data Loader --------------------------------------#
import tiktoken
##Uses a different data loader for the simpler sherlock/shakespear datasets or the fine web dataset

if (corpus == 'sherlock') or (corpus == 'shakespeare'):
    ##Loads all the dataset in memory
    class DataLoaderLite():
        def __init__(self, B, T, corpus = 'sherlock', split = 'train'):
            self.B = B
            self.T = T
            #At initialization of the data loader we read the tokens and store them in memory
            if corpus == 'shakespeare':
                with open('shakespeare.txt', 'r', encoding = 'utf-8') as f:
                    text = f.read()

            if corpus == 'sherlock':
                with open('sherlock.txt', 'r', encoding = 'utf-8') as f:
                    text = f.read()
            
            n = int((0.9*len(text)))

            enc = tiktoken.get_encoding('gpt2')

            if split == 'train':
                train_text = text[:n]
                self.tokens = torch.tensor(enc.encode(train_text))
                print(f"Loaded {len(self.tokens)} train tokens")
                print(f"1 train epoch is {len(self.tokens)//(B*T)} batches")


            elif split == 'val':
                val_text = text[n:]
                self.tokens = torch.tensor(enc.encode(val_text))
                print(f"Loaded {len(self.tokens)} val tokens")
                print(f"1 validation epoch is {len(self.tokens)//(B*T)} batches")
                
            else: raise ValueError("Wrong split specification, it has to be train or val")
        
        
            #state #We want to go over the full dataset
            self.current_position = 0
            

        def next_batch(self):
            B, T = self.B, self.T
            buf = self.tokens[self.current_position: self.current_position+B*T+1]
            x = buf[:-1].view(B, T) #inputs
            y = buf[1:].view(B, T)  #targets

            self.current_position += B*T

            #if we already went over the whole dataset reset the counter
            if self.current_position+(B*T+1) > len(self.tokens):
                self.current_position = 0

            return x, y


    class DataLoaderRandom:
        def __init__(self, B, T, corpus = 'sherlock', split = 'train'):
            self.B = B
            self.T = T
        
            #At initialization of the data loader we read the tokens and store them in memory
            if corpus == 'shakespeare':
                with open('shakespeare.txt', 'r', encoding = 'utf-8') as f:
                    text = f.read()

            if corpus == 'sherlock':
                with open('sherlock.txt', 'r', encoding = 'utf-8') as f:
                    text = f.read()
            
            n = int((0.9*len(text)))

            enc = tiktoken.get_encoding('gpt2')

            if split == 'train':
                train_text = text[:n]
                self.tokens = torch.tensor(enc.encode(train_text))
                print(f"Loaded {len(self.tokens)} train tokens")
                print(f"1 train epoch is {len(self.tokens)//(B*T)} batches")


            elif split == 'val':
                val_text = text[n:]
                self.tokens = torch.tensor(enc.encode(val_text))
                print(f"Loaded {len(self.tokens)} val tokens")
                print(f"1 validation epoch is {len(self.tokens)//(B*T)} batches")
                
            else: raise ValueError("Wrong split specification, it has to be train or val")

            
        def next_batch(self):

            B, T = self.B, self.T
            n = torch.randint(len(self.tokens)-B*T-2,(1,)).item()
            buf = self.tokens[n: n+B*T+1]
            x = buf[:-1].view(B, T) #inputs
            y = buf[1:].view(B, T)  #targets

            return x, y

elif corpus == 'fineweb':

    

    def load_tokens(filename):
        npt = np.fromfile(filename, dtype=np.uint16)
        # npt = np.load(filename)
        npt = npt.astype(np.int32) # added after video
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt
             

    class DataLoaderLite:
        def __init__(self, B, T, split, process_rank = 1, num_processes = 1):
            self.B = B
            self.T = T
            self.process_rank = process_rank
            self.num_processes = num_processes
            assert split in {'train', 'val'}

            # get the shard filenames
            data_root = "/workspace/modded-nanogpt/data/finewebedu10B"
            shards = os.listdir(data_root)
            shards = [s for s in shards if split in s]
            shards = sorted(shards)
            shards = [os.path.join(data_root, s) for s in shards]
            self.shards = shards
            assert len(shards) > 0, f"no shards found for split {split}"
            print(f"found {len(shards)} shards for split {split}")
            self.reset()

        def reset(self):
            # state, init at shard zero
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        def next_batch(self):
            B, T = self.B, self.T
            
            buf = self.tokens[self.current_position : self.current_position+B*T+1]
            x = (buf[:-1]).view(B, T) # inputs
            y = (buf[1:]).view(B, T) # targets
            # advance the position in the tensor
            self.current_position += B * T * self.num_processes
            # if loading the next batch would be out of bounds, advance to next shard
            if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = B * T * self.process_rank
            return x, y

else: raise ValueError('Error, corpus can be sherlock, shakespeare, or fineweb')

#-----------------------------------

enc = tiktoken.get_encoding('gpt2')

# create the log directory we will write checkpoints to and log to
log_dir = "log-"+corpus
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

# --------------------- Training ---------------------- #
if train:
    
    ##manually padding the vocab size to be a power of 2 for efficiency
    ##It adds some computation, but when objects are power of 2 are more efficient in cuda
    ##So it's faster overall
    gptconfig = GPTConfig(vocab_size = 50304)
    model = GPT(gptconfig)
    #-------------------- Wandb Setup --------------------#
    wandb.init(
        project="gpt-fineweb",  # change this to your desired project name
        name=f"run-{int(time.time())}",
        config={
            "dataset": corpus,
            "context_length": gptconfig.context_length,
            "vocab_size": gptconfig.vocab_size,
            "n_layer": gptconfig.n_layer,
            "n_head": gptconfig.n_head,
            "n_embd": gptconfig.n_embd,
        }
    )
    print(f"Total number of parameters: {sum(param.numel() for param in model.parameters())}")
    #import sys; sys.exit(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    if compile: model = torch.compile(model)

    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed(42)


    T = gptconfig.context_length
    if corpus != 'fineweb':
        if small_model == True:
            B = 32*1024//T
        else:
            B = 16
    else:
        B  = 8 #64

    if corpus == 'sherlock' or corpus == 'shakespeare':
        train_loader = DataLoaderRandom(B = B, T = T, corpus = corpus, split = 'train')
        val_loader = DataLoaderRandom(B = B, T = T, corpus = corpus, split = 'val')
    else:
        train_loader = DataLoaderLite(B = B, T = T, split="train")
        val_loader = DataLoaderLite(B = B, T = T, split="val")

 
    torch.set_float32_matmul_precision('high') #uses tensorFloat32 for the intermediate computations if the GPU supports it

    #optimizer = torch.optim.AdamW(model.parameters(), weight_decay = 0.1, lr = 6e-4, betas= (0.9, 0.95), eps = 1e-8, fused = True)
    if small_model == True:
        if optimizer_type == 'adam':
            optimizer = torch.optim.AdamW(model.parameters(), weight_decay = 0.1, lr = 1e-3, betas= (0.9, 0.95), eps = 1e-8, fused = True)
        # ECD_q1 
        else:
            optimizer = ECD_q1_scaled(model.parameters(), lr=learning_rate, eta = eta, F0=F0,  nu=nu, weight_decay=0)
    else:
        if optimizer_type == 'adam':
            optimizer = torch.optim.AdamW(model.parameters(), weight_decay = 0.1, lr = 1e-3, betas= (0.9, 0.95), eps = 1e-8, fused = True)
        # ECD_q1 
        else:
            optimizer = ECD_q1_scaled(model.parameters(), lr=learning_rate, eta = eta, F0=F0,  nu=nu, weight_decay=0)
    
    gradient_clipping = False
    if corpus == 'sherlock' or corpus == 'shakespeare':
        steps_all_dataset =  int(1e6/(B*T)) #approximate
    if corpus == 'fineweb':
        steps_all_dataset =  int((1e10/(B*T)))

    max_step = int(fraction_dataset*steps_all_dataset)

    print(f"Full dataset is {steps_all_dataset} steps")
    print(f"Training for {max_step} steps")
    
    
    #I am not implementing learning rate scheduling
    model.train()
    train_losses, val_losses, train_steps, val_steps = [], [], [], [] 
    for step in range(max_step):
        #t0 = time.time()
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e1.record()
        x,y = train_loader.next_batch()
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        ##This is the proper one that uses bfloats16, but needs an Ampere GPU. bfloats are nice because they have the same range as float 32
        if has_bfloat16:
            with torch.autocast(device_type = (torch.device(device)).type, dtype = torch.bfloat16): ##Read the documentation of autocast on putorch
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss.backward()
        
        #This clips the norm of the gradients to be maximum 1 https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
        if gradient_clipping: norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        else: norm = 0.0
        def closure():
            return loss            
        optimizer.step(closure)

        
        with torch.no_grad():
            train_steps.append(step)
            train_losses.append(loss.item())
            wandb.log({"train/loss": loss.item(), "step": step})
            if step%10 == 0:
                #t1 = time.time()
                e2.record()
                torch.cuda.synchronize()
                dt = (e1.elapsed_time(e2)) #time difference in milliseconds
                tokens_per_sec = (train_loader.B*train_loader.T)/ dt
                print(f"Step: {step} , norm: {norm:.4}, Loss: {loss.item():.4} , dt: {dt:.2f} , tok/sec: {tokens_per_sec:.2f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {loss.item():.4}\n")
                plt.plot(train_steps, train_losses, label= "Train loss")
                plt.plot(val_steps, val_losses, label= "Validation loss")
                
                #plt.yscale('log')
                plt.legend()
                plt.savefig('train-loss-124M-ECD-disorder-rotbreak.png', bbox_inches='tight')
                plt.close('all')

            if (step%valid_every == 0) or (step == max_step-1):
                val_steps.append(step)
                model.eval()
                x,y = val_loader.next_batch()
                x,y = x.to(device), y.to(device)

                if has_bfloat16:
                    with torch.autocast((torch.device(device)).type, dtype = torch.bfloat16): ##Read the documentation of autocast on putorch
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)

                val_losses.append(loss.item())
                wandb.log({"val/loss": loss.item(), "step": step})

            if (step%save_every == 0) or (step == max_step-1):

                if step != (max_step-1):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                else:
                    checkpoint_path = os.path.join(log_dir, f"model_final.pt")
    
                checkpoint = {
                    'model': model.state_dict(), 
                    'config': model.config,
                    'step': step, 
                    'train loss': train_losses[-1], 
                    'val loss': val_losses[-1],
                }
                torch.save(checkpoint, checkpoint_path)
                wandb.log({
                    "train/grad_norm": norm,
                    "train/tok_per_sec": tokens_per_sec,
                    "train/loss_10step": loss.item(),
                    "step": step
                })                
                model.train()
    

if inference:
    num_return_sequences = 5
    max_length = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ##Need to load the model
    checkpoint_path = os.path.join(log_dir, f"model_final.pt")
    checkpoint = torch.load(checkpoint_path, weights_only = False)
    model = GPT(checkpoint['config'])
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.to(device)


    tokens = enc.encode("Sherlock Holmes was sitting on his chair when ")
    tokens = torch.tensor(tokens, dtype = torch.long) 
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(num_return_sequences, len(tokens) )
    x = tokens.to(device)

    model.eval()
    while x.size(1) < max_length:
        with torch.no_grad():
            if has_bfloat16:
                with torch.autocast(device_type=(torch.device(device)).type, dtype=torch.bfloat16):
                    logits, loss = model(x) # (B, T, vocab_size)
            else:
                logits, loss = model(x) # (B, T, vocab_size)

            #Take the logits at the last position
            logits = logits[:, -1, :] #(B, vocab_size)
            #Get the probabilities
            probs = F.softmax(logits, dim = -1)
            
            #To avoid weird behavior, only keep the topk (k = 50 ) probabilities before sampling from them
            topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)

            #now sample a token according to these probabilities
            ix = torch.multinomial(topk_probs, 1) #(B, 1)

            #gather the corresponding indices
            x_col = torch.gather(topk_indices, -1, ix) #(B, 1)

            #append to the sequence
            x = torch.cat((x, x_col), dim = -1)

    #print the generated text
    decoded = ""
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded+= enc.decode(tokens)
        decoded+="\n\n"
    print(decoded)
    log_inference_file = os.path.join(log_dir, f"inference.txt")
    with open(log_inference_file, "w") as f:
        f.write(decoded)

print(f"Total run time in seconds: {(time.time()-initial_time)}")  

# after training completes:
print("Top‐level modules:", list(model.named_children()))
# you should see something like [('transformer', ...), ('lm_head', ...)]

# grab the list of blocks:
layers = model.transformer.h   # this is an nn.ModuleList of your Blocks

# # now iterate and pull out the AsymmetricCausalSelfAttention params
# for i, blk in enumerate(layers):
#     temp = blk.sa.head_temp.detach().cpu()   # note: 'sa', not 'attn'
#     scale = blk.sa.head_scale.detach().cpu()
#     print(f"Layer {i:2d}:  head_temp  std = {temp.std().item():.4f},  head_scale std = {scale.std().item():.4f}")


# def analyze_param(param, name):
#     """
#     param: a torch.Tensor of shape (n_head, head_dim)
#     name:  string label
#     """
#     arr = param.detach().cpu().numpy()
#     # overall stats across all heads & dims
#     flat = arr.ravel()
#     mu    = flat.mean()
#     sigma = flat.std()
#     rng   = flat.max() - flat.min()
#     cv    = sigma/abs(mu) if abs(mu)>0 else np.nan
#     print(f"  {name:20s} overall → mean {mu:.4f}, std {sigma:.4f}, range {rng:.4f}, cv {cv:.4f}")

#     # now look at variation *across heads* in the head‐means
#     head_means = arr.mean(axis=1)   # length n_head
#     hm_mu  = head_means.mean()
#     hm_std = head_means.std()
#     print(f"  {name:20s} head‐means → mean {hm_mu:.4f}, std {hm_std:.4f}\n")

# --- after training completes ---
# print("=== Rotation‐breaking stats per layer ===")
# for i, blk in enumerate(model.transformer.h):
#     print(f"Layer {i}:")
#     analyze_param(blk.sa.q_scale, "q_scale")
#     analyze_param(blk.sa.q_bias,  "q_bias")
#     analyze_param(blk.sa.k_scale, "k_scale")
#     analyze_param(blk.sa.k_bias,  "k_bias")

# print("=== Rotation-breaking / disorder stats per layer ===")
# for i, blk in enumerate(model.transformer.h):
#     sa = blk.sa
#     if hasattr(sa, "q_scale"):  # your fixed-asymmetry variant
#         def analyze_param(p, name):
#             arr = p.detach().float().cpu().view(-1).numpy()
#             print(f"  {name:12s} mean {arr.mean():.4f}  std {arr.std():.4f}  rng {(arr.max()-arr.min()):.4f}")
#         print(f"Layer {i}:")
#         analyze_param(sa.q_scale, "q_scale")
#         analyze_param(sa.q_bias,  "q_bias")
#         analyze_param(sa.k_scale, "k_scale")
#         analyze_param(sa.k_bias,  "k_bias")
#     elif hasattr(sa, "bQ"):     # disordered attention (this class)
#         print(f"Layer {i}: bQ mean {sa.bQ.float().mean().item():.4f} std {sa.bQ.float().std().item():.4f} | "
#               f"bK mean {sa.bK.float().mean().item():.4f} std {sa.bK.float().std().item():.4f}")





    
