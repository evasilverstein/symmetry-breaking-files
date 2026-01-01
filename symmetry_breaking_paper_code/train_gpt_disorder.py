# train_scale_disorder_ecd.py
# Token-budgeted trainer with Eq.(28) instrumentation, AdamW/SOAP LR calibration,
# W&B optimizer/HParam logging, robust loader, and optional K-bias disable.

from dataclasses import dataclass
import os, math, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


# --- optional logger ---
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# --- your optimizer (exact semantics preserved) ---
try:
    from optimizers.ECD_q1_scaled import ECD_q1_scaled
except ImportError:
    from ECD_q1_scaled import ECD_q1_scaled  # Fallback for direct execution

# -------------------------
# W&B optimizer serialization helpers
# -------------------------
def _to_serializable(x):
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    try:
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    try:
        if torch.is_tensor(x):
            return x.item() if x.numel() == 1 else f"tensor(shape={tuple(x.shape)}, dtype={x.dtype})"
    except Exception:
        pass
    return str(x)

def serialize_optimizer(opt):
    info = {
        "class": opt.__class__.__name__,
        "defaults": {},
        "param_groups": []
    }
    d = getattr(opt, "defaults", {})
    info["defaults"] = {k: _to_serializable(v) for k, v in d.items()}
    for g in getattr(opt, "param_groups", []):
        g_copy = {k: _to_serializable(v) for k, v in g.items() if k != "params"}
        info["param_groups"].append(g_copy)
    return info

def wandb_record_optimizer(run, opt_kind, opt_obj, ecd_kwargs=None, lr_calibrated=None):
    if run is None:
        return
    info = serialize_optimizer(opt_obj)
    payload = {
        "optimizer": {
            "kind": opt_kind,
            "class": info["class"],
            "defaults": info["defaults"],
            "param_groups": info["param_groups"],
        }
    }
    if lr_calibrated is not None:
        payload["optimizer"]["calibrated_lr"] = lr_calibrated
    run.config.update(payload, allow_val_change=True)

    # For ECD: exact names/defs as in your constructor
    if opt_kind == "ecd" and ecd_kwargs is not None:
        run.config.update({
            "ecd": {
                "lr": ecd_kwargs.get("lr"),
                "F0": ecd_kwargs.get("F0"),
                "eps1": ecd_kwargs.get("eps1"),
                "eps2": ecd_kwargs.get("eps2"),
                "nu": ecd_kwargs.get("nu"),
                "weight_decay": ecd_kwargs.get("weight_decay"),
                "eta": ecd_kwargs.get("eta"),
                "consEn": ecd_kwargs.get("consEn"),
            }
        }, allow_val_change=True)

# -------------------------
# misc utils
# -------------------------
def maybe_graph_break():
    try:
        import torch._dynamo as _dynamo
        _dynamo.graph_break()
    except Exception:
        pass

def device_type():
    return "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Model config presets
# -------------------------
@dataclass
class GPTConfig:
    context_length: int = 512
    vocab_size:     int = 50304
    n_layer:        int = 12
    n_head:         int = 12
    n_embd:         int = 768
    rotary_base:    int = 10000

PRESETS = {
    "124m": GPTConfig(n_layer=12, n_head=12, n_embd=768),
    "355m": GPTConfig(n_layer=24, n_head=16, n_embd=1024),
    "774m": GPTConfig(n_layer=36, n_head=20, n_embd=1280),
    "1.3b": GPTConfig(n_layer=48, n_head=25, n_embd=1600),
}

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors. This version
    correctly handles cases where the rotary dimension is smaller than the head dimension.
    """
    # The rotary dimension is determined by the size of the cos/sin tensors.
    rotary_dim = cos.shape[-1]
    
    # Unsqueeze cos and sin for broadcasting
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Slice the query and key into the part to be rotated and the part to be passed through.
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply RoPE to the first part
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate the rotated and passed-through parts back together
    q_embed = torch.cat((q_rot, q_pass), dim=-1)
    k_embed = torch.cat((k_rot, k_pass), dim=-1)
    
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, config:GPTConfig, device=None):
        super().__init__()
        self.config = config
        
        # Get the full head dimension
        head_dim = config.n_embd // config.n_head
        
        # Calculate the rotary dimension based on rope_pct
        # This is the dimension that RoPE will be applied to.
        rotary_dim = int(head_dim * 0.5) #config.rope_pct=0.5
        
        # The base for the inverse frequency calculation is rotary_dim
        # This mirrors the Megatron `dim = int(dim * rotary_percent)` logic
        inv_freq = 1.0 / (config.rotary_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Assuming no advanced RoPE scaling for now
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids): #the only information that x provides is device and dtype
        # This forward pass now correctly uses the smaller `inv_freq`
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # freqs now has shape [batch, seq_len, rotary_dim / 2]
            
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            # cos and sin now have shape [batch, seq_len, rotary_dim]

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)



# -------------------------
# Disordered attention & asymmetric MLP (matches your training file)
# -------------------------
class DisorderedCausalSelfAttention(nn.Module):
    def __init__(self, config, mean_Q=None, mean_K=None, std_Q=None, std_K=None,
                 mode="per_batch", use_k_bias=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d = config.n_embd // config.n_head
        C = config.n_embd
        self.c_attn = nn.Linear(C, 3*C)
        self.c_proj = nn.Linear(C, C); self.c_proj.NANOGPT_SCALE_INIT = 1

        def to_vec(x, default):
            if x is None:
                x = default
            x = torch.as_tensor(x, dtype=torch.float32)
            if x.ndim == 0:
                x = x.repeat(self.d)
            return x.view(self.d)

        mean_Q = to_vec(mean_Q, 0.5)
        mean_K = to_vec(mean_K, 0.3)
        std_Q  = to_vec(std_Q,  torch.linspace(0.05, 0.15, self.d))
        std_K  = to_vec(std_K,  torch.linspace(0.12, 0.08,  self.d))
        self.register_buffer("mean_Q", mean_Q)
        self.register_buffer("mean_K", mean_K)
        self.register_buffer("std_Q",  std_Q.abs())
        self.register_buffer("std_K",  std_K.abs())
        self.register_buffer("bQ", torch.zeros(self.n_head, self.d))
        self.register_buffer("bK", torch.zeros(self.n_head, self.d))
        self.mode = mode
        self.use_k_bias = use_k_bias

        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        #self.register_buffer("causal_mask", mask.view(1,1,config.context_length,config.context_length))

    @torch.no_grad()
    def _resample_biases(self, device):
        mQ, sQ = self.mean_Q.to(device), self.std_Q.to(device)
        mK, sK = self.mean_K.to(device), self.std_K.to(device)
        base_Q = mQ + sQ * torch.randn_like(mQ)
        base_K = mK + sK * torch.randn_like(mK)
        self.bQ.copy_(base_Q.unsqueeze(0).expand(self.n_head, -1).contiguous())
        self.bK.copy_(base_K.unsqueeze(0).expand(self.n_head, -1).contiguous())

    def forward(self, x, position_embeddings):
        B, T, C = x.shape
        nh, d = self.n_head, self.d

        if self.training and self.mode == "per_batch":
            maybe_graph_break()
            self._resample_biases(x.device)
        
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  

        qkv = self.c_attn(x); q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, nh, d).transpose(1,2)
        k = k.view(B, T, nh, d).transpose(1,2)
        v = v.view(B, T, nh, d).transpose(1,2)

        q = q + self.bQ.view(1, nh, 1, d)
        if self.use_k_bias:
            k = k + self.bK.view(1, nh, 1, d)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # scores = (q @ k.transpose(-2,-1)) / math.sqrt(d)
        # scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(scores, dim=-1)
        # y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.c_proj(y)

class RandomPReLU1d(nn.Module):
    def __init__(self, features, init_slope=0.2, slope_std=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features) * slope_std + init_slope)
    def forward(self, x):
        x2 = x.transpose(1, 2)
        y2 = F.prelu(x2, self.weight)
        return y2.transpose(1, 2)

class AsymmetricMLPPreLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden)
        self.act  = RandomPReLU1d(hidden, init_slope=0.2, slope_std=1.0)
        self.c_proj = nn.Linear(hidden, config.n_embd); self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config, use_k_bias=True):
        super().__init__()
        self.sa  = DisorderedCausalSelfAttention(config, mode="per_batch", use_k_bias=use_k_bias)
        self.mlp = AsymmetricMLPPreLU(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
    def forward(self, x, position_embedding):
        x = x + self.sa(self.ln1(x), position_embedding)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig, use_k_bias=True):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            #wpe  = nn.Embedding(config.context_length, config.n_embd),
            h    = nn.ModuleList([Block(config, use_k_bias=use_k_bias) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        self.p_emb = RotaryEmbedding(config=self.config)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** (-0.5)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.context_length
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        

        x = self.transformer.wte(idx) #+ self.transformer.wpe(pos)
        pos_emb = self.p_emb(x, pos)
        
        for block in self.transformer.h:
            x = block(x, pos_emb)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -------------------------
# Robust FineWeb loader
# -------------------------
def load_tokens_autodetect(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".bin":
        arr = np.memmap(filename, dtype=np.uint16, mode="r")
        return torch.tensor(arr.astype(np.int64), dtype=torch.long)
    if ext == ".npy":
        arr = np.load(filename)
        arr = arr.astype(np.int64)
        return torch.tensor(arr, dtype=torch.long)
    if ext in (".pt", ".pth"):
        t = torch.load(filename)
        return t.long() if torch.is_tensor(t) else torch.tensor(np.asarray(t).astype(np.int64), dtype=torch.long)
    raise RuntimeError(f"Unknown shard extension: {ext}")

def load_tokens_with_hint(filename, data_ext):
    if data_ext == "auto":
        return load_tokens_autodetect(filename)
    if data_ext == "bin":
        arr = np.memmap(filename, dtype=np.uint16, mode="r")
        return torch.tensor(arr.astype(np.int64), dtype=torch.long)
    if data_ext == "npy":
        arr = np.load(filename)
        return torch.tensor(arr.astype(np.int64), dtype=torch.long)
    if data_ext == "pt":
        t = torch.load(filename)
        return t.long() if torch.is_tensor(t) else torch.tensor(np.asarray(t).astype(np.int64), dtype=torch.long)
    raise RuntimeError(f"Bad --data_ext: {data_ext}")

class FineWebShards:
    def __init__(self, data_root, split, B, T, process_rank=0, num_processes=1,
                 data_ext="auto", vocab_size=None, sanity_check=False):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.data_ext = data_ext
        self.vocab_size = vocab_size
        self.sanity_check = sanity_check
        shards = sorted([os.path.join(data_root, s)
                         for s in os.listdir(data_root)
                         if (split in s and (data_ext=="auto" or s.endswith("."+data_ext)))])
        if data_ext == "auto":
            shards = [s for s in shards if os.path.splitext(s)[1].lower() in (".bin",".npy",".pt",".pth")]
        if not shards:
            raise FileNotFoundError(f"No shards for split={split} under {data_root} (ext={data_ext})")
        self.shards = shards
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens_with_hint(self.shards[self.current_shard], self.data_ext)
        if self.sanity_check and self.vocab_size is not None:
            mx = int(self.tokens.max().item())
            if mx >= self.vocab_size:
                raise ValueError(f"[sanity] shard {self.shards[self.current_shard]} has max token {mx} "
                                 f">= vocab_size {self.vocab_size}. Wrong file format or vocab.")
        self.pos = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.pos : self.pos + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.pos += B * T * self.num_processes
        if self.pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens_with_hint(self.shards[self.current_shard], self.data_ext)
            if self.sanity_check and self.vocab_size is not None:
                mx = int(self.tokens.max().item())
                if mx >= self.vocab_size:
                    raise ValueError(f"[sanity] shard {self.shards[self.current_shard]} has max token {mx} "
                                     f">= vocab_size {self.vocab_size}. Wrong file format or vocab.")
            self.pos = B * T * self.process_rank
        return x, y

# -------------------------
# Optimizers
# -------------------------
def iter_named_params(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            yield n, p

def make_muon_param_groups(model):
    hidden_weights, other = [], []
    EXCLUDE = ("transformer.wte", "transformer.wpe", "lm_head")
    for name, p in iter_named_params(model):
        if any(ex in name for ex in EXCLUDE):
            other.append(p); continue
        (hidden_weights if p.ndim >= 2 else other).append(p)
    return [
        dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=0.01),
        dict(params=other,        use_muon=False, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
    ]

def build_optimizer(model, kind, ecd_kwargs=None, soap_kwargs=None, adamw_kwargs=None):
    kind = kind.lower()
    if kind == "ecd":
        return ECD_q1_scaled(model.parameters(), **(ecd_kwargs or {}))
    if kind == "adam":
        kw = dict(weight_decay=0.1, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, fused=True)
        kw.update(adamw_kwargs or {})
        return torch.optim.AdamW(model.parameters(), **kw)
    if kind == "soap":
        try:
            import pytorch_optimizer as pyo
        except Exception as e:
            raise RuntimeError("Install pytorch-optimizer for SOAP") from e
        kw = dict(lr=3e-4, betas=(0.95, 0.95), weight_decay=0.01, precondition_frequency=10)
        kw.update(soap_kwargs or {})
        return pyo.SOAP(model.parameters(), **kw)
    if kind == "muon":
        try:
            from muon import MuonWithAuxAdam
        except Exception as e:
            raise RuntimeError("Install Muon for muon optimizer") from e
        return MuonWithAuxAdam(make_muon_param_groups(model))
    raise ValueError(f"Unknown optimizer {kind}")

# -------------------------
# Utilities: parameter vector + calibration
# -------------------------
def param_vector(model):
    return torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])

@torch.no_grad()
def theta0_norm(model):
    s = 0.0
    for p in model.parameters():
        s += float(torch.sum(p.data.float()**2))
    return math.sqrt(s)

def median(lst):
    a = sorted(lst)
    n = len(a)
    return 0.5*(a[n//2] + a[(n-1)//2]) if n else 0.0

def run_updates_for_speed(cfg, data_root, B, T, gdup, base_model, opt_kind, opt_kwargs,
                          updates, dtype, device):
    """Clone model, run 'updates' optimizer steps, return per-step ||Δθ|| list."""
    m = GPT(cfg, use_k_bias=True).to(device)
    m.load_state_dict(base_model.state_dict())
    loader = FineWebShards(data_root, "train", B=B, T=T)
    optimizer = build_optimizer(m, opt_kind, ecd_kwargs=opt_kwargs if opt_kind=="ecd" else None,
                                soap_kwargs=opt_kwargs if opt_kind=="soap" else None,
                                adamw_kwargs=opt_kwargs if opt_kind=="adam" else None)
    deltas = []
    prev = param_vector(m).detach().clone()

    m.train()
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        tot_loss = 0.0
        for _ in range(gdup):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type(), enabled=(dtype==torch.bfloat16), dtype=dtype):
                _, loss = m(x, y)
            (loss / gdup).backward()
            tot_loss += float(loss.detach().item())

        avg_loss = torch.tensor(tot_loss / gdup, device=device)

        def closure():
            return avg_loss

        try:
            optimizer.step(closure)
        except TypeError:
            optimizer.step()

        cur = param_vector(m).detach()
        deltas.append(float(torch.norm(cur - prev)))
        prev = cur

    return deltas

def calibrate_lr_against_ecd(cfg, data_root, B, T, gdup, base_model, dtype, device,
                             ecd_kwargs, target_updates=50, max_iters=5,
                             kind="adam", init_lr=1e-3, other_kwargs=None):
    """Find lr for AdamW/SOAP that matches ECD median ||Δθ|| over target_updates."""
    other_kwargs = dict(other_kwargs or {})
    ecd_dtheta = run_updates_for_speed(cfg, data_root, B, T, gdup, base_model, "ecd", ecd_kwargs,
                                       updates=target_updates, dtype=dtype, device=device)
    target = median(ecd_dtheta[len(ecd_dtheta)//5:])  # ignore first ~20%

    lr = init_lr
    for _ in range(max_iters):
        kw = other_kwargs.copy()
        kw["lr"] = lr
        dtheta = run_updates_for_speed(cfg, data_root, B, T, gdup, base_model, kind, kw,
                                       updates=max(10, target_updates//2), dtype=dtype, device=device)
        cur = median(dtheta[len(dtheta)//5:])
        if cur == 0 or target == 0:
            break
        ratio = cur / target
        if 0.95 <= ratio <= 1.05:
            break
        lr = lr / (ratio**0.5)
    return lr, target

# -------------------------
# Bias-invariance check (optional)
# -------------------------
@torch.no_grad()
def check_bias_invariance(nh, d, device):
    B, T = 2, 8
    q = torch.randn(B, nh, T, d, device=device)
    k = torch.randn(B, nh, T, d, device=device)
    bQ = torch.randn(nh, d, device=device)
    bK = torch.randn(nh, d, device=device)
    def attn(q, k, bQ, bK):
        q1 = q + bQ.view(1, nh, 1, d)
        k1 = k + bK.view(1, nh, 1, d)
        s = (q1 @ k1.transpose(-2,-1)) / math.sqrt(d)
        return torch.softmax(s, dim=-1)
    A_full = attn(q, k, bQ, bK)
    A_noK  = attn(q, k, bQ, torch.zeros_like(bK))
    return float((A_full - A_noK).abs().max().item())

# -------------------------
# Train
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=PRESETS.keys(), default="124m")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=50304)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--train_tokens", type=float, default=5e8)
    ap.add_argument("--valid_every_updates", type=int, default=1000)
    ap.add_argument("--optimizer", choices=["ecd","adam","soap","muon"], default="ecd")
    # ECD hypers: exact names/defs as in your ECD_q1_scaled constructor
    ap.add_argument("--ecd_lr", type=float, default=0.5)
    ap.add_argument("--ecd_eta", type=float, default=100.0)
    ap.add_argument("--ecd_F0", type=float, default=2.0)
    ap.add_argument("--ecd_eps1", type=float, default=1e-10)
    ap.add_argument("--ecd_eps2", type=float, default=1e-40)
    ap.add_argument("--ecd_nu", type=float, default=0.0)
    ap.add_argument("--ecd_wd", type=float, default=0.0)
    ap.add_argument("--ecd_consEn", action="store_true", help="Enable energy conservation (default True)")
    ap.add_argument("--no_ecd_consEn", dest="ecd_consEn", action="store_false", help="Disable energy conservation")
    ap.set_defaults(ecd_consEn=True)
    # argparse additions
    ap.add_argument("--soap_beta1", type=float, default=0.95)
    ap.add_argument("--soap_beta2", type=float, default=0.95)
    ap.add_argument("--soap_wd", type=float, default=1e-4)
    ap.add_argument("--soap_prefreq", type=int, default=10)



    # Data loader options
    ap.add_argument("--data_ext", choices=["auto","bin","npy","pt"], default="auto",
                    help="Force shard extension or auto-detect")
    ap.add_argument("--sanity_check_tokens", action="store_true",
                    help="Check max token id per shard vs. --vocab before training")

    # Eq.(28) helpers
    ap.add_argument("--hat_dt", type=float, default=0.1)
    ap.add_argument("--F2", type=float, default=1.0)
    ap.add_argument("--use_eq28_eta", action="store_true",
                    help="Override --ecd_eta with Eq.(28) suggested value")

    # LR calibration (AdamW/SOAP)
    ap.add_argument("--calibrate_lr", action="store_true",
                    help="For adam/soap: match per-step ||Δθ|| to ECD over a short burn-in")
    ap.add_argument("--calib_updates", type=int, default=50)
    ap.add_argument("--calib_max_iters", type=int, default=5)
    ap.add_argument("--adam_lr", type=float, default=1e-3)
    ap.add_argument("--soap_lr", type=float, default=3e-4)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default="/workspace/modded-nanogpt/data/finewebedu10B")
    ap.add_argument("--log_dir", type=str, default="runs")
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--save_every_updates", type=int, default=0)
    ap.add_argument("--use_bf16", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    # Bias controls
    ap.add_argument("--disable_k_bias", action="store_true", help="Use only bQ; set bK=0")
    ap.add_argument("--check_bias_invariance", action="store_true")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    cfg = PRESETS[args.model]
    cfg.context_length = args.ctx
    cfg.vocab_size = args.vocab
    device = device_type()
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available()) else torch.float32
    torch.set_float32_matmul_precision("high")

    model = GPT(cfg, use_k_bias=(not args.disable_k_bias)).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Data
    T = cfg.context_length
    B = args.batch_size
    gdup = args.grad_accum
    train_loader = FineWebShards(args.data_root, "train", B=B, T=T,
                                 data_ext=args.data_ext, vocab_size=cfg.vocab_size,
                                 sanity_check=args.sanity_check_tokens)
    val_loader   = FineWebShards(args.data_root, "val",   B=B, T=T,
                                 data_ext=args.data_ext, vocab_size=cfg.vocab_size,
                                 sanity_check=args.sanity_check_tokens)

    # Eq.(28) diagnostics
    theta0 = theta0_norm(model)
    steps_target = args.train_tokens / float(B*T*gdup)
    eta_suggested = ((steps_target * args.hat_dt) / max(theta0, 1e-12))**2 / max(args.F2, 1e-12)

    # W&B setup (name/tags reflect optimizer and calibration)
    run_name = args.name or f"{args.model}-ctx{args.ctx}-{args.optimizer}{'-cal' if (args.calibrate_lr and args.optimizer in ('adam','soap')) else ''}"
    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.log_dir, run_name); os.makedirs(ckpt_dir, exist_ok=True)

    if args.wandb and WANDB_AVAILABLE:
        tags = [args.optimizer, args.model, f"ctx{args.ctx}"]
        if args.calibrate_lr and args.optimizer in ("adam","soap"):
            tags.append("calibrated")
        wandb.init(
            project="ecd-scale",
            name=run_name,
            tags=tags,
            config={
                "dataset": "finewebedu10B",
                "model": vars(cfg),
                "params": n_params,
                "context_length": cfg.context_length,
                "batch_size": B,
                "grad_accum": gdup,
                "tokens_per_update": B*T*gdup,
                "train_tokens": args.train_tokens,
                "opt_kind": args.optimizer,
                "eq28": {
                    "theta0_norm": theta0,
                    "n_steps_target": steps_target,
                    "hat_dt": args.hat_dt,
                    "F2": args.F2,
                    "eta_suggested": eta_suggested
                },
            }
        )

    # Optionally override eta from Eq.(28)
    if args.optimizer == "ecd" and args.use_eq28_eta:
        print(f"[eq28] overriding --ecd_eta {args.ecd_eta} → {eta_suggested:.6g}")
        args.ecd_eta = float(eta_suggested)

    # Optional bias invariance check
    if args.check_bias_invariance:
        diff = check_bias_invariance(cfg.n_head, cfg.n_embd // cfg.n_head, device)
        print(f"[bias-check] max|A_full - A_noK| = {diff:.3e} (expected ~1e-7 to 1e-6)")
        if args.wandb and WANDB_AVAILABLE:
            wandb.log({"bias_invariance/max_abs_diff": diff, "step": 0})

    # Build optimizer (and optional calibration)
    if args.optimizer == "ecd":
        ecd_kwargs = dict(lr=args.ecd_lr, F0=args.ecd_F0, eps1=args.ecd_eps1,
                          eps2=args.ecd_eps2, nu=args.ecd_nu, weight_decay=args.ecd_wd,
                          eta=args.ecd_eta, consEn=args.ecd_consEn)
        opt = build_optimizer(model, "ecd", ecd_kwargs=ecd_kwargs)
    elif args.optimizer in ("adam", "soap"):
        lr0 = args.adam_lr if args.optimizer=="adam" else args.soap_lr
        if args.calibrate_lr:
            ecd_kwargs_cal = dict(lr=args.ecd_lr, F0=args.ecd_F0, eps1=args.ecd_eps1,
                                  eps2=args.ecd_eps2, nu=args.ecd_nu, weight_decay=args.ecd_wd,
                                  eta=args.ecd_eta, consEn=args.ecd_consEn)
            lr_cal, target = calibrate_lr_against_ecd(
                cfg, args.data_root, B, T, gdup, model, dtype, device,
                ecd_kwargs_cal, target_updates=args.calib_updates, max_iters=args.calib_max_iters,
                kind=args.optimizer, init_lr=lr0,
                other_kwargs=(dict(betas=(0.9,0.95), weight_decay=0.01) if args.optimizer=="adam"
                              else dict(betas=(0.95,0.95), weight_decay=0.01, precondition_frequency=10))
            )
            print(f"[calib] {args.optimizer} lr {lr0:g} → {lr_cal:g} to match median ||Δθ||≈{target:g}")
            lr0 = lr_cal
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({f"calib/{args.optimizer}_lr": lr0, "calib/target_dtheta_median": target, "step": 0})
        if args.optimizer == "adam":
            opt = build_optimizer(model, "adam", adamw_kwargs=dict(lr=lr0, betas=(0.9,0.95), weight_decay=0.01, eps=1e-8, fused=True))
        else:

            # when building SOAP:
#- opt = build_optimizer(model, "soap", soap_kwargs=dict(lr=lr0, betas=(0.95,0.95), weight_decay=0.01, precondition_frequency=10))
            opt = build_optimizer(model, "soap", soap_kwargs=dict(
                 lr=lr0,
                 betas=(args.soap_beta1, args.soap_beta2),
                 weight_decay=args.soap_wd,
                 precondition_frequency=args.soap_prefreq
             ))
            # opt = build_optimizer(model, "soap", soap_kwargs=dict(lr=lr0, betas=(0.95,0.95), weight_decay=0.01, precondition_frequency=10))
    elif args.optimizer == "muon":
        opt = build_optimizer(model, "muon")
    else:
        raise ValueError(f"bad optimizer: {args.optimizer}")

    # Record optimizer & HPs in W&B
    if args.wandb and WANDB_AVAILABLE:
        if args.optimizer == "ecd":
            wandb_record_optimizer(wandb.run, "ecd", opt, ecd_kwargs=ecd_kwargs)
            wandb.run.config.update({"ecd_derived": {
                "lr_over_sqrt_eta": args.ecd_lr / math.sqrt(args.ecd_eta),
                "nu_over_sqrt_dim": (args.ecd_nu / math.sqrt(n_params)) if n_params>0 else 0.0
            }}, allow_val_change=True)
        elif args.optimizer in ("adam", "soap"):
            lr_used = args.adam_lr if args.optimizer=="adam" else args.soap_lr
            if args.calibrate_lr:
                lr_used = lr0
            wandb_record_optimizer(wandb.run, args.optimizer, opt, lr_calibrated=lr_used)
        else:
            wandb_record_optimizer(wandb.run, args.optimizer, opt)

    # -------------------------
    # Training loop (token-budgeted)
    # -------------------------
    consumed_tokens = 0
    update = 0
    model.train()
    start = time.time()

    while consumed_tokens < args.train_tokens:
        t0 = time.time()
        total_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for _ in range(gdup):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type(), enabled=(dtype==torch.bfloat16), dtype=dtype):
                _, loss = model(x, y)
            (loss / gdup).backward()
            total_loss += float(loss.detach().item())
            consumed_tokens += (B * T)

        avg_loss = torch.tensor(total_loss / gdup, device=device)
        def closure():
            return avg_loss

        try:
            opt.step(closure)
        except TypeError:
            opt.step()

        update += 1
        dt = time.time() - t0
        tok_per_s = (B * T * gdup) / max(dt, 1e-9)

        if args.wandb and WANDB_AVAILABLE:
            wandb.log({"train/loss": avg_loss.item(),
                       "speed/tok_per_s": tok_per_s,
                       "train/consumed_tokens": consumed_tokens,
                       "update": update})

        if update % max(1, args.valid_every_updates) == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = val_loader.next_batch()
                vx, vy = vx.to(device), vy.to(device)
                with torch.autocast(device_type(), enabled=(dtype==torch.bfloat16), dtype=dtype):
                    _, vloss = model(vx, vy)
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({"val/loss": vloss.item(), "update": update})
            print(f"[up {update:6d}] loss {avg_loss.item():.4f} | val {vloss.item():.4f} "
                  f"| tok/s {tok_per_s:,.0f} | consumed {int(consumed_tokens):,}")
            model.train()

        if args.save_every_updates and (update % args.save_every_updates == 0):
            path = os.path.join(ckpt_dir, f"model_{update:06d}.pt")
            torch.save({"model": model.state_dict(), "config": cfg, "update": update,
                        "consumed_tokens": consumed_tokens}, path)

    # final save
    final_path = os.path.join(ckpt_dir, "model_final.pt")
    torch.save({"model": model.state_dict(), "config": cfg, "update": update,
                "consumed_tokens": consumed_tokens}, final_path)
    print(f"Done. Saved to {final_path}. Total time: {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
