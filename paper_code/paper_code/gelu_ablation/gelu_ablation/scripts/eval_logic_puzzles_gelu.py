#!/usr/bin/env python3
"""
Logic Puzzle Evaluation for GELU Ablation Models

IMPORTANT: This script is SPECIFICALLY for GELU ablation models.
DO NOT USE for PReLU models - use the PReLU evaluation scripts instead.

This script uses GELU MLP (as in standard GPT-2) to match the training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import math
from pathlib import Path
from dataclasses import dataclass
import numpy as np

import tiktoken


@dataclass
class GPTConfig:
    context_length: int = 512
    vocab_size:     int = 50304
    n_layer:        int = 12
    n_head:         int = 12
    n_embd:         int = 768
    rotary_base:    int = 10000
    rope_pct:       float = 0.5


class CausalSelfAttentionWithBias(nn.Module):
    """Attention with optional bQ/bK/bV biases applied at inference."""

    def __init__(self, config, bQ=None, bK=None, bV=None):
        super().__init__()
        self.n_head = config.n_head
        self.d = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

        # Handle bias shapes
        if bQ is not None:
            if bQ.dim() == 1:
                bQ = bQ.view(1, 1, 1, self.d).expand(1, self.n_head, 1, self.d).clone()
            elif bQ.dim() == 2:
                bQ = bQ.view(1, self.n_head, 1, self.d)
            self.register_buffer("bQ", bQ)
        else:
            self.register_buffer("bQ", torch.zeros(1, self.n_head, 1, self.d))

        if bK is not None:
            if bK.dim() == 1:
                bK = bK.view(1, 1, 1, self.d).expand(1, self.n_head, 1, self.d).clone()
            elif bK.dim() == 2:
                bK = bK.view(1, self.n_head, 1, self.d)
            self.register_buffer("bK", bK)
        else:
            self.register_buffer("bK", torch.zeros(1, self.n_head, 1, self.d))

        if bV is not None:
            if bV.dim() == 1:
                bV = bV.view(1, 1, 1, self.d).expand(1, self.n_head, 1, self.d).clone()
            elif bV.dim() == 2:
                bV = bV.view(1, self.n_head, 1, self.d)
            self.register_buffer("bV", bV)
        else:
            self.register_buffer("bV", torch.zeros(1, self.n_head, 1, self.d))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d).transpose(1, 2)

        # Apply biases
        q = q + self.bQ
        k = k + self.bK
        v = v + self.bV

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(scores, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class GELUMLP(nn.Module):
    """Standard GELU MLP as in GPT-2.

    THIS IS THE KEY DIFFERENCE from PReLU models.
    GELU ablation models use this, NOT AsymmetricMLPPreLU.
    """
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden)
        self.c_proj = nn.Linear(hidden, config.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config, bQ=None, bK=None, bV=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionWithBias(config, bQ=bQ, bK=bK, bV=bV)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GELUMLP(config)  # GELU, not PReLU!

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, layer_biases=None):
        super().__init__()
        self.config = config

        if layer_biases is None:
            layer_biases = {}

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.context_length, config.n_embd),
            h=nn.ModuleList([
                Block(config,
                      bQ=layer_biases.get(i, (None, None, None))[0],
                      bK=layer_biases.get(i, (None, None, None))[1],
                      bV=layer_biases.get(i, (None, None, None))[2])
                for i in range(config.n_layer)
            ]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


def extract_mean_biases(state_dict, n_layer, d_head, n_head,
                        default_mean_Q=0.5, default_mean_V=0.5):
    """Extract mean bQ, bK, bV values from checkpoint."""
    mean_Q = None
    mean_V = None

    for k, v in state_dict.items():
        if 'mean_Q' in k and mean_Q is None:
            mean_Q = v.clone()
        if 'mean_V' in k and mean_V is None:
            mean_V = v.clone()

    if mean_Q is None:
        mean_Q = torch.full((d_head,), default_mean_Q)
    if mean_V is None:
        mean_V = torch.full((d_head,), default_mean_V)

    layer_biases = {}
    for layer_idx in range(n_layer):
        layer_biases[layer_idx] = (mean_Q.clone(), None, mean_V.clone())

    return layer_biases, mean_Q, mean_V


def load_gelu_model(ckpt_path, device='cuda', use_biases=False,
                    override_mean_Q=None, override_mean_V=None):
    """Load a GELU ablation model checkpoint."""

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if 'config' in ckpt:
        cfg = ckpt['config']
        config = GPTConfig(
            context_length=getattr(cfg, 'context_length', 512),
            vocab_size=getattr(cfg, 'vocab_size', 50304),
            n_layer=getattr(cfg, 'n_layer', 12),
            n_head=getattr(cfg, 'n_head', 12),
            n_embd=getattr(cfg, 'n_embd', 768),
        )
    else:
        config = GPTConfig()

    d_head = config.n_embd // config.n_head
    print(f"  Config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    print(f"  MLP type: GELU (standard GPT-2)")

    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Determine if biases should be applied
    layer_biases = {}
    if use_biases:
        default_mean_Q = override_mean_Q if override_mean_Q is not None else 0.5
        default_mean_V = override_mean_V if override_mean_V is not None else 0.5

        layer_biases, mean_Q, mean_V = extract_mean_biases(
            state_dict, config.n_layer, d_head, config.n_head,
            default_mean_Q=default_mean_Q, default_mean_V=default_mean_V
        )
        print(f"  Biases: bQ mean={mean_Q.mean().item():.4f}, bV mean={mean_V.mean().item():.4f}")
    else:
        print(f"  Biases: None (symmetric mode)")

    # Build model
    model = GPT(config, layer_biases=layer_biases)

    # Clean state dict
    cleaned = {}
    for k, v in state_dict.items():
        k_clean = k.replace('.sa.', '.attn.')
        k_clean = k_clean.replace('.ln1.', '.ln_1.')
        k_clean = k_clean.replace('.ln2.', '.ln_2.')

        # Skip bias-related keys
        if any(x in k_clean for x in ['.bQ', '.bK', '.bV', '.mean_Q', '.mean_K', '.mean_V', '.std_Q', '.std_K', '.std_V']):
            continue
        cleaned[k_clean] = v

    # Load weights
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    real_missing = [k for k in missing if not any(x in k for x in ['bQ', 'bK', 'bV'])]
    if real_missing:
        print(f"  Warning: Missing keys: {real_missing[:5]}...")

    model = model.to(device)
    model.eval()

    return model, config


# ============================================================================
# Evaluation Tasks
# ============================================================================

def compute_sequence_loss(model, tokens, device):
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(tokens)
        logits = logits[:, :-1, :]
        targets = tokens[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), reduction='mean')
    return loss.item()


def compute_next_token_loss(model, prompt_tokens, expected_token, device):
    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(tokens)
        next_logits = logits[0, -1, :]
        target = torch.tensor([expected_token], dtype=torch.long, device=device)
        loss = F.cross_entropy(next_logits.unsqueeze(0), target)
    return loss.item()


def compute_completion_loss(model, prompt_tokens, completion_tokens, device):
    if not completion_tokens:
        return float('nan')

    full_tokens = prompt_tokens + completion_tokens
    tokens = torch.tensor(full_tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(tokens)
        start_idx = len(prompt_tokens) - 1
        completion_logits = logits[0, start_idx:start_idx+len(completion_tokens), :]
        completion_targets = torch.tensor(completion_tokens, dtype=torch.long, device=device)
        loss = F.cross_entropy(completion_logits, completion_targets, reduction='mean')

    return loss.item()


def get_next_token_probs(model, tokens, device, top_k=10):
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(tokens)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
    return top_indices.cpu().tolist(), top_probs.cpu().tolist()


TASKS = {
    "pattern_numeric": [
        {"prompt": "1, 2, 3, 4,", "expected": " 5", "type": "completion"},
        {"prompt": "2, 4, 6, 8,", "expected": " 10", "type": "completion"},
        {"prompt": "1, 1, 2, 3, 5, 8,", "expected": " 13", "type": "completion"},
        {"prompt": "10, 20, 30, 40,", "expected": " 50", "type": "completion"},
    ],
    "pattern_alpha": [
        {"prompt": "A, B, C, D,", "expected": " E", "type": "completion"},
        {"prompt": "X, Y, Z, A,", "expected": " B", "type": "completion"},
    ],
    "retrieval_near": [
        {"prompt": "The color is blue. The shape is round. The color is", "expected": " blue", "type": "completion"},
        {"prompt": "John is tall. Mary is short. John is", "expected": " tall", "type": "completion"},
    ],
    "retrieval_far": [
        {"prompt": "The capital of France is Paris. The capital of Germany is Berlin. The capital of Spain is Madrid. The capital of Italy is Rome. The capital of France is", "expected": " Paris", "type": "completion"},
        {"prompt": "Alice likes apples. Bob likes bananas. Carol likes cherries. Dave likes dates. Alice likes", "expected": " apples", "type": "completion"},
    ],
    "simple_inference": [
        {"prompt": "If it rains, the ground gets wet. It is raining. The ground", "expected": " gets wet", "type": "completion"},
        {"prompt": "All birds can fly. A sparrow is a bird. A sparrow can", "expected": " fly", "type": "completion"},
    ],
    "negation": [
        {"prompt": "The sun is bright.", "compare": "The sun is not bright.", "type": "perplexity_compare"},
        {"prompt": "Water is wet.", "compare": "Water is not wet.", "type": "perplexity_compare"},
        {"prompt": "Fire is hot.", "compare": "Fire is not hot.", "type": "perplexity_compare"},
    ],
    "syntax": [
        {"prompt": "Hello, world!", "compare": "Hello world", "type": "perplexity_compare"},
        {"prompt": '"I am here," she said.', "compare": "I am here she said", "type": "perplexity_compare"},
    ],
    "copy": [
        {"prompt": "Copy: abc -> abc. Copy: xyz ->", "expected": " xyz", "type": "completion"},
        {"prompt": "Repeat: hello -> hello. Repeat: world ->", "expected": " world", "type": "completion"},
    ],
}


def run_completion_task(model, enc, task, device):
    prompt_tokens = enc.encode(task["prompt"])
    expected_text = task["expected"]
    expected_tokens = enc.encode(expected_text)

    top_indices, top_probs = get_next_token_probs(model, prompt_tokens, device, top_k=10)
    expected_first = expected_tokens[0] if expected_tokens else None

    first_token_loss = compute_next_token_loss(model, prompt_tokens, expected_first, device) if expected_first else float('nan')
    completion_loss = compute_completion_loss(model, prompt_tokens, expected_tokens, device)

    return {
        "prompt": task["prompt"],
        "expected": expected_text,
        "top_predictions": [enc.decode([idx]) for idx in top_indices[:5]],
        "top_probs": top_probs[:5],
        "expected_in_top1": expected_first == top_indices[0] if expected_first else False,
        "expected_in_top5": expected_first in top_indices[:5] if expected_first else False,
        "expected_rank": top_indices.index(expected_first) + 1 if expected_first in top_indices else -1,
        "first_token_loss": first_token_loss,
        "completion_loss": completion_loss,
    }


def run_perplexity_compare_task(model, enc, task, device):
    tokens1 = enc.encode(task["prompt"])
    tokens2 = enc.encode(task["compare"])

    loss1 = compute_sequence_loss(model, tokens1, device)
    loss2 = compute_sequence_loss(model, tokens2, device)

    ppl1 = math.exp(loss1)
    ppl2 = math.exp(loss2)

    return {
        "prompt1": task["prompt"],
        "prompt2": task["compare"],
        "loss1": loss1,
        "loss2": loss2,
        "ppl1": ppl1,
        "ppl2": ppl2,
        "ppl_ratio": ppl2 / ppl1 if ppl1 > 0 else float('inf'),
    }


def evaluate_model(model, enc, device):
    results = {}
    for task_category, tasks in TASKS.items():
        category_results = []
        for task in tasks:
            if task["type"] == "completion":
                result = run_completion_task(model, enc, task, device)
            elif task["type"] == "perplexity_compare":
                result = run_perplexity_compare_task(model, enc, task, device)
            else:
                continue
            category_results.append(result)
        results[task_category] = category_results
    return results


def summarize_results(results):
    summary = {}
    all_completion_losses = []

    for category, tasks in results.items():
        if not tasks:
            continue

        if tasks[0].get("expected_in_top1") is not None:
            top1_acc = sum(1 for t in tasks if t["expected_in_top1"]) / len(tasks)
            top5_acc = sum(1 for t in tasks if t["expected_in_top5"]) / len(tasks)
            avg_rank = np.mean([t["expected_rank"] for t in tasks if t["expected_rank"] > 0]) if any(t["expected_rank"] > 0 for t in tasks) else -1

            valid_losses = [t["first_token_loss"] for t in tasks if not math.isnan(t["first_token_loss"])]
            avg_first_token_loss = np.mean(valid_losses) if valid_losses else float('nan')

            valid_completion_losses = [t["completion_loss"] for t in tasks if not math.isnan(t["completion_loss"])]
            avg_completion_loss = np.mean(valid_completion_losses) if valid_completion_losses else float('nan')

            all_completion_losses.extend(valid_losses)

            summary[category] = {
                "top1_accuracy": top1_acc,
                "top5_accuracy": top5_acc,
                "avg_rank": avg_rank,
                "avg_first_token_loss": avg_first_token_loss,
                "avg_completion_loss": avg_completion_loss,
                "n_tasks": len(tasks),
            }
        else:
            avg_ratio = np.mean([t["ppl_ratio"] for t in tasks])
            avg_loss1 = np.mean([t["loss1"] for t in tasks])
            avg_loss2 = np.mean([t["loss2"] for t in tasks])

            summary[category] = {
                "avg_ppl_ratio": avg_ratio,
                "avg_loss_correct": avg_loss1,
                "avg_loss_incorrect": avg_loss2,
                "n_tasks": len(tasks),
            }

    if all_completion_losses:
        summary["_overall"] = {
            "avg_logic_puzzle_loss": np.mean(all_completion_losses),
            "n_completion_tasks": len(all_completion_losses),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate GELU ablation models on logic puzzles')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint to evaluate')
    parser.add_argument('--output', type=str, default='logic_eval_gelu.json', help='Output file')
    parser.add_argument('--use_biases', action='store_true', default=False,
                        help='Apply bQ/bV biases (for bQ+bV models)')
    parser.add_argument('--mean_Q', type=float, default=None, help='Override mean_Q value')
    parser.add_argument('--mean_V', type=float, default=None, help='Override mean_V value')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    enc = tiktoken.get_encoding("gpt2")

    print(f"\n{'='*70}")
    print(f"GELU ABLATION - Logic Puzzle Evaluation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Use biases: {args.use_biases}")
    print(f"{'='*70}")

    model, config = load_gelu_model(
        args.checkpoint, device,
        use_biases=args.use_biases,
        override_mean_Q=args.mean_Q,
        override_mean_V=args.mean_V
    )

    print(f"\nRunning evaluation...")
    results = evaluate_model(model, enc, device)
    summary = summarize_results(results)

    output_data = {
        "checkpoint": args.checkpoint,
        "model_type": "gelu_ablation",
        "mlp_type": "GELU",
        "use_biases": args.use_biases,
        "summary": summary,
        "detailed": results,
    }

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if "_overall" in summary:
        overall = summary["_overall"]
        print(f"\n*** OVERALL LOGIC PUZZLE LOSS: {overall['avg_logic_puzzle_loss']:.4f} ***")
        print(f"    (averaged over {overall['n_completion_tasks']} completion tasks)")

    print()
    for category, stats in summary.items():
        if category.startswith("_"):
            continue
        print(f"{category}:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
        print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
