#!/usr/bin/env python3
"""
Simple Logic and Reasoning Evaluation for bQ Analysis

Tests whether models with stronger bQ alignment (ECD) perform differently
on simple reasoning tasks compared to models with weaker alignment (AdamW, SOAP).

Task categories designed for 124M models:
1. Pattern completion - simple sequences
2. Retrieval - near vs far context
3. Simple inference - basic logical structure
4. Negation - understanding "not"
5. Syntax sensitivity - punctuation/structure matters

The hypothesis: If bQ alignment provides useful inductive bias (e.g., recency),
models with stronger alignment might show different behavior on these tasks.
"""

import torch
import torch.nn.functional as F
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# For tokenization
import tiktoken

# ============================================================================
# Model Loading (adapted from analysis scripts)
# ============================================================================

@dataclass
class GPTConfig:
    context_length: int = 512
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


def load_model_for_eval(ckpt_path, device='cuda'):
    """Load a checkpoint for evaluation."""
    import torch.nn as nn
    import math

    # Define model architecture (simplified for eval - no bQ resampling needed)
    class CausalSelfAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.n_head = config.n_head
            self.d = config.n_embd // config.n_head
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd)
            mask = torch.tril(torch.ones(config.context_length, config.context_length))
            self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.c_attn(x)
            q, k, v = qkv.split(C, dim=2)
            q = q.view(B, T, self.n_head, self.d).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.d).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.d).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
            scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(scores, dim=-1)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(y)

    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        def forward(self, x):
            return self.c_proj(F.gelu(self.c_fc(x)))

    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = nn.LayerNorm(config.n_embd)
            self.mlp = MLP(config)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

    class GPT(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.context_length, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
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

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Get config
    config_dict = ckpt.get('config', {})

    # Model size presets
    MODEL_PRESETS = {
        '124m': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
        '355m': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},
    }

    if isinstance(config_dict, dict) and 'model' in config_dict:
        preset = MODEL_PRESETS.get(config_dict['model'], MODEL_PRESETS['124m'])
        config = GPTConfig(
            context_length=config_dict.get('context_length', 512),
            vocab_size=config_dict.get('vocab_size', 50304),
            n_layer=preset['n_layer'],
            n_head=preset['n_head'],
            n_embd=preset['n_embd']
        )
    else:
        # Try to get from config object or use defaults
        config = GPTConfig(
            context_length=getattr(config_dict, 'context_length', 512),
            vocab_size=getattr(config_dict, 'vocab_size', 50304),
            n_layer=getattr(config_dict, 'n_layer', 12),
            n_head=getattr(config_dict, 'n_head', 12),
            n_embd=getattr(config_dict, 'n_embd', 768)
        )

    # Build model
    model = GPT(config)

    # Load state dict
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        raise KeyError(f"No model weights found. Keys: {list(ckpt.keys())}")

    # Clean up state dict keys
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace('_orig_mod.', '')
        # Map disordered attention keys to standard attention
        k = k.replace('.sa.', '.attn.')
        cleaned[k] = v

    # Load with strict=False to handle bQ/bK buffers
    model.load_state_dict(cleaned, strict=False)
    model = model.to(device)
    model.eval()

    return model, config


# ============================================================================
# Evaluation Tasks
# ============================================================================

def compute_perplexity(model, tokens, device):
    """Compute perplexity for a sequence of tokens."""
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(tokens)
        # Shift for next-token prediction
        logits = logits[:, :-1, :]
        targets = tokens[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), reduction='mean')
    return torch.exp(loss).item()


def get_next_token_probs(model, tokens, device, top_k=10):
    """Get probabilities for next token."""
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(tokens)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
    return top_indices.cpu().tolist(), top_probs.cpu().tolist()


def generate(model, tokens, max_new_tokens, device, temperature=1.0):
    """Generate tokens autoregressively."""
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens[:, -512:])  # Truncate to context length
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
    return tokens[0].cpu().tolist()


# ============================================================================
# Task Definitions
# ============================================================================

TASKS = {
    # Pattern completion - tests sequence understanding
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

    # Retrieval - tests whether recency bias helps or hurts
    "retrieval_near": [
        {"prompt": "The color is blue. The shape is round. The color is", "expected": " blue", "type": "completion"},
        {"prompt": "John is tall. Mary is short. John is", "expected": " tall", "type": "completion"},
    ],

    "retrieval_far": [
        {"prompt": "The capital of France is Paris. The capital of Germany is Berlin. The capital of Spain is Madrid. The capital of Italy is Rome. The capital of France is", "expected": " Paris", "type": "completion"},
        {"prompt": "Alice likes apples. Bob likes bananas. Carol likes cherries. Dave likes dates. Alice likes", "expected": " apples", "type": "completion"},
    ],

    # Simple inference
    "simple_inference": [
        {"prompt": "If it rains, the ground gets wet. It is raining. The ground", "expected": " gets wet", "type": "completion"},
        {"prompt": "All birds can fly. A sparrow is a bird. A sparrow can", "expected": " fly", "type": "completion"},
    ],

    # Negation sensitivity - compare perplexity
    "negation": [
        {"prompt": "The sun is bright.", "compare": "The sun is not bright.", "type": "perplexity_compare"},
        {"prompt": "Water is wet.", "compare": "Water is not wet.", "type": "perplexity_compare"},
        {"prompt": "Fire is hot.", "compare": "Fire is not hot.", "type": "perplexity_compare"},
    ],

    # Syntax/structure - punctuation matters (high bQ alignment tokens)
    "syntax": [
        {"prompt": "Hello, world!", "compare": "Hello world", "type": "perplexity_compare"},
        {"prompt": '"I am here," she said.', "compare": "I am here she said", "type": "perplexity_compare"},
    ],

    # Copying - exact retrieval
    "copy": [
        {"prompt": "Copy: abc -> abc. Copy: xyz ->", "expected": " xyz", "type": "completion"},
        {"prompt": "Repeat: hello -> hello. Repeat: world ->", "expected": " world", "type": "completion"},
    ],
}


def run_completion_task(model, enc, task, device):
    """Run a completion task and check if expected token is top prediction."""
    prompt_tokens = enc.encode(task["prompt"])
    expected_tokens = enc.encode(task["expected"])

    top_indices, top_probs = get_next_token_probs(model, prompt_tokens, device, top_k=10)

    # Check if first expected token is in top predictions
    expected_first = expected_tokens[0] if expected_tokens else None

    result = {
        "prompt": task["prompt"],
        "expected": task["expected"],
        "top_predictions": [enc.decode([idx]) for idx in top_indices[:5]],
        "top_probs": top_probs[:5],
        "expected_in_top1": expected_first == top_indices[0] if expected_first else False,
        "expected_in_top5": expected_first in top_indices[:5] if expected_first else False,
        "expected_rank": top_indices.index(expected_first) + 1 if expected_first in top_indices else -1,
    }

    return result


def run_perplexity_compare_task(model, enc, task, device):
    """Compare perplexity between two prompts."""
    tokens1 = enc.encode(task["prompt"])
    tokens2 = enc.encode(task["compare"])

    ppl1 = compute_perplexity(model, tokens1, device)
    ppl2 = compute_perplexity(model, tokens2, device)

    return {
        "prompt1": task["prompt"],
        "prompt2": task["compare"],
        "ppl1": ppl1,
        "ppl2": ppl2,
        "ppl_ratio": ppl2 / ppl1 if ppl1 > 0 else float('inf'),
    }


def evaluate_model(model, enc, device):
    """Run all evaluation tasks on a model."""
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
    """Compute summary statistics from results."""
    summary = {}

    for category, tasks in results.items():
        if not tasks:
            continue

        if tasks[0].get("expected_in_top1") is not None:
            # Completion task
            top1_acc = sum(1 for t in tasks if t["expected_in_top1"]) / len(tasks)
            top5_acc = sum(1 for t in tasks if t["expected_in_top5"]) / len(tasks)
            avg_rank = np.mean([t["expected_rank"] for t in tasks if t["expected_rank"] > 0]) if any(t["expected_rank"] > 0 for t in tasks) else -1

            summary[category] = {
                "top1_accuracy": top1_acc,
                "top5_accuracy": top5_acc,
                "avg_rank": avg_rank,
                "n_tasks": len(tasks),
            }
        else:
            # Perplexity comparison
            avg_ratio = np.mean([t["ppl_ratio"] for t in tasks])
            summary[category] = {
                "avg_ppl_ratio": avg_ratio,
                "n_tasks": len(tasks),
            }

    return summary


# ============================================================================
# Main
# ============================================================================

CHECKPOINTS = {
    "ecd_disordered": "experiments/ecd_vs_soap/checkpoints/ecd_best_disordered/ecd-124m-ckpt-best-lrhat1.0-seed42_best.pt",
    "adamw_disordered": "experiments/ecd_vs_soap/checkpoints/adamw_best_disordered/adamw-124m-ckpt-best-lr0.0001-seed42_best.pt",
    "sgdm_disordered": "experiments/ecd_vs_soap/results/124m_extended/sgdm_best_disordered/sgdm-124m-extended-disordered-lr0.0300-seed42_best.pt",
    "sgdm_symmetric": "experiments/ecd_vs_soap/results/124m_extended/sgdm_best_symmetric/sgdm-124m-extended-symmetric-lr0.0300-seed42_best.pt",
    # ECD symmetric will be added after it finishes training
}


def main():
    parser = argparse.ArgumentParser(description='Evaluate models on logic puzzles')
    parser.add_argument('--checkpoint', type=str, help='Single checkpoint to evaluate')
    parser.add_argument('--all', action='store_true', help='Evaluate all checkpoints')
    parser.add_argument('--output', type=str, default='logic_eval_results.json', help='Output file')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")

    all_results = {}

    if args.checkpoint:
        checkpoints = {"custom": args.checkpoint}
    elif args.all:
        checkpoints = CHECKPOINTS
    else:
        print("Please specify --checkpoint or --all")
        return

    for name, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        if not Path(ckpt_path).exists():
            print(f"  Checkpoint not found: {ckpt_path}")
            continue

        try:
            model, config = load_model_for_eval(ckpt_path, device)
            results = evaluate_model(model, enc, device)
            summary = summarize_results(results)

            all_results[name] = {
                "checkpoint": ckpt_path,
                "summary": summary,
                "detailed": results,
            }

            print(f"\nSummary for {name}:")
            for category, stats in summary.items():
                print(f"  {category}:")
                for k, v in stats.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.3f}")
                    else:
                        print(f"    {k}: {v}")

            # Free memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error evaluating {name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Print comparison table
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)

        categories = set()
        for r in all_results.values():
            categories.update(r.get("summary", {}).keys())

        for category in sorted(categories):
            print(f"\n{category}:")
            for name, data in all_results.items():
                stats = data.get("summary", {}).get(category, {})
                if "top1_accuracy" in stats:
                    print(f"  {name}: top1={stats['top1_accuracy']:.1%}, top5={stats['top5_accuracy']:.1%}")
                elif "avg_ppl_ratio" in stats:
                    print(f"  {name}: ppl_ratio={stats['avg_ppl_ratio']:.3f}")


if __name__ == "__main__":
    main()
