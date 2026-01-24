#!/usr/bin/env python3
"""
Unified Training Script for ECD Symmetry Breaking Paper

Supports:
- Symmetric mode (standard attention, no bQ) or Disordered mode (bQ symmetry breaking)
- Optional bV for V-O symmetry breaking
- Multiple optimizers: ECD, Adam, SGDM, SOAP

Default hyperparameters match tested seed 789/83 runs:
- vocab_size = 50304
- ecd_lr = 1.0, ecd_eta = 100, ecd_F0 = 2.0
- mean_Q = 0.5, mean_V = 0.5 (for bQ+bV runs)

Usage Examples:
    # Symmetric baseline (standard attention, no bQ)
    python train.py --symmetric --optimizer adam

    # Disordered with bQ only (default)
    python train.py --optimizer ecd

    # Disordered with bQ + bV
    python train.py --use_v_bias --mean_V 0.5 --std_V 0.05 --optimizer ecd
"""

import os
import sys
import math
import time
import argparse

import torch
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecd_symbreak.config import GPTConfig, PRESETS
from ecd_symbreak.optimizer import ECD_q1_scaled
from ecd_symbreak.model import GPT
from ecd_symbreak.data import FineWebShards
from ecd_symbreak.utils import (
    get_device_type,
    serialize_optimizer,
    wandb_record_optimizer,
)

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from soap import SOAP
    SOAP_AVAILABLE = True
except ImportError:
    SOAP_AVAILABLE = False


def build_optimizer(model, kind, ecd_kwargs=None, adamw_kwargs=None,
                    sgdm_kwargs=None, soap_kwargs=None):
    """Build optimizer based on kind."""
    kind = kind.lower()

    if kind == "ecd":
        return ECD_q1_scaled(model.parameters(), **(ecd_kwargs or {}))

    if kind == "adam":
        kw = dict(weight_decay=0.1, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, fused=True)
        kw.update(adamw_kwargs or {})
        return torch.optim.AdamW(model.parameters(), **kw)

    if kind == "sgdm":
        kw = dict(lr=0.03, momentum=0.95, nesterov=True, weight_decay=0.0)
        kw.update(sgdm_kwargs or {})
        return torch.optim.SGD(model.parameters(), **kw)

    if kind == "soap":
        if not SOAP_AVAILABLE:
            raise ImportError("SOAP not available. Install from soap.py")
        kw = dict(lr=3e-4, betas=(0.95, 0.95), weight_decay=0.01, precondition_frequency=10)
        kw.update(soap_kwargs or {})
        return SOAP(model.parameters(), **kw)

    raise ValueError(f"Unknown optimizer {kind}")


def main():
    ap = argparse.ArgumentParser(
        description="Unified GPT training with symmetric/disordered modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model architecture
    ap.add_argument("--model", choices=PRESETS.keys(), default="124m")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=50304)

    # Training parameters
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--train_tokens", type=float, default=5e8)
    ap.add_argument("--valid_every_updates", type=int, default=1000)

    # Optimizer selection
    ap.add_argument("--optimizer", choices=["ecd", "adam", "sgdm", "soap"], default="ecd")

    # MODE: symmetric vs disordered
    ap.add_argument("--symmetric", action="store_true", default=False,
                    help="Use symmetric architecture (standard attention, no bQ)")

    # ECD hyperparameters (defaults from tested runs)
    ap.add_argument("--ecd_lr", type=float, default=1.0)
    ap.add_argument("--ecd_eta", type=float, default=100.0)
    ap.add_argument("--ecd_F0", type=float, default=2.0)
    ap.add_argument("--ecd_eps1", type=float, default=1e-10)
    ap.add_argument("--ecd_eps2", type=float, default=1e-40)
    ap.add_argument("--ecd_nu", type=float, default=0.0)
    ap.add_argument("--ecd_wd", type=float, default=0.0)
    ap.add_argument("--ecd_consEn", action="store_true", default=True)
    ap.add_argument("--no_ecd_consEn", dest="ecd_consEn", action="store_false")

    # AdamW hyperparameters
    ap.add_argument("--adam_lr", type=float, default=1e-4)
    ap.add_argument("--adam_wd", type=float, default=0.01)
    ap.add_argument("--adam_beta1", type=float, default=0.9)
    ap.add_argument("--adam_beta2", type=float, default=0.95)

    # SGDM hyperparameters
    ap.add_argument("--sgdm_lr", type=float, default=0.03)
    ap.add_argument("--sgdm_momentum", type=float, default=0.95)
    ap.add_argument("--sgdm_nesterov", action="store_true", default=True)

    # SOAP hyperparameters
    ap.add_argument("--soap_lr", type=float, default=3e-4)
    ap.add_argument("--soap_beta1", type=float, default=0.95)
    ap.add_argument("--soap_beta2", type=float, default=0.95)
    ap.add_argument("--soap_wd", type=float, default=0.01)
    ap.add_argument("--soap_prefreq", type=int, default=10)

    # Data loader options
    ap.add_argument("--data_ext", choices=["auto", "bin", "npy", "pt"], default="auto")
    ap.add_argument("--sanity_check_tokens", action="store_true")

    # Bias controls - Q-K sector
    ap.add_argument("--disable_k_bias", action="store_true",
                    help="Use only bQ; set bK=0")

    # Bias controls - V-O sector
    ap.add_argument("--use_v_bias", action="store_true", default=False,
                    help="Enable bV for V-O symmetry breaking")
    ap.add_argument("--mean_V", type=float, default=0.5,
                    help="Mean of bV distribution (default: 0.5 for tested runs)")
    ap.add_argument("--std_V", type=float, default=0.05,
                    help="Std of bV distribution")

    # Architecture matching
    ap.add_argument("--use_prelu", action="store_true", default=True,
                    help="Use PReLU MLP (default for symmetry breaking)")
    ap.add_argument("--use_gelu", dest="use_prelu", action="store_false",
                    help="Use GELU MLP instead of PReLU")

    # Run configuration
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default="./data/finewebedu10B")
    ap.add_argument("--log_dir", type=str, default="runs")
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--save_every_updates", type=int, default=0)
    ap.add_argument("--use_bf16", action="store_true")
    ap.add_argument("--wandb", action="store_true")

    args = ap.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Build config
    cfg = PRESETS[args.model]
    cfg.context_length = args.ctx
    cfg.vocab_size = args.vocab

    device = get_device_type()
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available()) else torch.float32
    torch.set_float32_matmul_precision("high")

    # Build model
    model = GPT(
        cfg,
        symmetric=args.symmetric,
        use_k_bias=(not args.disable_k_bias),
        use_v_bias=args.use_v_bias,
        mean_V=args.mean_V if args.use_v_bias else None,
        std_V=args.std_V if args.use_v_bias else None,
        use_prelu=args.use_prelu,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Print configuration
    mode_str = "SYMMETRIC" if args.symmetric else "DISORDERED"
    print(f"=== Configuration ===")
    print(f"  Mode: {mode_str}")
    print(f"  Model: {args.model} ({n_params:,} params)")
    if args.symmetric:
        print(f"  Q-K sector: bQ=disabled (symmetric)")
    else:
        print(f"  Q-K sector: bQ=enabled, bK={'enabled' if not args.disable_k_bias else 'disabled'}")
    print(f"  V-O sector: bV={'enabled' if args.use_v_bias else 'disabled'}")
    if args.use_v_bias:
        print(f"    mean_V={args.mean_V}, std_V={args.std_V}")
    print(f"  MLP: {'PReLU (learnable)' if args.use_prelu else 'GELU'}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"=====================")

    # Data loaders
    T = cfg.context_length
    B = args.batch_size
    gdup = args.grad_accum
    train_loader = FineWebShards(
        args.data_root, "train", B=B, T=T,
        data_ext=args.data_ext, vocab_size=cfg.vocab_size,
        sanity_check=args.sanity_check_tokens
    )
    val_loader = FineWebShards(
        args.data_root, "val", B=B, T=T,
        data_ext=args.data_ext, vocab_size=cfg.vocab_size,
        sanity_check=args.sanity_check_tokens
    )

    # W&B setup
    mode_tag = "symmetric" if args.symmetric else "disorder"
    symbreak_tag = "bQ" if not args.symmetric else "nobQ"
    if args.use_v_bias:
        symbreak_tag += "+bV"

    run_name = args.name or f"{args.model}-{args.optimizer}-{mode_tag}-{symbreak_tag}-seed{args.seed}"
    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.wandb and WANDB_AVAILABLE:
        tags = [args.optimizer, args.model, mode_tag]
        if args.use_v_bias:
            tags.append("bV")
        wandb.init(
            project="ecd-symbreak",
            name=run_name,
            tags=tags,
            config={
                "dataset": "finewebedu10B",
                "model": vars(cfg),
                "params": n_params,
                "train_tokens": args.train_tokens,
                "opt_kind": args.optimizer,
                "mode": mode_tag,
                "symmetry_breaking": {
                    "symmetric": args.symmetric,
                    "bQ": not args.symmetric,
                    "bK": not args.symmetric and not args.disable_k_bias,
                    "bV": args.use_v_bias,
                    "mean_V": args.mean_V if args.use_v_bias else None,
                    "std_V": args.std_V if args.use_v_bias else None,
                },
            }
        )

    # Build optimizer
    if args.optimizer == "ecd":
        ecd_kwargs = dict(
            lr=args.ecd_lr, F0=args.ecd_F0, eps1=args.ecd_eps1,
            eps2=args.ecd_eps2, nu=args.ecd_nu, weight_decay=args.ecd_wd,
            eta=args.ecd_eta, consEn=args.ecd_consEn
        )
        opt = build_optimizer(model, "ecd", ecd_kwargs=ecd_kwargs)
        if args.wandb and WANDB_AVAILABLE:
            wandb_record_optimizer(wandb.run, "ecd", opt, ecd_kwargs=ecd_kwargs)
    elif args.optimizer == "adam":
        adamw_kwargs = dict(
            lr=args.adam_lr, weight_decay=args.adam_wd,
            betas=(args.adam_beta1, args.adam_beta2)
        )
        opt = build_optimizer(model, "adam", adamw_kwargs=adamw_kwargs)
        if args.wandb and WANDB_AVAILABLE:
            wandb_record_optimizer(wandb.run, "adam", opt)
    elif args.optimizer == "sgdm":
        sgdm_kwargs = dict(
            lr=args.sgdm_lr, momentum=args.sgdm_momentum,
            nesterov=args.sgdm_nesterov
        )
        opt = build_optimizer(model, "sgdm", sgdm_kwargs=sgdm_kwargs)
        if args.wandb and WANDB_AVAILABLE:
            wandb_record_optimizer(wandb.run, "sgdm", opt)
    elif args.optimizer == "soap":
        soap_kwargs = dict(
            lr=args.soap_lr, betas=(args.soap_beta1, args.soap_beta2),
            weight_decay=args.soap_wd, precondition_frequency=args.soap_prefreq
        )
        opt = build_optimizer(model, "soap", soap_kwargs=soap_kwargs)
        if args.wandb and WANDB_AVAILABLE:
            wandb_record_optimizer(wandb.run, "soap", opt)

    # Training loop
    consumed_tokens = 0
    update = 0
    best_val_loss = float('inf')
    model.train()
    start = time.time()

    while consumed_tokens < args.train_tokens:
        t0 = time.time()
        total_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for _ in range(gdup):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(get_device_type(), enabled=(dtype == torch.bfloat16), dtype=dtype):
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
            wandb.log({
                "train/loss": avg_loss.item(),
                "speed/tok_per_s": tok_per_s,
                "train/consumed_tokens": consumed_tokens,
                "update": update
            })

        if update % max(1, args.valid_every_updates) == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = val_loader.next_batch()
                vx, vy = vx.to(device), vy.to(device)
                with torch.autocast(get_device_type(), enabled=(dtype == torch.bfloat16), dtype=dtype):
                    _, vloss = model(vx, vy)
            val_loss = vloss.item()

            if args.wandb and WANDB_AVAILABLE:
                wandb.log({"val/loss": val_loss, "update": update})

            print(f"[up {update:6d}] loss {avg_loss.item():.4f} | val {val_loss:.4f} "
                  f"| tok/s {tok_per_s:,.0f} | consumed {int(consumed_tokens):,}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(ckpt_dir, "model_best.pt")
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg,
                    "update": update,
                    "consumed_tokens": consumed_tokens,
                    "val_loss": val_loss,
                    "symmetric": args.symmetric,
                    "symmetry_breaking": {
                        "symmetric": args.symmetric,
                        "bQ": not args.symmetric,
                        "bK": not args.symmetric and not args.disable_k_bias,
                        "bV": args.use_v_bias,
                        "mean_V": args.mean_V,
                        "std_V": args.std_V,
                    }
                }, best_path)
                print(f"  -> New best! Saved to {best_path}")

            model.train()

        if args.save_every_updates and (update % args.save_every_updates == 0):
            path = os.path.join(ckpt_dir, f"model_{update:06d}.pt")
            torch.save({
                "model": model.state_dict(),
                "config": cfg,
                "update": update,
                "consumed_tokens": consumed_tokens
            }, path)

    # Final save
    final_path = os.path.join(ckpt_dir, "model_final.pt")
    torch.save({
        "model": model.state_dict(),
        "config": cfg,
        "update": update,
        "consumed_tokens": consumed_tokens,
        "best_val_loss": best_val_loss,
        "symmetric": args.symmetric,
        "symmetry_breaking": {
            "symmetric": args.symmetric,
            "bQ": not args.symmetric,
            "bK": not args.symmetric and not args.disable_k_bias,
            "bV": args.use_v_bias,
            "mean_V": args.mean_V,
            "std_V": args.std_V,
        }
    }, final_path)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Saved to {final_path}. Total time: {time.time()-start:.1f}s")

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
