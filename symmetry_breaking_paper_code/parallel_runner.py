#!/usr/bin/env python3
"""
Parallel Experiment Runner
==========================
Runs multiple independent experiments across available GPUs.

Usage:
    python parallel_runner.py --config experiments.yaml
    python parallel_runner.py --dry-run  # Show what would run without executing

Example experiments.yaml:
    gpus: [0, 1, 2, 3]
    base_cmd: "python train_gpt_full_symbreak_rope_flash.py"
    common_args:
      --model: 124m
      --train_tokens: 5e8
      --use_bf16: true
    experiments:
      - name: ecd_bV_seed42
        args: {--optimizer: ecd, --use_v_bias: true, --seed: 42}
      - name: ecd_bV_seed123
        args: {--optimizer: ecd, --use_v_bias: true, --seed: 123}
      - name: adam_bV_seed42
        args: {--optimizer: adam, --use_v_bias: true, --seed: 42}
"""

import argparse
import subprocess
import os
import sys
import time
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import queue

@dataclass
class Experiment:
    name: str
    cmd: str
    gpu: int
    log_file: str
    status: str = "pending"  # pending, running, completed, failed
    process: Optional[subprocess.Popen] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class ParallelRunner:
    def __init__(self, gpus: List[int], log_dir: str = "parallel_logs"):
        self.gpus = gpus
        self.available_gpus = queue.Queue()
        for gpu in gpus:
            self.available_gpus.put(gpu)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiments: List[Experiment] = []
        self.lock = threading.Lock()

    def add_experiment(self, name: str, cmd: str):
        """Add an experiment to the queue."""
        log_file = str(self.log_dir / f"{name}.log")
        exp = Experiment(name=name, cmd=cmd, gpu=-1, log_file=log_file)
        self.experiments.append(exp)

    def _run_experiment(self, exp: Experiment, gpu: int):
        """Run a single experiment on a specific GPU."""
        exp.gpu = gpu
        exp.status = "running"
        exp.start_time = time.time()

        # Prepend CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        print(f"[GPU {gpu}] Starting: {exp.name}")

        try:
            with open(exp.log_file, 'w') as log:
                log.write(f"=== {exp.name} ===\n")
                log.write(f"GPU: {gpu}\n")
                log.write(f"Command: {exp.cmd}\n")
                log.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.write("=" * 50 + "\n\n")
                log.flush()

                exp.process = subprocess.Popen(
                    exp.cmd,
                    shell=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env
                )
                exp.process.wait()

            exp.end_time = time.time()
            duration = exp.end_time - exp.start_time

            if exp.process.returncode == 0:
                exp.status = "completed"
                print(f"[GPU {gpu}] Completed: {exp.name} ({duration/60:.1f} min)")
            else:
                exp.status = "failed"
                print(f"[GPU {gpu}] FAILED: {exp.name} (exit code {exp.process.returncode})")

        except Exception as e:
            exp.status = "failed"
            exp.end_time = time.time()
            print(f"[GPU {gpu}] ERROR: {exp.name} - {e}")

        finally:
            # Return GPU to pool
            self.available_gpus.put(gpu)

    def run_all(self, dry_run: bool = False):
        """Run all experiments, scheduling across available GPUs."""
        if dry_run:
            print("=== DRY RUN - Would execute: ===\n")
            for i, exp in enumerate(self.experiments):
                gpu = self.gpus[i % len(self.gpus)]
                print(f"[GPU {gpu}] {exp.name}")
                print(f"    {exp.cmd}\n")
            return

        print(f"=== Running {len(self.experiments)} experiments on GPUs {self.gpus} ===\n")

        threads = []
        exp_queue = list(self.experiments)

        while exp_queue or any(t.is_alive() for t in threads):
            # Clean up finished threads
            threads = [t for t in threads if t.is_alive()]

            # Start new experiments if GPUs available
            while exp_queue and not self.available_gpus.empty():
                exp = exp_queue.pop(0)
                gpu = self.available_gpus.get()

                t = threading.Thread(target=self._run_experiment, args=(exp, gpu))
                t.start()
                threads.append(t)

            time.sleep(1)

        # Wait for all to complete
        for t in threads:
            t.join()

        # Print summary
        print("\n" + "=" * 50)
        print("=== SUMMARY ===")
        print("=" * 50)

        completed = [e for e in self.experiments if e.status == "completed"]
        failed = [e for e in self.experiments if e.status == "failed"]

        print(f"\nCompleted: {len(completed)}/{len(self.experiments)}")
        for exp in completed:
            duration = (exp.end_time - exp.start_time) / 60
            print(f"  ✓ {exp.name} ({duration:.1f} min)")

        if failed:
            print(f"\nFailed: {len(failed)}")
            for exp in failed:
                print(f"  ✗ {exp.name} - see {exp.log_file}")

        print(f"\nLogs saved to: {self.log_dir}/")


def build_cmd(base_cmd: str, common_args: Dict, exp_args: Dict) -> str:
    """Build command string from base command and arguments."""
    parts = [base_cmd]

    # Add common args
    for key, val in common_args.items():
        if val is True:
            parts.append(key)
        elif val is not False and val is not None:
            parts.append(f"{key} {val}")

    # Add experiment-specific args
    for key, val in exp_args.items():
        if val is True:
            parts.append(key)
        elif val is not False and val is not None:
            parts.append(f"{key} {val}")

    return " ".join(parts)


# ===== Example usage for our experiments =====

def create_bV_statistics_experiments():
    """Create experiments for gathering more bV statistics."""
    base_cmd = "python3 train_gpt_unified.py"

    common = {
        "--model": "124m",
        "--train_tokens": "5e8",
        "--batch_size": "8",
        "--use_bf16": True,
        "--wandb": True,
        "--valid_every_updates": "1000",
    }

    experiments = []

    # More seeds for bV experiments
    for std_v in [0.02, 0.05]:
        for seed in [456, 789, 1000, 2000]:
            exp = {
                "name": f"ecd_bV_stdV{std_v}_seed{seed}",
                "args": {
                    "--optimizer": "ecd",
                    "--use_v_bias": True,
                    "--std_V": std_v,
                    "--seed": seed,
                    "--log_dir": "experiments/ecd_vs_soap/results/bV_extended",
                    "--name": f"ecd-bV-stdV{std_v}-seed{seed}",
                }
            }
            experiments.append(exp)

    return base_cmd, common, experiments


def create_rope_flash_comparison():
    """Create experiments comparing RoPE and Flash attention."""
    base_cmd = "python3 train_gpt_unified.py"

    common = {
        "--model": "124m",
        "--train_tokens": "5e8",
        "--batch_size": "8",
        "--use_bf16": True,
        "--wandb": True,
        "--valid_every_updates": "1000",
        "--use_v_bias": True,
        "--std_V": "0.05",
    }

    experiments = []

    for use_rope in [False, True]:
        for use_flash in [False, True]:
            for seed in [42, 123]:
                rope_tag = "rope" if use_rope else "learned"
                flash_tag = "flash" if use_flash else "std"

                exp = {
                    "name": f"ecd_bV_{rope_tag}_{flash_tag}_seed{seed}",
                    "args": {
                        "--optimizer": "ecd",
                        "--use_rope": use_rope,
                        "--use_flash": use_flash,
                        "--seed": seed,
                        "--log_dir": "experiments/ecd_vs_soap/results/rope_flash_comparison",
                        "--name": f"ecd-bV-{rope_tag}-{flash_tag}-seed{seed}",
                    }
                }
                experiments.append(exp)

    return base_cmd, common, experiments


def create_symmetric_vs_disorder_comparison():
    """Create experiments comparing symmetric vs disordered architectures across optimizers."""
    base_cmd = "python3 train_gpt_unified.py"

    common = {
        "--model": "124m",
        "--train_tokens": "5e8",
        "--batch_size": "8",
        "--use_bf16": True,
        "--wandb": True,
        "--valid_every_updates": "1000",
    }

    experiments = []

    # Compare symmetric vs disordered for each optimizer
    for optimizer in ["ecd", "adam", "sgdm", "soap"]:
        for symmetric in [True, False]:
            for seed in [42, 123]:
                mode_tag = "sym" if symmetric else "dis"

                exp = {
                    "name": f"{optimizer}_{mode_tag}_seed{seed}",
                    "args": {
                        "--optimizer": optimizer,
                        "--symmetric": symmetric,
                        "--seed": seed,
                        "--log_dir": "experiments/ecd_vs_soap/results/sym_vs_dis",
                        "--name": f"{optimizer}-{mode_tag}-seed{seed}",
                    }
                }
                experiments.append(exp)

    return base_cmd, common, experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel experiments")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--experiment-set", choices=["bV_stats", "rope_flash", "sym_vs_dis"],
                        default="bV_stats",
                        help="Which experiment set to run")
    parser.add_argument("--log-dir", type=str, default="parallel_logs",
                        help="Directory for log files")

    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    # Select experiment set
    if args.experiment_set == "bV_stats":
        base_cmd, common, experiments = create_bV_statistics_experiments()
    elif args.experiment_set == "rope_flash":
        base_cmd, common, experiments = create_rope_flash_comparison()
    elif args.experiment_set == "sym_vs_dis":
        base_cmd, common, experiments = create_symmetric_vs_disorder_comparison()

    # Create runner
    runner = ParallelRunner(gpus=gpus, log_dir=args.log_dir)

    # Add experiments
    for exp in experiments:
        cmd = build_cmd(base_cmd, common, exp["args"])
        runner.add_experiment(exp["name"], cmd)

    # Run
    runner.run_all(dry_run=args.dry_run)
