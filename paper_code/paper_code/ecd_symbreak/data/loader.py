"""
FineWeb data loader for ECD symmetry breaking experiments.

Supports multiple file formats (.bin, .npy, .pt) with automatic detection.
"""

import os
import numpy as np
import torch


def load_tokens_autodetect(filename):
    """
    Auto-detect file format and load tokens.

    Supports:
    - .bin: Binary files with optional 256-int32 header
    - .npy: NumPy array files
    - .pt/.pth: PyTorch tensor files

    Args:
        filename: Path to token file

    Returns:
        torch.Tensor of token indices (int64)
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".bin":
        # Handle binary files with header (256 int32s)
        with open(filename, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=256)
            if header[0] == 20240520:  # Magic number indicates header present
                num_tokens = header[2]
                tokens = np.fromfile(f, dtype=np.uint16, count=num_tokens)
                return torch.from_numpy(tokens.astype(np.int64))
            else:
                # No header, read entire file
                f.seek(0)
                arr = np.fromfile(f, dtype=np.uint16)
                return torch.tensor(arr.astype(np.int64), dtype=torch.long)

    if ext == ".npy":
        arr = np.load(filename)
        arr = arr.astype(np.int64)
        return torch.tensor(arr, dtype=torch.long)

    if ext in (".pt", ".pth"):
        t = torch.load(filename)
        return t.long() if torch.is_tensor(t) else torch.tensor(
            np.asarray(t).astype(np.int64), dtype=torch.long
        )

    raise RuntimeError(f"Unknown shard extension: {ext}")


def load_tokens_with_hint(filename, data_ext):
    """
    Load tokens with format hint.

    Args:
        filename: Path to token file
        data_ext: Format hint ("auto", "bin", "npy", "pt")

    Returns:
        torch.Tensor of token indices (int64)
    """
    if data_ext in ("auto", "bin"):
        return load_tokens_autodetect(filename)
    if data_ext == "npy":
        arr = np.load(filename)
        return torch.tensor(arr.astype(np.int64), dtype=torch.long)
    if data_ext == "pt":
        t = torch.load(filename)
        return t.long() if torch.is_tensor(t) else torch.tensor(
            np.asarray(t).astype(np.int64), dtype=torch.long
        )
    raise RuntimeError(f"Bad data_ext: {data_ext}")


class FineWebShards:
    """
    Data loader for FineWeb sharded token files.

    Loads pre-tokenized data from sharded files and provides batches.

    Args:
        data_root: Directory containing shard files
        split: Data split ("train" or "val")
        B: Batch size
        T: Sequence length (context length)
        process_rank: Current process rank for distributed training
        num_processes: Total number of processes
        data_ext: File format hint ("auto", "bin", "npy", "pt")
        vocab_size: Expected vocabulary size (for sanity checking)
        sanity_check: Enable sanity checks on token values
    """

    def __init__(self, data_root, split, B, T, process_rank=0, num_processes=1,
                 data_ext="auto", vocab_size=None, sanity_check=False):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.data_ext = data_ext
        self.vocab_size = vocab_size
        self.sanity_check = sanity_check

        # Find shard files
        shards = sorted([
            os.path.join(data_root, s)
            for s in os.listdir(data_root)
            if (split in s and (data_ext == "auto" or s.endswith("." + data_ext)))
        ])
        if data_ext == "auto":
            shards = [
                s for s in shards
                if os.path.splitext(s)[1].lower() in (".bin", ".npy", ".pt", ".pth")
            ]
        if not shards:
            raise FileNotFoundError(
                f"No shards for split={split} under {data_root} (ext={data_ext})"
            )
        self.shards = shards
        self.reset()

    def reset(self):
        """Reset to beginning of data."""
        self.current_shard = 0
        self.tokens = load_tokens_with_hint(self.shards[self.current_shard], self.data_ext)
        if self.sanity_check and self.vocab_size is not None:
            mx = int(self.tokens.max().item())
            if mx >= self.vocab_size:
                raise ValueError(
                    f"[sanity] shard {self.shards[self.current_shard]} has max token {mx} "
                    f">= vocab_size {self.vocab_size}. Wrong file format or vocab."
                )
        self.pos = self.B * self.T * self.process_rank

    def next_batch(self):
        """
        Get next batch of data.

        Returns:
            x: Input tokens (B, T)
            y: Target tokens (B, T)
        """
        B, T = self.B, self.T
        buf = self.tokens[self.pos: self.pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.pos += B * T * self.num_processes

        # Move to next shard if needed
        if self.pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens_with_hint(
                self.shards[self.current_shard], self.data_ext
            )
            if self.sanity_check and self.vocab_size is not None:
                mx = int(self.tokens.max().item())
                if mx >= self.vocab_size:
                    raise ValueError(
                        f"[sanity] shard {self.shards[self.current_shard]} has max token {mx} "
                        f">= vocab_size {self.vocab_size}. Wrong file format or vocab."
                    )
            self.pos = B * T * self.process_rank

        return x, y
