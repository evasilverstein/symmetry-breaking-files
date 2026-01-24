"""
Utility functions for ECD symmetry breaking experiments.
"""

import torch


def maybe_graph_break():
    """
    Insert a graph break for torch.compile compatibility.

    This is critical when using torch.compile with per-batch bias resampling
    to prevent graph breaks during bias updates.
    """
    try:
        import torch._dynamo as _dynamo
        _dynamo.graph_break()
    except Exception:
        pass


def get_device_type():
    """Get the device type string for torch.autocast."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_serializable(x):
    """Convert a value to a JSON-serializable format."""
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
    """
    Serialize optimizer state to a JSON-serializable dictionary.

    Args:
        opt: Optimizer instance

    Returns:
        Dictionary with optimizer class, defaults, and param_groups
    """
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
    """
    Record optimizer configuration to W&B.

    Args:
        run: W&B run object
        opt_kind: Optimizer type string (e.g., "ecd", "adam")
        opt_obj: Optimizer instance
        ecd_kwargs: ECD-specific kwargs (if opt_kind == "ecd")
        lr_calibrated: Calibrated learning rate (optional)
    """
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
