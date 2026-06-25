# pruning_l1structured.py
# Post-training L1 structured pruning + fine-tune recovery
# Usage:
#   python3.10 pruning_l1structured.py --ckpt logs/checkpoints/checkpoint_base_epoch=15_val_acc=0.9788.ckpt --sparsity 0.3 --finetune_epochs 10

import argparse
import functools
import json
import os
import torch
import torch.nn.utils.prune as prune
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger

from base_flower import FlowerLightModule, FlowerDataModule

TRACKING_URI    = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "pruning"


# ==============================================
# STAGE 1 — Sparsity utilities (same as unstructured)
# ==============================================
from pruning_l1unstructured import get_sparsity, print_sparsity_report


# ==============================================
# STAGE 2 —  L1 Structured Pruning (per-layer)
# ==============================================

def apply_pruning(model: torch.nn.Module, sparsity: float) -> torch.nn.Module:
    """
    Apply per-layer L1 structured pruning to all backbone Conv2d layers.

    - Per-layer: each Conv2d is pruned independently to the same sparsity ratio;
                 PyTorch provides no global_structured helper.
    - L1:        filters are ranked by L1 norm; lowest-norm filters are zeroed.
    - dim=0:     pruning axis is the output channel (filter) dimension.

    IMPORTANT: tensor shapes do NOT change. Zeros are applied via a mask.
               Physical size reduction requires model surgery (rebuilding Conv2d
               with fewer out_channels) -- not implemented here.
    """
    pruned_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and "classifier" not in name:
            n_filters  = module.weight.shape[0]
            n_to_prune = max(1, int(n_filters * sparsity))
            if n_to_prune >= n_filters:
                print(
                    f"  Skipping {name}: only {n_filters} filters, "
                    f"cannot prune {sparsity:.0%}"
                )
                continue

            prune.ln_structured(
                module,
                name="weight",
                amount=sparsity,
                n=1,    # L1 norm
                dim=0,  # prune along output channel dimension
            )
            pruned_count += 1

    print(f"\n  Applied L1 structured pruning to {pruned_count} Conv2d layers")
    print(f"  Target filter sparsity per layer: {sparsity:.1%}")
    return model

# ----------------------------------------------------------------------
# STAGE 3 — Make pruning permanent + quick validation
# ----------------------------------------------------------------------

def make_pruning_permanent(model: torch.nn.Module) -> torch.nn.Module:
    """
    Remove the pruning masks (weight_mask, weight_orig) and write the
    zeroed weights directly into the weight buffer.

    Must be called BEFORE fine-tuning; otherwise the optimizer will
    continue updating the pruned (zero) weights during recovery training.

    Note: tensor shapes remain unchanged after make_permanent.
    Zero filters are permanently baked in but still occupy float32 storage.
    Physical size reduction requires rebuilding Conv2d with fewer out_channels
    (model surgery) -- not implemented here.
    """
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, "weight_mask"):
                prune.remove(module, "weight")

    print("  Pruning masks removed -- zeroed filters written permanently into weight buffers.")
    return model

def quick_validate(model: torch.nn.Module, datamodule) -> float:
    """Simple validation loop — no Lightning overhead. Returns val_acc."""
    datamodule.setup(stage="fit")
    loader  = datamodule.val_dataloader()
    model.eval()
    device  = next(model.parameters()).device
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            imgs   = batch[0].to(device)
            labels = batch[1].to(device)
            preds  = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = correct / total if total else 0.0
    print(f"   [Quick Val] val_acc = {acc:.4f} ({correct}/{total})")
    return acc

# ==============================================
# STAGE 4 — Fine-tune recovery
# ==============================================

def finetune_after_pruning(pl_model, datamodule,
                           sparsity: float, finetune_epochs: int,
                           lr: float, before_stats: dict) -> None:
    """
    Full pipeline:
      load checkpoint -> apply pruning -> make permanent -> fine-tune -> save.
    """
    original_optimizer = pl_model.optimizer
    pl_model.optimizer = lambda params: original_optimizer(params, lr=lr)
    # pl_model.optimizer = functools.partial(pl_model.optimizer, lr=lr)

    run_name = f"pruned_l1structured_{sparsity:.0%}_ft{finetune_epochs}ep"
    logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=TRACKING_URI,
        run_name=run_name,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath="./logs/checkpoints",
        monitor="val_acc",
        filename=(
            f"checkpoint_pruned_l1structured_{sparsity:.0%}"
            + "_{epoch:02d}_{val_acc:.4f}"
        ),
        save_top_k=1,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    trainer = pl.Trainer(
        max_epochs=finetune_epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        enable_model_summary=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    logger.log_hyperparams({
        "pruning_method":           "l1_structured_perlayer",
        "pruning_sparsity":         sparsity,
        "pruned_layers":            "Conv2d only (backbone, classifier excluded)",
        "finetune_epochs":          finetune_epochs,
        "finetune_lr":              lr,
        "sparsity_before_finetune": round(before_stats["global_sparsity"], 4),
    })
    # pl_model.model.training()
    trainer.fit(pl_model, datamodule=datamodule)
    return checkpoint_cb.best_model_path

# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def run(ckpt_path: str,
        sparsity: float,
        finetune_epochs: int,
        lr: float,
        output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ── STAGE 1: Load ──────────────────────────────────────────────────
    print(f"\n[STAGE 1] Loading checkpoint: {ckpt_path}")
    torch.serialization.add_safe_globals([functools.partial])
    pl_model = FlowerLightModule.load_from_checkpoint(ckpt_path)
    params_before = sum(p.numel() for p in pl_model.model.parameters())
    print(f"   params before pruning: {params_before:,}")
    datamodule = FlowerDataModule()
    # print("\n[STAGE 1.1] pre-pruning validation")
    # quick_validate(pl_model.model, datamodule)

    # ── STAGE 2: Apply L1 structured pruning ────────────────────────
    print(f"\n[STAGE 2] Applying L1 structured pruning (sparsity={sparsity:.1%})")
    pl_model.model = apply_pruning(pl_model.model, sparsity)

    before_stats = get_sparsity(pl_model.model)
    print_sparsity_report(before_stats)

    params_after = sum(p.numel() for p in pl_model.model.parameters())
    print(f"   params : {params_before:,} -> {params_after:,}  "
          f"(unstructured: count unchanged, {before_stats['global_sparsity']:.1%} zeroed)")

    # ── STAGE 3: Make permanent + quick validation ─────────────────────
    print("\n[STAGE 3] Making pruning permanent")
    pl_model.model = make_pruning_permanent(pl_model.model)

    after_stats = get_sparsity(pl_model.model)
    print(f"   Sparsity after make_permanent: {after_stats['global_sparsity']:.1%}")
    # print("\n[STAGE 3.1] Post-pruning validation")
    # quick_validate(pl_model.model, datamodule)

    # ── STAGE 4: Fine-tune (optional) ─────────────────────────────────
    if finetune_epochs > 0:
        print(f"\n[STAGE 4] Fine-tuning for {finetune_epochs} epochs (lr={lr})")
        best_ckpt = finetune_after_pruning(pl_model, datamodule, sparsity,
                             finetune_epochs, lr, before_stats)
        print("\n[STAGE 4] Post-finetune validation")
        quick_validate(pl_model.model, datamodule)
        print(f"   Best checkpoint : {best_ckpt}")
    else:
        print("\n[STAGE 4] Skipped (--finetune_epochs=0)")

    # ── STAGE 5: Save ─────────────────────────────────────────────────
    print("\n[STAGE 5] Saving")
    pct        = int(round(sparsity * 100))
    model_path = os.path.join(output_dir, f"efficientnet_b0_pruned_l1_st_{pct}.pth")
    meta_path  = os.path.join(output_dir, f"efficientnet_b0_pruned_l1_st_{pct}.json")

    torch.save(pl_model.model.to("cpu").eval(), model_path)
    with open(meta_path, "w") as f:
        json.dump({
            "base_ckpt":              os.path.abspath(ckpt_path),
            "pruning_method":         "l1_structured_perlayer",
            "sparsity":               sparsity,
            "params_before":          params_before,
            "params_after":           params_after,
            "global_sparsity_actual": round(after_stats["global_sparsity"], 4),
            "finetune_epochs":        finetune_epochs,
            "lr":                     lr,
        }, f, indent=2)

    print(f"   Saved  : {model_path}")
    print(f"   Reload : model = torch.load('{model_path}', weights_only=False)")
    print(f"\n   Next step:")
    print(f"   python3.10 benchmark.py --ckpt {model_path} "
          f"--run_name benchmark_pruned_l1structured_{sparsity:.0%}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-training L1 pruning + fine-tune recovery for EfficientNet-B0 Flower"
    )
    parser.add_argument("--ckpt",            type=str,   required=True,
                        help="Path to base Lightning checkpoint (.ckpt)")
    parser.add_argument("--sparsity",        type=float, default=0.3,
                        help="Per Layer pruning sparsity, e.g. 0.3 = zero out 30%% of Conv weights")
    parser.add_argument("--finetune_epochs", type=int,   default=5,
                        help="Number of epochs to fine-tune after pruning (0 = skip)")
    parser.add_argument("--lr",              type=float, default=1e-4,
                        help="Learning rate for fine-tune recovery (should be << training lr)")
    parser.add_argument("--output_dir",      type=str,   default="./logs/pruned",
                        help="Directory to save pruned model and metadata")
    args = parser.parse_args()

    run(
        ckpt_path       = args.ckpt,
        sparsity        = args.sparsity,
        finetune_epochs = args.finetune_epochs,
        lr              = args.lr,
        output_dir      = args.output_dir,
    )


if __name__ == "__main__":
    main()
