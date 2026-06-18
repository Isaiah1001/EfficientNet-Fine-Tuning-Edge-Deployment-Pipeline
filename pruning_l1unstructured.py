# pruning_l1unstructured.py
# Post-training L1 unstructured pruning + fine-tune recovery
#
# Usage:
#   python3.10 pruning_l1unstructured.py --ckpt logs/checkpoints/checkpoint_base_epoch=15_val_acc=0.9788.ckpt --sparsity 0.3 --finetune_epochs 10
#
# Pipeline:
#   STAGE 1 — load trained Lightning checkpoint
#   STAGE 2 — apply global L1 unstructured pruning (Conv2d backbone only)
#   STAGE 3 — make pruning permanent + quick validation
#   STAGE 4 — optional recovery fine-tune via Lightning Trainer
#   STAGE 5 — save pruned model (.pth) + metadata (.json)
#
# Note: L1 unstructured pruning zeros individual weights but does NOT
#       reduce tensor shapes, so MACs on dense hardware are unchanged.
#       Combine with quantization for real size/speed reduction.

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

# ----------------------------------------------------------------------
# STAGE 1 — Sparsity utilities
# ----------------------------------------------------------------------

def get_sparsity(model: torch.nn.Module) -> dict:
    """Compute the fraction of zero weights in Conv2d and Linear layers."""
    total_weights = 0
    zero_weights  = 0
    layer_stats   = {}

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            w       = module.weight.data
            n_total = w.numel()
            n_zero  = (w == 0).sum().item()
            total_weights += n_total
            zero_weights  += n_zero
            layer_stats[name] = {
                "total":    n_total,
                "zeros":    n_zero,
                "sparsity": n_zero / n_total,
            }

    global_sparsity = zero_weights / total_weights if total_weights > 0 else 0.0
    return {"global_sparsity": global_sparsity, "layers": layer_stats}


def print_sparsity_report(stats: dict) -> None:
    print(f"   Global sparsity : {stats['global_sparsity']:.1%}")
    print(f"   Top sparse layers:")
    sorted_layers = sorted(
        [(n, s) for n, s in stats["layers"].items() if s["sparsity"] > 0],
        key=lambda x: x[1]["sparsity"], reverse=True,
    )
    for name, s in sorted_layers[:10]:
        print(f"   {name:<50} {s['sparsity']:.1%} ({s['zeros']:,}/{s['total']:,})")

# ----------------------------------------------------------------------
# STAGE 2 — L1 unstructured pruning
# ----------------------------------------------------------------------

def apply_pruning(model: torch.nn.Module, sparsity: float) -> torch.nn.Module:
    """Apply global L1 unstructured pruning across all Conv2d backbone layers."""
    parameters_to_prune = [
        (module, "weight")
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Conv2d)
        and "classifier" not in name
    ]
    print(f"   Pruning {len(parameters_to_prune)} Conv2d layers (classifier excluded)")
    print(f"   Target sparsity : {sparsity:.1%}")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    return model

# ----------------------------------------------------------------------
# STAGE 3 — Make pruning permanent + quick validation
# ----------------------------------------------------------------------

def make_pruning_permanent(model: torch.nn.Module) -> torch.nn.Module:
    """Remove pruning masks and write zeroed weights permanently into buffers."""
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, "weight_mask"):
                prune.remove(module, "weight")
    print("   Pruning masks removed — zero weights written into weight buffers.")
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

# ----------------------------------------------------------------------
# STAGE 4 — Fine-tune recovery
# ----------------------------------------------------------------------

def finetune(pl_model, datamodule, sparsity: float,
             finetune_epochs: int, lr: float,
             before_stats: dict) -> str:
    """
    Full pipeline:
      load checkpoint -> apply pruning -> make permanent -> fine-tune -> save.
    """
    original_optimizer = pl_model.optimizer
    pl_model.optimizer = lambda params: original_optimizer(params, lr=lr)

    run_name = f"pruned_l1unstructured_{sparsity:.0%}_ft{finetune_epochs}ep"

    logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=TRACKING_URI,
        run_name=run_name,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath="./logs/checkpoints",
        monitor="val_acc",
        filename=(
            f"checkpoint_pruned_l1unstructured_{sparsity:.0%}"
            + "_{epoch:02d}_{val_acc:.4f}"
        ),
        save_top_k=1,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
     
    trainer = pl.Trainer(
        max_epochs=finetune_epochs,
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        enable_model_summary=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    logger.log_hyperparams({
        "pruning_method":           "l1_unstructured_global",
        "pruning_sparsity":         sparsity,
        "pruned_layers":            "Conv2d only (backbone, classifier excluded)",
        "finetune_epochs":          finetune_epochs,
        "finetune_lr":              lr,
        "sparsity_before_finetune": round(before_stats["global_sparsity"], 4),
    })
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
    pl_model   = FlowerLightModule.load_from_checkpoint(ckpt_path)
    params_before = sum(p.numel() for p in pl_model.model.parameters())
    print(f"   params before pruning: {params_before:,}")
    datamodule = FlowerDataModule()
    # print("\n[STAGE 1.1] pre-pruning validation")
    # quick_validate(pl_model.model, datamodule)

    # ── STAGE 2: Apply L1 unstructured pruning ────────────────────────
    print(f"\n[STAGE 2] Applying L1 unstructured pruning (sparsity={sparsity:.1%})")
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
        best_ckpt = finetune(pl_model, datamodule, sparsity,
                             finetune_epochs, lr, before_stats)
        print("\n[STAGE 4] Post-finetune validation")
        quick_validate(pl_model.model, datamodule)
        print(f"   Best checkpoint : {best_ckpt}")
    else:
        print("\n[STAGE 4] Skipped (--finetune_epochs=0)")

    # ── STAGE 5: Save ─────────────────────────────────────────────────
    print("\n[STAGE 5] Saving")
    pct        = int(round(sparsity * 100))
    model_path = os.path.join(output_dir, f"efficientnet_b0_pruned_l1_unst_{pct}.pth")
    meta_path  = os.path.join(output_dir, f"efficientnet_b0_pruned_l1_unst_{pct}.json")

    torch.save(pl_model.model.to("cpu").eval(), model_path)
    with open(meta_path, "w") as f:
        json.dump({
            "base_ckpt":              os.path.abspath(ckpt_path),
            "pruning_method":         "l1_unstructured_global",
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
          f"--run_name benchmark_pruned_l1unstructured_{sparsity:.0%}")

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
                        help="Global pruning sparsity, e.g. 0.3 = zero out 30%% of Conv weights")
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
