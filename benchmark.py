# benchmark.py
# Benchmark pruned/compressed models: Accuracy, Precision, Recall, F1, Latency, Model Size, Peak Memory, FLOPs
# Usage:
#   python3.10 benchmark.py --model_path logs/pruned/efficientnet_b0_pruned_l1_st_30.pth --run_name benchmark_pruned_30pct

import argparse
import json
import os
import time
import torch
from pathlib import Path

import mlflow
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from base_flower import FlowerDataModule

# ==============================================
# Config
# ==============================================
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "benchmark"

# ==============================================
# Helper: FLOPs calculation
# ==============================================
def count_flops(model, input_size=(1, 3, 224, 224)):
    """Calculate FLOPs using fvcore (fallback to thop if unavailable)."""
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
        flops = FlopCountAnalysis(model, dummy_input).total()
        return flops
    except ImportError:
        try:
            from thop import profile
            dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            return flops
        except ImportError:
            print("⚠️  Neither fvcore nor thop installed. Skipping FLOPs.")
            return None

# ==============================================
# Benchmark
# ==============================================
def benchmark(model_path: str, run_name: str, batch_size: int = 128, device: str = "cuda"):
    """
    Benchmark a single model on:
      - Accuracy (overall)
      - Precision, Recall, F1 (macro)
      - Latency (avg, p50, p99)
      - Model Size (params, storage MB)
      - Peak Memory (GB)
      - FLOPs (GFLOPs)
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")    
    # ── 1) Load model ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_path}")
    print(f"{'='*60}")
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    
    # ── 2) Load metadata (if exists) ───────────────────────────────────
    meta_path = Path(model_path).with_suffix(".json")
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"📄 Metadata: {meta_path.name}")
    
    # ── 3) Model Size ──────────────────────────────────────────────────
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    storage_mb = os.path.getsize(model_path) / (1024 ** 2)
    
    print(f"\n📦 Model Size:")
    print(f"   Total params      : {params_total:,}")
    print(f"   Trainable params  : {params_trainable:,}")
    print(f"   Storage size      : {storage_mb:.2f} MB")
    
    # ── 4) FLOPs ───────────────────────────────────────────────────────
    flops = count_flops(model.cpu(), input_size=(1, 3, 224, 224))
    model.to(device)
    gflops = flops / 1e9 if flops else None
    if gflops:
        print(f"   FLOPs             : {gflops:.2f} GFLOPs")
    
    # ── 5) Prepare test data ───────────────────────────────────────────
    datamodule = FlowerDataModule(bs=batch_size)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    
    # ── 6) Warm-up  ────────────────────────────────────
    print(f"\n🔥 Warming up...")
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch[0].to(device)
            _ = model(imgs)
            break
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    # ── 7) Test inference: Accuracy + Precision/Recall/F1 + Latency ────
    print(f"\n⏱️  Running inference on test set...")
    all_preds = []
    all_labels = []
    latencies = []
    
    with torch.no_grad():
        for batch in test_loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
                
            start = time.perf_counter()
            outputs = model(imgs)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            
            latencies.append((time.perf_counter() - start) * 1000)  # ms
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # ── 8) Compute metrics ─────────────────────────────────────────────
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    latency_avg = sum(latencies) / len(latencies)
    latency_p50 = sorted(latencies)[len(latencies) // 2]
    latency_p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    
    print(f"\n✅ Test Metrics:")
    print(f"   Accuracy         : {accuracy:.4f}")
    print(f"   Precision (macro): {precision:.4f}")
    print(f"   Recall (macro)   : {recall:.4f}")
    print(f"   F1 (macro)       : {f1:.4f}")
    print(f"\n⏱️  Latency:")
    print(f"   Avg              : {latency_avg:.2f} ms")
    print(f"   P50              : {latency_p50:.2f} ms")
    print(f"   P99              : {latency_p99:.2f} ms")
    
    # ── 9) Peak Memory ─────────────────────────────────────────────────
    peak_memory_gb = None
    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\n🧠 Peak Memory     : {peak_memory_gb:.3f} GB")
    
    # ── 10) MLflow Logging ─────────────────────────────────────────────
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_path": model_path,
            "batch_size": batch_size,
            "device": str(device),
            **meta  # log metadata from .json
        })
        
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_precision_macro": precision,
            "test_recall_macro": recall,
            "test_f1_macro": f1,
            "latency_avg_ms": latency_avg,
            "latency_p50_ms": latency_p50,
            "latency_p99_ms": latency_p99,
            "params_total": params_total,
            "params_trainable": params_trainable,
            "storage_mb": storage_mb,
        })
        
        if gflops:
            mlflow.log_metric("gflops", gflops)
        if peak_memory_gb:
            mlflow.log_metric("peak_memory_gb", peak_memory_gb)
    
    print(f"\n✅ Results logged to MLflow: {EXPERIMENT_NAME}/{run_name}")
    print(f"{'='*60}\n")

# ==============================================
# Main
# ==============================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark pruned/compressed models on test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .pth model file")
    parser.add_argument("--run_name", type=str, required=True,
                        help="MLflow run name")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference (default: 128)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (default: cuda)")
    args = parser.parse_args()
    
    benchmark(
        model_path=args.model_path,
        run_name=args.run_name,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main()