# benchmark_onnx.py
# Benchmark ONNX models (FP32 + INT8) on CPU via ONNX Runtime
# Metrics: Accuracy, Precision, Recall, F1, Latency (avg/p50/p99),
#          Model Size (MB), FLOPs (GFLOPs), Peak Memory (RSS GB)
#
# Usage:
#   python3.10 benchmark_onnx.py --model_path flower_efficientnet_basemodel.onnx --run_name basemodel
#
#   python3.10 benchmark_onnx.py --model_path flower_efficientnet_quantization.onnx --run_name quantization

import argparse
import json
import os
import time
import tracemalloc
from pathlib import Path

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from base_flower import FlowerDataModule

# ==============================================
# Config
# ==============================================
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "benchmark_quantization"

# ==============================================
# Helper: FLOPs via onnx-opcounter
# ==============================================
def count_flops_onnx(model_path: str):
    """
    Estimate FLOPs (MACs) from an ONNX graph.
    Requires: pip install onnx-opcounter
    Falls back gracefully if not installed.
    """
    try:
        from onnx_opcounter import calculate_macs
        model = onnx.load(model_path)
        macs = calculate_macs(model)
        return int(macs)
    except ImportError:
        print("⚠️  onnx-opcounter not installed. Skipping FLOPs.")
        print("    Install with: pip install onnx-opcounter")
        return None
    except Exception as e:
        print(f"⚠️  FLOPs calculation failed: {e}")
        return None

# ==============================================
# Benchmark
# ==============================================
def benchmark(model_path: str, run_name: str, batch_size: int = 1,
              intra_op_threads: int = 0, inter_op_threads: int = 0,
              warmup_runs: int = 10):
    """
    Benchmark a single ONNX model on CPU via ONNX Runtime.

    Metrics:
    - Accuracy (overall)
    - Precision, Recall, F1 (macro)
    - Latency (avg, p50, p99) per batch [ms]
    - Model Size (storage MB)
    - Peak Memory (RSS GB via tracemalloc)
    - FLOPs (GFLOPs via onnx-opcounter)
    """

    print(f"\n{'='*60}")
    print(f"Benchmarking (ONNX / CPU): {model_path}")
    print(f"{'='*60}")

    # ── 1) Session setup ───────────────────────────────────────────
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = intra_op_threads   # 0 = ORT default
    sess_options.inter_op_num_threads = inter_op_threads   # 0 = ORT default
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    input_name  = session.get_inputs()[0].name    # "input"
    output_name = session.get_outputs()[0].name   # "logits"

    print(f"  ORT version     : {ort.__version__}")
    print(f"  Input  name     : {input_name}  shape: {session.get_inputs()[0].shape}")
    print(f"  Output name     : {output_name} shape: {session.get_outputs()[0].shape}")
    print(f"  intra_op_threads: {intra_op_threads} (0=ORT default)")
    print(f"  inter_op_threads: {inter_op_threads} (0=ORT default)")

    # ── 2) Load metadata (.json sidecar, same convention as benchmark.py) ─
    meta_path = Path(model_path).with_suffix(".json")
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"📄 Metadata: {meta_path.name}")

    # ── 3) Model Size ──────────────────────────────────────────────
    storage_mb = os.path.getsize(model_path) / (1024 ** 2)
    print(f"\n📦 Model Size:")
    print(f"   Storage size : {storage_mb:.2f} MB")

    # ── 4) FLOPs ───────────────────────────────────────────────────
    flops = count_flops_onnx(model_path)
    mflops = flops / 1e6 if flops else None
    if mflops:
        print(f"   FLOPs        : {mflops:.2f} MFLOPs")

    # ── 5) Prepare test data ───────────────────────────────────────
    datamodule = FlowerDataModule(bs=batch_size)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    # ── 6) Warm-up (CPU — no CUDA sync needed) ────────────────────
    print(f"\n🔥 Warming up ({warmup_runs} runs)...")
    for batch in test_loader:
        imgs_np = batch[0].numpy().astype(np.float32)
        for _ in range(warmup_runs):
            _ = session.run([output_name], {input_name: imgs_np})
        break

    # ── 7) Inference: Accuracy + Precision/Recall/F1 + Latency ────
    print(f"\n⏱️  Running inference on test set...")
    all_preds  = []
    all_labels = []
    latencies  = []

    tracemalloc.start()

    for batch in test_loader:
        imgs_np = batch[0].numpy().astype(np.float32)
        labels  = batch[1].numpy()

        start   = time.perf_counter()
        outputs = session.run([output_name], {input_name: imgs_np})
        latencies.append((time.perf_counter() - start) * 1000)  # ms

        preds = np.argmax(outputs[0], axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_mem / (1024 ** 2)

    # ── 8) Compute metrics ─────────────────────────────────────────
    accuracy  = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    latency_avg = sum(latencies) / len(latencies)
    latency_p50 = sorted(latencies)[len(latencies) // 2]
    latency_p99 = sorted(latencies)[int(len(latencies) * 0.99)]

    print(f"\n✅ Test Metrics:")
    print(f"   Accuracy         : {accuracy:.4f}")
    print(f"   Precision (macro): {precision:.4f}")
    print(f"   Recall (macro)   : {recall:.4f}")
    print(f"   F1 (macro)       : {f1:.4f}")
    print(f"\n⏱️  Latency (per batch, batch_size={batch_size}):")
    print(f"   Avg : {latency_avg:.2f} ms")
    print(f"   P50 : {latency_p50:.2f} ms")
    print(f"   P99 : {latency_p99:.2f} ms")
    print(f"\n🧠 Peak Memory (RSS): {peak_memory_mb:.4f} MB")

    # ── 9) MLflow Logging ──────────────────────────────────────────
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_path"       : model_path,
            "runtime"          : "onnxruntime",
            "provider"         : "CPUExecutionProvider",
            "batch_size"       : batch_size,
            "intra_op_threads" : intra_op_threads,
            "inter_op_threads" : inter_op_threads,
            "warmup_runs"      : warmup_runs,
            "ort_version"      : ort.__version__,
            **meta,
        })

        metrics = {
            "test_accuracy"       : accuracy,
            "test_precision_macro": precision,
            "test_recall_macro"   : recall,
            "test_f1_macro"       : f1,
            "latency_avg_ms"      : latency_avg,
            "latency_p50_ms"      : latency_p50,
            "latency_p99_ms"      : latency_p99,
            "storage_mb"          : storage_mb,
            "peak_memory_mb"      : peak_memory_mb,
        }
        if mflops:
            metrics["mflops"] = mflops

        mlflow.log_metrics(metrics)

    print(f"\n✅ Results logged to MLflow: {EXPERIMENT_NAME}/{run_name}")
    print(f"{'='*60}\n")


# ==============================================
# Main
# ==============================================
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX models (FP32 / INT8) on CPU via ONNX Runtime"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to .onnx file (FP32 or INT8)"
    )
    parser.add_argument(
        "--run_name", type=str, required=True,
        help="MLflow run name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for inference (default: 1)"
    )
    parser.add_argument(
        "--intra_op_threads", type=int, default=0,
        help="ORT intra-op parallelism threads (0 = ORT default, typically = CPU cores)"
    )
    parser.add_argument(
        "--inter_op_threads", type=int, default=0,
        help="ORT inter-op parallelism threads (0 = ORT default)"
    )
    parser.add_argument(
        "--warmup_runs", type=int, default=10,
        help="Number of warm-up inference runs before measurement (default: 10)"
    )
    args = parser.parse_args()

    benchmark(
        model_path       = args.model_path,
        run_name         = args.run_name,
        batch_size       = args.batch_size,
        intra_op_threads = args.intra_op_threads,
        inter_op_threads = args.inter_op_threads,
        warmup_runs      = args.warmup_runs,
    )


if __name__ == "__main__":
    main()
