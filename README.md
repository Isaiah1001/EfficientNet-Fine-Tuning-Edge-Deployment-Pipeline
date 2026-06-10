# EfficientNet Fine-Tuning For Edge Deployment Pipeline

Industrial visual inspection systems require models that are accurate, fast, and resource-efficient for deployment on edge devices. This project develops a deployment-oriented computer vision pipeline that covers data preparation, model selection, fine-tuning a pre-trained EfficientNet on domain-specific data, model compression, and INT8 ONNX deployment. It focuses on building a practical and reproducible workflow that balances accuracy, inference latency, and deployment constraints for real-world applications.

## Goal

This project builds on my previous repository, [CNN Learning Journey](https://github.com/Isaiah1001/CNN-Learning-Journey), which focused on understanding the end-to-end computer vision pipeline and the foundational tools used in deep learning workflows. While that project emphasized learning and experimentation, this repository applies a similar pipeline in a more streamlined and deployment-oriented setting. The focus is placed on data inspection and preparation, model and training strategy selection, compression, and edge deployment planning.

## What This Project Covers

- Data inspection, cleaning, and preparation for training.
- Dataset analysis, including class distribution and split strategy design.
- Fine-tuning a pre-trained EfficientNet for a domain-specific image classification task.
- Evaluation of training performance and model behavior.
- Model compression and export for efficient edge inference.
- INT8-quantized ONNX deployment and latency-oriented benchmarking on edge-oriented hardware.

## Repository Structure

```text
EfficientNet-Fine-Tuning-Edge-Deployment-Pipeline/
├── 01_data/
│   ├── data_analysis.py
│   ├── split_dataset.py
│   └── augmentation_policy.py
├── 02_model/
├── 03_training/
├── 04_compression/
├── 05_deployment
├── requirements.txt
└── README.md
```
