# save_base_model.py
# 从 .ckpt converted into .pth
# Usage: python3.10 save_base_model.py --ckpt logs/checkpoints/checkpoint_base_epoch=29_val_acc=0.9756.ckpt

import argparse
import os
import torch
from base_flower import FlowerLightModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./logs/base_models")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # load ckpt
    pl_module = FlowerLightModule.load_from_checkpoint(args.ckpt)
    
    # retrive val_acc
    ckpt = torch.load(args.ckpt, weights_only=False)
    epoch = ckpt.get("epoch", 0)
    
    val_acc = None
    if "callbacks" in ckpt:
        for cb in ckpt["callbacks"].values():
            if "best_model_score" in cb:
                val_acc = cb["best_model_score"].item()
                break
    
    # save
    name = f"base_epoch={epoch}_val_acc={val_acc:.4f}"
    pth_path = os.path.join(args.output_dir, f"{name}.pth")
    
    torch.save(pl_module.model.to("cpu").eval(), pth_path)
    print(f"✅ Saved: {pth_path}")

if __name__ == "__main__":
    main()