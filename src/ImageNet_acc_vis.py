import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import re

def parse_one_accuracy(file_path) -> List[float]:
    """Parses a training log file to extract only Val Acc values."""
    val_acc_values = []
    pattern = re.compile(
        r"(?:✅\s*)?Epoch\s*\d+:\s*Train\s+Acc\s*[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?,\s*Val\s+Acc\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)"
    )

    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                val_acc_values.append(float(match.group(1)))
    
    if len(val_acc_values) < 100:
        print(f"Warning: incomplete parsing in {file_path}. Found only {len(val_acc_values)} values.")

    return val_acc_values

def parse_all_accuracies(model2path: Dict) -> Dict:
    """Parses all files in the model2path dictionary to extract accuracies."""
    model2acc = {}
    for model, path in model2path.items():
        model2acc[model] = parse_one_accuracy(path)
    return model2acc

def plot_acc_vis(model2acc: Dict, patch_size: int) -> None:
    """Plots the accuracy visualization for the given model accuracies."""
    for model, accs in model2acc.items():
        plt.plot(accs, label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Top-1 Accuracy")
    plt.title(f"patch_size={patch_size}, batch size = 512, 4x4 GPUs, after 100 epochs")
    plt.legend()
    plt.savefig(f"DRIP_ImageNet_acc_vis_{patch_size}.png")

def main():
    PATCH_SIZE = 16

    # model2path = {
    #     'ViT-B-16': 'ImageNet_ViT16.txt',
    #     "DRIP-16-25%, 2+10": 'ImageNet_DRIP_2_10.txt',
    #     "DRIP-16-25%, 4+8": 'ImageNet_DRIP_4_8.txt'
    # }
    model2path = {
        # 'ViT-B-16, 5e-5': 'ImageNet_ViT16.txt',
        # 'ViT-B-16, 5e-4': 'less_aggressive_ViT.txt',
        # 'ViT-B-16, 3e-3': 'faithful_ImageNet_ViT16.txt',
        'DRIP-16-25%, 4+8, 5e-4': 'less_aggressive_DRIP_4_8_replace.txt',
        "DRIP-16-25%, 2+10, 5e-5": 'ImageNet_DRIP_2_10.txt',
        "DRIP-16-25%, 4+8, 5e-5": 'ImageNet_DRIP_4_8.txt'
    }

    model2acc = parse_all_accuracies(model2path)
    plot_acc_vis(model2acc, patch_size=PATCH_SIZE)


if __name__ == "__main__":
    main()