import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import re

def parse_one_accuracy(file_path, newcodebase: bool) -> List[float]:
    """Parses a training log file to extract only Val Acc values."""
    val_acc_values = []
    if newcodebase:
        pattern = re.compile(
            r"Test:\s+Acc@1\s+([0-9]*\.?[0-9]+)\s+Acc@5\s+([0-9]*\.?[0-9]+)"
        )
    else:
        pattern = re.compile(
            r"(?:✅\s*)?Epoch\s*\d+:\s*Train\s+Acc\s*[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?,\s*Val\s+Acc\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)"
        )

    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                if newcodebase:
                    val_acc_values.append(float(match.group(1))/100)
                else:
                    val_acc_values.append(float(match.group(1)))
    
    # sanity check: exactly 300 epochs
    assert len(val_acc_values) == 300, f"{file_path} Expected 300 epochs, got {len(val_acc_values)}"
    return val_acc_values

def parse_all_accuracies(model2path: Dict) -> Dict:
    """Parses all files in the model2path dictionary to extract accuracies."""
    model2acc = {}
    for model, path in model2path.items():
        model2acc[model] = parse_one_accuracy(path, newcodebase=True)
    return model2acc

def plot_acc_vis(model2acc: Dict, patch_size: int) -> None:
    """Plots the accuracy visualization for the given model accuracies."""
    for model, accs in model2acc.items():
        plt.plot(accs, label=model, linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Top-1 Accuracy")
    plt.title("ImageNet Top-1 Test Accuracy Over Epochs")
    
    # Place legend outside below plot
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    plt.tight_layout()
    plt.savefig(f"FINAL_ImageNet_acc_vis.png", bbox_inches="tight")
    plt.close()


def main():
    PATCH_SIZE = 16
    model2path = {
        #'ViT-B-16': 'AUG_27_Vit_0.0003_imagenet.txt', # baseline
        'DRIP-4X-16, 5+7': 'SEP_1_DRIP_ViTbased_4x_5_7.txt', # check
        'DRIP-4X-16, 4+8': 'AUG_29_DRIP_ViTbased_4x_4_8.txt', # check
        'DRIP-4X-16, 2+10': 'AUG_29_DRIP_ViTbased_4x_2_10.txt', # check
        'DRIP-10X-16, 4+8': 'AUG_29_DRIP_ViTbased_10x_4_8.txt', # check
        'DRIP-10X-16, 5+7': 'SEP_1_DRIP_ViTbased_10x_5_7.txt', # check
        #'Transformer-XL-16': 'AUG_29_DRIP_XL_baseline.txt', # XL baseline
        'DRIP-2X-16*, 4+8': 'AUG_23_DRIP_4x_4_8_0.0003.txt', # check, start here
        'DRIP-4X-16*, 4+8': 'AUG_29_DRIP_XLbased_4x_4_8.txt', # check
        'DRIP-4X-16*, 2+10': 'SEP_1_DRIP_XLbased_4x_2_10.txt', # check
        'DRIP-10X-16*, 4+8': 'AUG_29_DRIP_ViTbased_10x_4_8.txt', # check
        'DRIP-10X-16*, 5+7': 'SEP_1_DRIP_XLbased_10x_5_7.txt' # check
    }

    model2acc = parse_all_accuracies(model2path)
    plot_acc_vis(model2acc, patch_size=PATCH_SIZE)


if __name__ == "__main__":
    main()