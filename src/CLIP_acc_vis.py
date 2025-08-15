import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import re


def parse_one_accuracy(file_path) -> List[float]:
    """Parses a single file to extract the top1 accuracy values."""
    top1_values = []
    pattern = re.compile(r"Eval Epoch:\s*\d+.*?imagenet-zeroshot-val-top1:\s*([0-9.]+)")

    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                top1_values.append(float(match.group(1)))
    
    if len(top1_values) < 10:
        print(f"Warning: incomplete parsing in {file_path}. Found only {len(top1_values)} values.")

    return top1_values

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
    plt.title(f"26M samples, patch_size={patch_size}, lr=5e-5, after 10 epochs")
    plt.legend()
    plt.savefig(f"NEW_acc_vis_{patch_size}.png")

def main():

    PATCH_SIZE = 16

    if PATCH_SIZE == 32:
        model2path = {
            'ViT-B-32': 'new_vit32.log',
            "DRIP-32-50%, 2+10": 'new_DRIP_2x.log',
            "DRIP-32-25%, 5+7": 'new_DRIP_debug_4x_5_7.log',
            "DRIP-32-25%, 4+8": 'new_DRIP_debug_4x_4_8.log',
            "DRIP-32-25%, 2+10": 'new_DRIP_debug_4x.log',
            "H-DRIP-32-50%-50%, 3+3+6": 'H-DRIP.log',
            "S-DRIP-32-40%-60%, 2+10": 'S-DRIP.log',
        }
    elif PATCH_SIZE == 16:
        # model2path = {
        #     'ViT-B-16': 'new_vit16.log',
        #     "DRIP-16-50%, 2+10": 'new_DRIP_16p_2x.log',
        #     "DRIP-16-25%, 5+7": 'new_DRIP_16p_4x_5_7.log',
        #     "DRIP-16-25%, 4+8": 'new_DRIP_16p_4x_4_8.log',
        #     "DRIP-16-25%, 2+10": 'new_DRIP_16p_4x.log',
        #     "H-DRIP-16-50%-50%, 3+3+6": 'H-DRIP-16.log',
        #     "S-DRIP-16-40%-60%, 2+10": 'S-DRIP-16.log',
        # }
        # model2path = {
        #     'ViT-B-16': 'Aug6_ViT16.log',
        #     "DRIP-16-25%, 2+10": 'Aug6_DRIP_25_2_10.log',
        #     "DRIP-16-25%, 4+8": 'Aug6_DRIP_25_4_8.log',
        #     "H-DRIP-16-50%-50%, 3+3+6": 'Aug6_H-DRIP.log',
        #     "S-DRIP-16-20%-30%, 2+10": 'Aug6_S-DRIP.log',
        # }
        model2path = {
            'ViT-B-16': 'Aug13_ViT16.log',
            "DRIP-16-25%, 2+10": 'Aug13_DRIP_2_10.log',
            "DRIP-16-25%, 4+8": 'Aug13_DRIP_4_8.log'
        }


    else:
        raise ValueError("Unsupported patch size. Only 16 and 32 are supported.")


    model2acc = parse_all_accuracies(model2path)
    plot_acc_vis(model2acc, patch_size=PATCH_SIZE)



if __name__ == "__main__":
    main()