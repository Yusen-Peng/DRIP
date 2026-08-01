import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def setup_plot_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.1,
        "lines.linewidth": 2.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def load_trainer_state(path):
    path = Path(path)
    with open(path, "r") as f:
        state = json.load(f)

    rows = []
    for item in state["log_history"]:
        if "step" not in item:
            continue
        rows.append({
            "step": item.get("step"),
            "epoch": item.get("epoch"),
            "lm_loss": item.get("loss"),
            "boundary_loss": item.get("latest_boundary_loss"),
            "learning_rate": item.get("learning_rate"),
            "grad_norm": item.get("grad_norm"),
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["step"])
    return df.sort_values("step")


def smooth(series, window=15):
    return series.rolling(window=window, min_periods=1, center=True).mean()

def binomial_boundary_loss_floor(L, prior):
    import math
    k = math.floor((L + 1) * prior)
    log_prob = (
        math.lgamma(L + 1)
        - math.lgamma(k + 1)
        - math.lgamma(L - k + 1)
        + k * math.log(prior)
        + (L - k) * math.log(1 - prior)
    )
    return -log_prob / L




def plot_training_curves(trainer_state_path, output_path, smooth_window=15):
    setup_plot_style()
    df = load_trainer_state(trainer_state_path)
    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))

    ax1.plot(
        df["step"],
        smooth(df["lm_loss"], smooth_window),
        label="Language modeling loss",
        color="tab:blue"
    )
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("LM loss")
    ax1.grid(True, linewidth=0.6, alpha=0.35)
    ax2 = ax1.twinx()

    ax2.plot(
        df["step"],
        smooth(df["boundary_loss"], smooth_window),
        label="Boundary loss",
        color="tab:orange"
    )

    ax2.set_ylabel("Boundary loss")
    
    floor_4x = binomial_boundary_loss_floor(L=576, prior=0.25)
    ax2.axhline(
        floor_4x,
        color="gray",
        linestyle=":",
        linewidth=2.0,
        label="Boundary loss floor",
    )
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()


    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        frameon=False,
        loc="upper right",
    )

    # ax1.set_title("Training Dynamics of DRIP")
    plt.savefig(output_path)
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=300)
    plt.close()



if __name__ == "__main__":

    TRAINER_JSON_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train_full/trainer_state.json"
    # TRAINER_JSON_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain_second_to_last/trainer_state.json"


    if "finetune" in TRAINER_JSON_PATH:
        OUTPUT_PATH = f"src/curves/finetune_training_curves.pdf"
    elif "pretrain" in TRAINER_JSON_PATH:
        OUTPUT_PATH = f"src/curves/pretrain_training_curves.pdf"
    else:
        raise ValueError(f"Unknown training type in path: {TRAINER_JSON_PATH}")

    plot_training_curves(
        trainer_state_path=TRAINER_JSON_PATH,
        output_path=OUTPUT_PATH,
        smooth_window=15,
    )
