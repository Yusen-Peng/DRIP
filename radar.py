# radar_plot_zscore_pretty.py
# Usage: python radar_plot_zscore_pretty.py results.csv
#
# Same logic as your script; only appearance is improved.

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from pathlib import Path

OUTPUT_STEM = "radar_models"

def close_loop(arr):
    return np.concatenate([arr, arr[:1]])

def main(csv_path: Path):
    df = pd.read_csv(csv_path)
    assert "Model" in df.columns, "CSV must include a 'Model' column"
    metrics = [c for c in df.columns if c != "Model"]

    # --- your original (kept): no z-score here, just pass-through values ---
    df_z = df.copy()
    for m in metrics:
        vals = df[m].astype(float)
        df_z[m] = vals * 1.0  # convert to percentage / keep as-is

    # --- styling: global rcParams + vibrant palette ---
    plt.rcParams.update({
        "font.size": 13,
        "font.weight": "bold",         
        "axes.linewidth": 3.0,
        "grid.linewidth": 3.0,
        "grid.alpha": 1.0,
    })
    palette = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488",
               "#F39B7F", "#91D1C2", "#7E6148", "#B09C85"]
    plt.rcParams["axes.prop_cycle"] = cycler(color=palette)

    # --- geometry ---
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles_closed = close_loop(angles)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_thetagrids(np.degrees(angles), metrics)

    # make metric labels tangent to the circle (curved look)
    xticks = ax.get_xticks()
    for ang, lbl in zip(xticks, ax.get_xticklabels()):
        rot = np.degrees(ang) - 90
        if rot < -90:
            rot += 180
        lbl.set_rotation(rot)
        lbl.set_rotation_mode("anchor")
        lbl.set_va("center")
        lbl.set_ha("center")

    # minimal rings + dashed reference at 0
    ax.set_yticks([-1, 0, 1])     # keep just a few rounds
    ax.set_yticklabels([])        # number-free
    ax.yaxis.grid(True, linestyle=(0, (3, 3)), linewidth=2, alpha=0.6)
    theta = np.linspace(0, 2*np.pi, 512)
    ax.plot(theta, np.full_like(theta, 0), linestyle="--", linewidth=2.0, alpha=0.7)

    # soften frame and tighten layout area
    ax.spines["polar"].set_visible(False)
    ax.set_position([0.05, 0.05, 0.90, 0.90])

    # draw model outlines (no fill)
    handles, labels = [], []
    for _, row in df_z.iterrows():
        vals = row[metrics].astype(float).to_numpy()
        v_closed = close_loop(vals)
        line, = ax.plot(angles_closed, v_closed, linewidth=3)
        handles.append(line)
        labels.append(row["Model"])

    # legend polish
    leg = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.03),
                    ncol=min(3, len(labels)), frameon=False, handlelength=2.8, handletextpad=0.6)

    # save (tight crop)
    plt.savefig(f"{OUTPUT_STEM}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ Saved: {OUTPUT_STEM}.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python radar_plot_zscore_pretty.py <results.csv>")
        sys.exit(1)
    main(Path(sys.argv[1]))