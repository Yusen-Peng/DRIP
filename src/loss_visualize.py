import re
import sys
import matplotlib.pyplot as plt


def parse_losses(log_path):
    pattern = re.compile(
        r"cls_loss:\s*([0-9]*\.?[0-9]+)\s*,\s*boundary_loss:\s*([0-9]*\.?[0-9]+)"
    )

    cls_losses = []
    boundary_losses = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # remove weird ANSI stuff from tqdm
            line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)

            match = pattern.search(line)
            if match:
                cls_losses.append(float(match.group(1)))
                boundary_losses.append(float(match.group(2)))

    return cls_losses, boundary_losses


def plot_losses(cls_losses, boundary_losses, log_file: str):
    steps = list(range(len(cls_losses)))

    plt.figure(figsize=(12, 5))
    # plt.plot(steps, cls_losses, label="cls_loss")
    plt.plot(steps, boundary_losses, label="boundary_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_file.replace(".txt", ".png"))

if __name__ == "__main__":
    log_file = sys.argv[1]

    cls_losses, boundary_losses = parse_losses(log_file)

    print(f"Parsed {len(cls_losses)} points")

    if len(cls_losses) == 0:
        print("No matches found 😭")
    else:
        plot_losses(cls_losses, boundary_losses, log_file)