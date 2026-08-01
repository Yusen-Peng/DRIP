import torch
import argparse
from open_clip_local.BP import BoundaryPredictor


def main(args):
    torch.manual_seed(0)

    B = 1
    T = 196
    D = 768

    model = BoundaryPredictor(
        d_model=D,
        d_inner=int(D * 4),
        activation_function="gelu",
        temp=0.1,
        prior=0.25,
        bp_type='gumbel',
        threshold=0.5,
        smart_init=args.smart_init
    )

    # fake input: [seq_len, batch_size, d_model]
    x = torch.randn(T, B, D)

    soft, hard = model.inference(x)

    print("\n==== STATS ====")
    print("soft mean:", soft.mean().item())
    print("hard mean (fraction of 1s):", hard.mean().item())

    print("\n==== SAMPLE OUTPUT ====")
    print("soft:\n", soft)
    print("hard:\n", hard)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smart-init", action="store_true")
    args = parser.parse_args()

    main(args)