import argparse
from helper import now, today

def parse_training_arguments():
    parser = argparse.ArgumentParser(description="GAN Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subset-fraction", type=float, default=1., help="fraction of the full data being used (for testing out the code)")
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers in torch.DataLoader for parallel")
    parser.add_argument("--n-epoch", type=int, default=25, help="number of epochs for training")

    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--feature-dim", type=int, default=100, help="feature space (noise) dimension")
    parser.add_argument("--lr", type=float, default=None, help="learning rate (overwrites --lrG and --lrD)")
    parser.add_argument("--time-scale", type=float, default=1.,
                        help=("time scale in learning rate (used with --lr to overwrite --lrD and --lrG), see"
                              "\t- Gans trained by a two time-scale update rule converge to a local nash equilibrium. "
                              "\t  M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter - 2017\n"
                              "\t- https://sthalles.github.io/advanced_gans/ suggests use 4."))
    parser.add_argument("--lrG", type=float, default=1e-4, help="generator learning rate")
    parser.add_argument("--lrD", type=float, default=1e-4, help="discriminator learning rate")
    parser.add_argument("--net-size", type=int, default=64, help="network size")

    parser.add_argument("--G-batch", action="store_true",
                        help="Batch normalization in training G")
    parser.add_argument("--D-batch", action="store_true",
                        help="Batch normalization in training D")

    parser.add_argument("--alpha-region", type=float, default=1.0, help="weight for the region (C, T, L) classification")

    # ignores when log
    parser.add_argument("--n-step-eval", type=int, default=10, help="number of epochs between evaluation")
    parser.add_argument("--eval-size", type=int, default=2000, help="number of test images generated for evaluation")
    parser.add_argument("--exp", type=str, default=today(), help="experiment suffix")
    parser.add_argument("--run-name", type=str, default=now(), help="run name (default to be a time tag)")
    parser.add_argument("--continued", action="store_true",
                        help=("continued from checkpoint:\n"
                              "automatically activated if --checkpoint provided; \n"
                              "use the default checkpoint (EXP/RUN_NAME.ckpt) if no --checkpoint provided"))
    parser.add_argument("--checkpoint", type=str, default=None, help="the checkpoint to be loaded")
    parser.add_argument("--no-pbar", action="store_true", help="no progress bar")
    parser.add_argument("--diter", type=int, default=1, help="number of iter of discriminator for 1 iter of gene")
    parser.add_argument("--data-augm", action="store_true", help="Run data small data augmentation")

    return parser
