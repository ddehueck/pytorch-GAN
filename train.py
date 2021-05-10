import argparse
import torch as t
from trainer import MNISTTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Implementation of a GAN')

    """
    Training Hyperparameters
    """

    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train for (default: 300)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for optimizer (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='number of examples in a batch (default: 32)')
    parser.add_argument('--device', type=int, default=t.device("cuda:0" if t.cuda.is_available() else "cpu"),
                        help='device to train on (default: cuda:0 if cuda is available otherwise cpu)')

    """
    Model Hyperparameters
    """

    parser.add_argument('--latent-size', type=int, default=64,
                        help='size of latent space vectors (default: 64)')
    parser.add_argument('--g-hidden-size', type=int, default=256,
                        help='number of hidden units per layer in G (default: 256)')
    parser.add_argument('--d-hidden-size', type=int, default=256,
                        help='number of hidden units per layer in D (default: 256)')

    # Parse and Train!
    args = parser.parse_args()
    trainer = MNISTTrainer(args)
    trainer.train()
