import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Generative Adverserial Network Discriminator Class

    Takes in an image as input and outputs a probability indicating whether or not
    the input belongs to a the real data distribution.
    """

    def __init__(self, img_size):
        super(Discriminator, self).__init__()

        self.img_size = img_size
        self.l1 = nn.Linear(img_size, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass of a discriminator

        :param x: Image tensor
        :return: Float in range [0, 1] - probability score
        """

        # Resize x into a vector
        x = x.view(-1, self.img_size)

        # Pass through layers with a non-linearity
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        # Use sigmoid to convert to a probability
        return t.sigmoid(self.l4(x))
