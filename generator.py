import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generative Adverserial Network Generator Class

    Takes in a latent vector z and returns a vector in
    the same image space that the discriminator is trained on.
    """

    def __init__(self, img_size, latent_size):
        super(Generator, self).__init__()
        # Hyperparameters
        self.img_size = img_size
        self.latent_size = latent_size
        # Layers
        self.l1 = nn.Linear(latent_size, 64)
        self.l2 = nn.Linear(64, 256)
        self.l3 = nn.Linear(256, 512)
        self.l4 = nn.Linear(512, img_size)

    def forward(self, z):
        """
        Forward pass of a generator

        :param z: Latent space vector - size: batch_size x latent_size
        :return: Tensor of self.img_size
        """
        z = z.view(-1 , self.latent_size)
        # Pass through layers with a non-linearity
        z = F.relu(self.l1(z))
        z = F.relu(self.l2(z))
        z = F.relu(self.l3(z))

        # Outputs a vector in the same space as the real data
        return F.relu(self.l4(z))

    def sample_z(self, batch_size):
        return t.randn(batch_size, self.latent_size)
