import torch.nn as nn


class Generator(nn.Module):
    """
    Generative Adversarial Network Generator Class

    Takes in a latent vector z and returns a vector in
    the same image space that the discriminator is trained on.

    """

    def __init__(self, img_size, latent_size, hidden_size):
        super(Generator, self).__init__()

        self.latent_size = latent_size

        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, img_size),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Forward pass of a generator

        :param z: Latent space vector - size: batch_size x latent_size
        :return: Tensor of self.img_size
        """
        return self.model(z)

