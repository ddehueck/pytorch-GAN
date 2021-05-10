import torch as t
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class GANData:

    def __init__(self, args, root='./data',):
        self.args = args
        self.mnist_dataset = MNISTGANDataset(root=root)
        self.real_loader = DataLoader(
            self.mnist_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

    def sample_latent_space(self, batch_size=None):
        """
        Sample a normal distribution for latent space vectors
        (usually denoted by z)
        :return: a BATCH SIZE x LATENT SIZE tensor
        """
        batch_size = self.args.batch_size if batch_size is None else batch_size
        return t.randn(batch_size, self.args.latent_size)

    def get_fake_labels(self):
        """
        :return: a vector of zeros of length batch size
        """
        return t.zeros(self.args.batch_size, 1)


class MNISTGANDataset(datasets.MNIST):

    def __init__(self, root):
        super(MNISTGANDataset, self).__init__(
            root=root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
           ])
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target indicates that this
            is a real image: 1
        """

        # Replace target with ones
        img, target = super().__getitem__(index)
        return img, t.ones(1)
