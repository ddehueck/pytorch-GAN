from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


class GANData:

    def __init__(self, root='./data', batch_size=16):
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        self.trainset = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
           ])
        )

        self.trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

    def get_img_size(self):
        img, _ = self.trainset[0]
        return int(np.prod(img.size()))
