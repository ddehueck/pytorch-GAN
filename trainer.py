import torch as t
import torch.nn as nn
import torch.optim as optim
from mnist_data import GANData
from models.discriminator import Discriminator
from models.generator import Generator
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision


class MNISTTrainer:

    def __init__(self, args):
        self.args = args
        # To write to Tensorboard
        self.writer = SummaryWriter()

        # Holds all data classes needed
        self.data = GANData(args, root='./data')

        # Instantiate models and load to device
        self.D = Discriminator(784, args.d_hidden_size).to(args.device)
        self.G = Generator(784, args.latent_size, args.g_hidden_size).to(args.device)

        # Instantiate criterion used for both D and G
        self.criterion = nn.BCELoss()

        # Instantiate an optimizer for both D and G
        self.d_optim = optim.Adam(self.D.parameters(), lr=args.lr)
        self.g_optim = optim.Adam(self.G.parameters(), lr=args.lr)

    def train(self):
        """
        Main training loop for this trainer. To be called in train.py.
        """
        device = self.args.device

        print(f'Training on device: {device}')
        print(f'Beginning training for {self.args.epochs} epochs...')

        for epoch in range(self.args.epochs):

            running_d_loss, running_g_loss = 0.0, 0.0

            for real_imgs, real_labels in tqdm(self.data.real_loader):
                # Load MNIST images and labels to device
                real_imgs, real_labels = real_imgs.to(device), real_labels.to(device)
                # Load latent vectors and labels to device
                z, fake_labels = self.data.sample_latent_space().to(device), self.data.get_fake_labels().to(device)

                #####################################
                #       Update Discriminator        #
                #####################################

                # Get probability scores for real and fake data
                real_logits = self.D(real_imgs)

                fake_imgs = self.G(z)
                fake_logits = self.D(fake_imgs)

                d_real_loss = self.criterion(real_logits, real_labels)
                d_fake_loss = self.criterion(fake_logits, fake_labels)
                d_loss = d_real_loss + d_fake_loss

                # # Backpropagation and update
                self.d_optim.zero_grad()
                d_loss.backward()
                self.d_optim.step()

                #####################################
                #       Update Generator            #
                #####################################

                # Load another batch of latent vectors device
                z = self.data.sample_latent_space(batch_size=len(real_imgs)).to(device)

                # Get generated images and and record loss
                fake_imgs = self.G(z)
                fake_logits = self.D(fake_imgs)
                g_loss = self.criterion(fake_logits, real_labels)

                # Backpropagation and update
                self.g_optim.zero_grad()
                g_loss.backward()
                self.g_optim.step()

                # Keep track of losses and global step
                running_g_loss += g_loss.item()
                running_d_loss += d_loss.item()

            #####################################
            #       Log Info for Epoch          #
            #####################################

            log_str = f"\n{'Completed Epoch:':<20}{epoch + 1:<10}"
            # Value to normalize so we get loss/sample
            norm = len(self.data.mnist_dataset)
            log_str += f"\n{'Discriminator Loss:':<20}{running_d_loss/norm:<10}"
            log_str += f"\n{'Generator Loss:':<20}{running_g_loss/norm:<10}\n"
            print(log_str)

            # Add information to Tensorboard
            self.writer.add_scalar('discriminator_loss', running_d_loss/norm, epoch)
            self.writer.add_scalar('generator_loss', running_g_loss/norm, epoch)

            self.writer.add_scalar('avg_real_logit', t.mean(real_logits).item(), epoch)
            self.writer.add_scalar('avg_fake_logit', t.mean(fake_logits).item(), epoch)

            self.writer.add_scalar('avg_gen_grad', t.mean(self.G.model[0].weight.grad).item(), epoch)
            self.writer.add_scalar('avg_dis_grad', t.mean(self.D.model[0].weight.grad).item(), epoch)

            z = self.data.sample_latent_space(batch_size=36).to(device)
            generated_imgs = self.G(z)

            img_grid = torchvision.utils.make_grid(generated_imgs.reshape(36, 1, 28, 28), nrow=6)
            self.writer.add_image('generated_images', img_grid, epoch)

        # Close tensorboard writer when we're done with training
        self.writer.close()
