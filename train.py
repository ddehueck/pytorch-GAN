import torch
import torch.optim as optim
from data import GANData
from discriminator import Discriminator
from generator import Generator
from criteria.generator import GeneratorCriterion
from criteria.discriminator import DiscriminatorCriterion
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision

# Hyperparameters
EPOCHS = 500
LEARNING_RATE = 10e-3
BATCH_SIZE = 16
K = 1
latent_size = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

# Holds all data classes needed
data = GANData(batch_size=BATCH_SIZE)
img_size = data.get_img_size()

# Instantiate models
D = Discriminator(img_size).to(DEVICE)
G = Generator(img_size, latent_size).to(DEVICE)

# Instantiate criterion for both D and G
D_criterion = DiscriminatorCriterion()
G_criterion = GeneratorCriterion()

# Instantiate an optimizer for both D and G
D_optim = optim.SGD(D.parameters(), lr=LEARNING_RATE, momentum=0.9)
G_optim = optim.SGD(D.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Training Loop
print('\nBeginning training for {} epochs...'.format(EPOCHS))
for epoch in range(EPOCHS):
    print('\n Starting epoch: {}/{}'.format(epoch + 1, EPOCHS))

    running_d_loss = 0.0
    running_g_loss = 0.0

    for i in tqdm(range(len(data.trainset)//BATCH_SIZE)):

        # Optimize Discriminator
        for k in range(K):
            # Sample minibatches from P_data and P_z
            data_mb, _ = next(iter(data.trainloader))
            data_mb = data_mb.to(DEVICE)
            z_mb = G.sample_z(BATCH_SIZE).to(DEVICE)

            # Clear accumulated gradients
            D_optim.zero_grad()

            # Get probability scrores for minibatches
            generated_samples = G(z_mb)
            z_logits = D(generated_samples)
            data_logits = D(data_mb)
            D_loss = D_criterion(data_logits, z_logits)

            # Keep track of losses
            running_d_loss += D_loss.item()

            # Backprop
            D_loss.backward()
            D_optim.step()

        # Update Generator
        z_mb = G.sample_z(BATCH_SIZE).to(DEVICE)

        # Clear accumulated gradients
        G_optim.zero_grad()

        # Get sample and record loss
        generated_samples = G(z_mb)
        z_logits = D(generated_samples)
        G_loss = G_criterion(z_logits)

        # Backprop
        G_loss.backward()
        G_optim.step()

        # Keep track of losses
        running_g_loss += G_loss.item()

    # ---------------------------------------
    # Record epoch information
    # ---------------------------------------
    g_norm = len(data.trainset) // BATCH_SIZE
    d_norm = K * g_norm

    print('[Discriminator Loss]:', running_d_loss/d_norm)
    print("[Generator Loss]:", running_g_loss/g_norm)

    global_step = g_norm * epoch
    writer.add_scalar('discriminator_loss', running_d_loss/d_norm, global_step)
    writer.add_scalar('generator_loss', running_g_loss/g_norm, global_step)

    z_mb = G.sample_z(8).to(DEVICE)
    generated_samples = G(z_mb)

    grid = torchvision.utils.make_grid(generated_samples.reshape(8, 1, 28, 28))
    writer.add_image('images', grid, global_step)

writer.close()
