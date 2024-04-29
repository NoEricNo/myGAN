import torch
from torch import nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def save_generator_output(generator, epoch, batch, device, num_images=64, z_dim=100, save=True, show=False):
    with torch.no_grad():
        generator.eval()
        # Generate images from the fixed noise
        fixed_noise = torch.randn(num_images, z_dim, device=device)  # Generate new noise for each call
        fake_images = generator(fixed_noise).detach().cpu()
        generator.train()

    # Make a grid from the images and save or show it
    img_grid = vutils.make_grid(fake_images, padding=2, normalize=True)

    if save:
        # Ensure the output directory exists
        import os
        os.makedirs("./output_images", exist_ok=True)
        vutils.save_image(img_grid, f"./output_images/epoch_{epoch}_batch_{batch}.png")

    if show:
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Generated Images at Epoch {epoch} Batch {batch}")
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.show()


class GAN(nn.Module):
    def __init__(self, generator, discriminator, device):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

        # Optimizers
        self.lr = 0.0002
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # Loss function
        self.criterion = nn.BCELoss()

    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            for i, (imgs, _) in enumerate(dataloader):

                # Adversarial ground truths
                valid = torch.ones((imgs.size(0), 1), device=self.device, requires_grad=False)
                fake = torch.zeros((imgs.size(0), 1), device=self.device, requires_grad=False)

                # Configure input
                real_imgs = imgs.to(self.device)

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = torch.randn(imgs.size(0), 100).to(self.device)

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.criterion(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.criterion(self.discriminator(real_imgs), valid)
                fake_loss = self.criterion(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                # Logging or any other operations per batch or per epoch
                if i % 50 == 0:
                    print(f"Epoch {epoch}/{num_epochs}, Batch {i}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
                    fixed_noise = torch.randn(64, 100, device=self.device)  # 64 is the number of images you want to generate
                    save_generator_output(self.generator, fixed_noise, epoch, i, self.device, save=True, show=False)
