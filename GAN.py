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
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion_rating = nn.BCELoss()
        self.criterion_existence = nn.BCELoss()#!!!!!!!!!!!!!!!!change this

    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            for i, (real_data, real_existence) in enumerate(dataloader):
                real_data, real_existence = real_data.to(self.device), real_existence.to(self.device)
                valid = torch.ones((real_data.size(0), 1), device=self.device)
                fake = torch.zeros((real_data.size(0), 1), device=self.device)

                # Generate fake data
                noise = torch.randn(real_data.size(0), 100, device=self.device)
                fake_data = self.generator(noise)

                # Discriminator step
                self.optimizer_D.zero_grad()
                real_loss = self.criterion_rating(self.discriminator(real_data), valid)
                fake_loss = self.criterion_rating(self.discriminator(fake_data.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # Generator step
                self.optimizer_G.zero_grad()
                regenerated_validity = self.criterion_rating(self.discriminator(fake_data), valid)
                g_loss = regenerated_validity
                g_loss.backward()
                self.optimizer_G.step()

                if i % 50 == 0:
                    print(f"Epoch {epoch}/{num_epochs}, Batch {i}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

