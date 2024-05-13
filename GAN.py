import torch
from torch import nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def save_generator_output(generator, epoch, batch, device, num_images=64, z_dim=100, save=True, show=False):
    with torch.no_grad():
        generator.eval()
        fixed_noise = torch.randn(num_images, z_dim, device=device)
        fake_data = generator(fixed_noise).detach().cpu()
        generator.train()


class GAN(nn.Module):
    def __init__(self, generator, discriminator, device):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion_rating = nn.CrossEntropyLoss() # might need to change this
        self.criterion_existence = nn.BCELoss()# or this

    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            for i, (real_data, real_existence) in enumerate(dataloader):
                real_data, real_existence = real_data.to(self.device), real_existence.to(self.device)
                valid = torch.ones((real_data.size(0), 1), device=self.device)
                fake = torch.zeros((real_data.size(0), 1), device=self.device)
                fake_ratings, fake_existence = fake_data[:, :, :-1], fake_data[:, :, -1]

                # Generate fake data
                noise = torch.randn(real_data.size(0), 100, device=self.device)
                fake_data = self.generator(noise)

                # Discriminator step

                self.optimizer_D.zero_grad()
                real_ratings, real_existence = real_data[:, :, :-1], real_data[:, :, -1]
                real_loss_ratings = self.criterion_rating(self.discriminator(real_ratings), valid)
                real_loss_existence = self.criterion_existence(real_existence, real_existence)
                fake_loss_ratings = self.criterion_rating(self.discriminator(fake_ratings.detach()), fake)
                fake_loss_existence = self.criterion_existence(fake_existence.detach(), fake)
                d_loss = (real_loss_ratings + real_loss_existence + fake_loss_ratings + fake_loss_existence) / 4
                d_loss.backward()
                self.optimizer_D.step()


                # Generator step
                self.optimizer_G.zero_grad()
                regenerated_validity = self.criterion_rating(self.discriminator(fake_ratings), valid)
                regenerated_existence = self.criterion_existence(fake_existence, real_existence)
                g_loss = (regenerated_validity + regenerated_existence) / 2
                g_loss.backward()
                self.optimizer_G.step()
                self.optimizer_G.step()

                if i % 50 == 0:
                    print(f"Epoch {epoch}/{num_epochs}, Batch {i}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

