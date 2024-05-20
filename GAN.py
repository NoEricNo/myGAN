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
    def __init__(self, generator, discriminator, device, num_movies):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.num_movies = num_movies  # Store num_movies as an attribute
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion_rating = nn.BCELoss()
        self.criterion_existence = nn.BCELoss()

    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")  # Print epoch number at the start of each epoch
            total_d_loss = 0
            total_g_loss = 0
            num_batches = len(dataloader)

            for i, (real_data, real_existence) in enumerate(dataloader):
                real_data, real_existence = real_data.to(self.device), real_existence.to(self.device)
                valid = torch.ones((real_data.size(0), 1), device=self.device)
                fake = torch.zeros((real_data.size(0), 1), device=self.device)

                # Generate fake data
                noise = torch.randn(real_data.size(0), 100, device=self.device)
                fake_data = self.generator(noise)
                fake_ratings, fake_existence = fake_data[:, :, :-1], fake_data[:, :, -1]

                # Discriminator step
                self.optimizer_D.zero_grad()
                real_ratings, real_existence = real_data[:, :, :-1], real_data[:, :, -1]

                # Compute discriminator losses
                real_loss_ratings = self.criterion_rating(self.discriminator(real_ratings), valid)
                real_loss_existence = self.criterion_existence(real_existence.view(-1, self.num_movies),
                                                               real_existence.view(-1, self.num_movies))
                fake_loss_ratings = self.criterion_rating(self.discriminator(fake_ratings.detach()), fake)
                fake_loss_existence = self.criterion_existence(fake_existence.detach().view(-1, self.num_movies),
                                                               fake_existence.view(-1, self.num_movies))

                # Total discriminator loss
                d_loss_ratings = (real_loss_ratings + fake_loss_ratings) / 2
                d_loss_existence = (real_loss_existence + fake_loss_existence) / 2
                d_loss = (d_loss_ratings + d_loss_existence) / 2
                d_loss.backward(retain_graph=True)  # Retain the graph for generator's backward pass
                self.optimizer_D.step()

                # Generator step
                self.optimizer_G.zero_grad()
                regenerated_validity = self.criterion_rating(self.discriminator(fake_ratings), valid)
                regenerated_existence = self.criterion_existence(fake_existence.view(-1, self.num_movies),
                                                                 real_existence.view(-1, self.num_movies))

                # Total generator loss
                g_loss_validity = regenerated_validity
                g_loss_existence = regenerated_existence
                g_loss = (g_loss_validity + g_loss_existence) / 2
                g_loss.backward()  # No retain_graph=True here
                self.optimizer_G.step()

                # Accumulate losses
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()

                if i % 2 == 0:
                    print(f"Batch {i}/{num_batches}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

            # Print average losses at the end of the epoch
            avg_d_loss = total_d_loss / num_batches
            avg_g_loss = total_g_loss / num_batches
            print(f"Epoch {epoch + 1} completed. Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")

            # Optionally, save generator output at the end of each epoch
            save_generator_output(self.generator, epoch, num_batches, self.device)

