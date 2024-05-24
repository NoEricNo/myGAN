import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

class GAN:
    def __init__(self, generator, discriminator, device, num_movies, lr_g, lr_d, beta1, beta2, label_smoothing, noise_factor, update_d_freq):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.num_movies = num_movies
        self.label_smoothing = label_smoothing
        self.noise_factor = noise_factor
        self.update_d_freq = update_d_freq

        self.criterion_validity = nn.BCELoss()
        self.criterion_existence = nn.BCELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(beta1, beta2))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

    def train(self, dataloader, num_epochs, accumulation_steps, scaler):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            for i, real_data in enumerate(dataloader):
                real_data = real_data.to(self.device).float()
                batch_size = real_data.size(0)

                # Train Discriminator
                noise = torch.randn(batch_size, 100, device=self.device).float()
                fake_data = self.generator(noise)

                real_validity, real_existence_pred = self.discriminator(real_data)
                fake_validity, fake_existence_pred = self.discriminator(fake_data.detach())

                real_labels = torch.ones_like(real_validity, device=self.device) * self.label_smoothing
                fake_labels = torch.zeros_like(fake_validity, device=self.device)

                d_loss_real_validity = self.criterion_validity(real_validity, real_labels)
                d_loss_fake_validity = self.criterion_validity(fake_validity, fake_labels)
                d_loss_real_existence = self.criterion_existence(real_existence_pred, real_data[:, :, -1])
                d_loss_fake_existence = self.criterion_existence(fake_existence_pred, fake_data[:, :, -1])

                d_loss = (d_loss_real_validity + d_loss_fake_validity + d_loss_real_existence + d_loss_fake_existence) / 4

                d_loss = d_loss / accumulation_steps
                d_loss.backward(retain_graph=True)

                if (i + 1) % accumulation_steps == 0:
                    self.optimizer_d.step()
                    self.optimizer_d.zero_grad()

                # Train Generator
                if i % self.update_d_freq == 0:
                    noise = torch.randn(batch_size, 100, device=self.device).float()
                    fake_data = self.generator(noise)
                    validity, existence_pred = self.discriminator(fake_data)

                    real_labels = torch.ones_like(validity, device=self.device)

                    g_loss_validity = self.criterion_validity(validity, real_labels)
                    g_loss_existence = self.criterion_existence(existence_pred, fake_data[:, :, -1])

                    g_loss = (g_loss_validity + g_loss_existence) / 2

                    g_loss = g_loss / accumulation_steps
                    g_loss.backward()

                    if (i + 1) % accumulation_steps == 0:
                        self.optimizer_g.step()
                        self.optimizer_g.zero_grad()

                if i % 10 == 0:
                    print(f"Batch {i}/{len(dataloader)}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

            print(f"Epoch {epoch+1} completed. Avg D Loss: {d_loss.item():.4f}, Avg G Loss: {g_loss.item():.4f}")
