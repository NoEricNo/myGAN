import torch
import torch.nn as nn
import logging
import torch.optim as optim


class GAN(nn.Module):
    def __init__(self, generator, discriminator, device, criterion, optimizer_g, optimizer_d, logger):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.criterion = criterion
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.logger = logger

    def forward(self, real_data, real_existence):
        batch_size = real_data.size(0)
        noise = torch.randn(batch_size, self.generator.input_size).to(self.device)

        fake_ratings, fake_existence = self.generator(noise)

        real_data_flat = real_data.view(batch_size, -1)
        real_existence_flat = real_existence.view(batch_size, -1)
        fake_ratings_flat = fake_ratings.view(batch_size, -1)
        fake_existence_flat = fake_existence.view(batch_size, -1)

        real_validity, real_existence_pred = self.discriminator(real_data_flat, real_existence_flat)
        fake_validity, fake_existence_pred = self.discriminator(fake_ratings_flat, fake_existence_flat)

        return real_validity, real_existence_pred, fake_validity, fake_existence_pred

    def train_epoch(self, data_loader, chunk_size, num_chunks, num_epochs):
        epoch_d_losses = []
        epoch_g_losses = []

        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs} started")

            for chunk_idx, chunk_loader in enumerate(data_loader):
                if chunk_idx >= num_chunks:
                    break

                for real_data, real_existence in chunk_loader:
                    real_data, real_existence = real_data.to(self.device), real_existence.to(self.device)

                    # Discriminator Training
                    self.optimizer_d.zero_grad()
                    real_validity, real_existence_pred, fake_validity, fake_existence_pred = self.forward(real_data,
                                                                                                          real_existence)

                    real_validity_loss = self.criterion(real_validity, torch.ones_like(real_validity))
                    real_existence_loss = self.criterion(real_existence_pred,
                                                         real_existence.view(real_existence.size(0), -1))
                    fake_validity_loss = self.criterion(fake_validity, torch.zeros_like(fake_validity))
                    fake_existence_loss = self.criterion(fake_existence_pred, torch.zeros_like(fake_existence_pred))

                    d_loss = real_validity_loss + real_existence_loss + fake_validity_loss + fake_existence_loss
                    d_loss.backward()
                    self.optimizer_d.step()

                    # Generator Training
                    self.optimizer_g.zero_grad()
                    _, _, fake_validity, fake_existence_pred = self.forward(real_data, real_existence)

                    g_validity_loss = self.criterion(fake_validity, torch.ones_like(fake_validity))
                    g_existence_loss = self.criterion(fake_existence_pred, torch.ones_like(fake_existence_pred))

                    g_loss = g_validity_loss + g_existence_loss
                    g_loss.backward()
                    self.optimizer_g.step()

                    epoch_d_losses.append(d_loss.item())
                    epoch_g_losses.append(g_loss.item())

                self.logger.info(
                    f"Chunk {chunk_idx + 1}/{num_chunks}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs}, D Loss: {sum(epoch_d_losses) / len(epoch_d_losses):.4f}, G Loss: {sum(epoch_g_losses) / len(epoch_g_losses):.4f}")

        return epoch_d_losses, epoch_g_losses