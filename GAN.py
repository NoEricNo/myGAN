import torch
import torch.nn as nn
import torch.nn.functional as F


class GAN(nn.Module):
    def __init__(self, generator, discriminator, device, optimizer_g, optimizer_d, logger):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.logger = logger

    def adversarial_loss(self, predictions, targets):
        predictions = predictions.view(-1, 1)  # Reshape predictions to (batch_size, 1)
        targets = targets.view(-1, 1)  # Reshape targets to (batch_size, 1)
        return nn.BCEWithLogitsLoss()(predictions, targets)

    def distribution_loss(self, real_ratings, fake_ratings, fake_existence):
        mask = fake_existence > 0
        real_mean = real_ratings.sum() / real_ratings.numel()
        fake_mean = (fake_ratings * mask).sum() / mask.sum()
        real_std = ((real_ratings - real_mean) ** 2).sum() / real_ratings.numel()
        fake_std = (((fake_ratings * mask) - real_mean) ** 2).sum() / mask.sum()
        mean_loss = F.mse_loss(fake_mean, real_mean)
        std_loss = F.mse_loss(fake_std, real_std)
        return mean_loss + std_loss

    def latent_factor_loss(self, fake_ratings, item_factors):
        # Project the generated ratings onto the item latent factors
        projected_ratings = fake_ratings @ item_factors.T
        # Calculate the loss as the difference between the generated ratings and their projections
        return F.mse_loss(fake_ratings, projected_ratings)

    def custom_loss(self, real_ratings, fake_ratings, real_existence, fake_existence, item_factors_chunk):
        fake_ratings = fake_ratings * fake_existence

        adversarial_loss = self.adversarial_loss(real_ratings, fake_ratings)
        distribution_loss = self.distribution_loss(real_ratings, fake_ratings, fake_existence)
        latent_loss = self.latent_factor_loss(fake_ratings, item_factors_chunk)

        return (adversarial_loss + distribution_loss + latent_loss) / 3

    def train_epoch(self, data_loader, num_chunks, num_epochs, item_factors):
        epoch_d_losses = []
        epoch_g_losses = []

        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs} started")

            for chunk_idx, chunk_loader in enumerate(data_loader):
                if chunk_idx >= num_chunks:
                    break

                for real_ratings, real_existence, user_ids, movie_ids in chunk_loader:
                    real_ratings = real_ratings.to_dense().float().to(self.device)
                    real_existence = real_existence.to_dense().float().to(self.device)
                    user_ids = user_ids.float().to(self.device).view(-1, 1)
                    movie_ids = movie_ids.float().to(self.device).view(-1, 1)

                    # Extract the item factors for the current chunk
                    item_factors_chunk = item_factors[movie_ids.squeeze().long()]

                    # Train Discriminator
                    self.optimizer_d.zero_grad()
                    real_validity = self.discriminator(real_ratings, real_existence, user_ids, movie_ids)

                    noise = torch.randn(real_ratings.size(0), self.generator.input_size).to(self.device)
                    fake_ratings, fake_existence = self.generator(noise)

                    fake_ratings = fake_ratings.to_dense()
                    fake_existence = fake_existence.to_dense()
                    fake_ratings = fake_ratings[:, :real_ratings.size(1)]
                    fake_existence = fake_existence[:, :real_existence.size(1)]
                    fake_ratings = fake_ratings * fake_existence

                    # Rounding the generated ratings
                    fake_ratings = torch.clamp(fake_ratings, 0.5, 5.0)
                    fake_ratings = torch.round(fake_ratings * 2) / 2

                    fake_validity = self.discriminator(fake_ratings, fake_existence, user_ids, movie_ids)

                    real_validity_loss = self.adversarial_loss(real_validity, torch.ones_like(real_validity))
                    fake_validity_loss = self.adversarial_loss(fake_validity, torch.zeros_like(fake_validity))

                    d_loss = real_validity_loss + fake_validity_loss
                    d_loss.backward()
                    self.optimizer_d.step()

                    # Train Generator
                    self.optimizer_g.zero_grad()
                    noise = torch.randn(real_ratings.size(0), self.generator.input_size).to(self.device)
                    fake_ratings, fake_existence = self.generator(noise)

                    fake_ratings = fake_ratings.to_dense()
                    fake_existence = fake_existence.to_dense()
                    fake_ratings = fake_ratings[:, :real_ratings.size(1)]
                    fake_existence = fake_existence[:, :real_existence.size(1)]
                    fake_ratings = fake_ratings * fake_existence

                    # Rounding the generated ratings
                    fake_ratings = torch.clamp(fake_ratings, 0.5, 5.0)
                    fake_ratings = torch.round(fake_ratings * 2) / 2

                    fake_validity = self.discriminator(fake_ratings, fake_existence, user_ids, movie_ids)

                    # Use custom loss function
                    g_loss = self.custom_loss(real_ratings, fake_ratings, real_existence, fake_existence,
                                              item_factors_chunk)
                    g_loss.backward()
                    self.optimizer_g.step()

                    epoch_d_losses.append(d_loss.item())
                    epoch_g_losses.append(g_loss.item())

                self.logger.info(
                    f"Chunk {chunk_idx + 1}/{num_chunks} (Epoch {epoch + 1}), D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs}, D Loss: {sum(epoch_d_losses) / len(epoch_d_losses):.4f}, G Loss: {sum(epoch_g_losses) / len(epoch_g_losses):.4f}")

        return epoch_d_losses, epoch_g_losses
