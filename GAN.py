import torch
import torch.nn as nn
import torch.nn.functional as F

class GAN(nn.Module):
    def __init__(self, generator, discriminator, device, optimizer_g, optimizer_d, logger, wandb):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.logger = logger
        self.wandb = wandb
        self.lambda_gp = 10  # Gradient penalty coefficient

    def distribution_loss(self, real_ratings, fake_ratings, fake_existence):
        mask = fake_existence.to_dense() > 0
        real_ratings = real_ratings.to_dense()
        fake_ratings = fake_ratings.to_dense()

        real_mean = real_ratings.sum() / real_ratings.numel()
        fake_mean = (fake_ratings * mask).sum() / mask.sum()
        real_std = ((real_ratings - real_mean) ** 2).sum() / real_ratings.numel()
        fake_std = (((fake_ratings * mask) - real_mean) ** 2).sum() / mask.sum()

        mean_loss = F.mse_loss(fake_mean, real_mean)
        std_loss = F.mse_loss(fake_std, real_std)
        return mean_loss + std_loss

    def latent_factor_loss(self, fake_ratings, item_factors):
        fake_ratings = fake_ratings.to_dense()
        latent_representation = fake_ratings @ item_factors.T
        reconstructed_ratings = latent_representation @ item_factors
        return F.mse_loss(fake_ratings, reconstructed_ratings)

    def custom_loss(self, fake_validity, real_ratings, fake_ratings, fake_existence, item_factors):
        fake_ratings = fake_ratings.to_dense()
        fake_existence = fake_existence.to_dense()
        fake_ratings = fake_ratings * fake_existence

        # Adversarial loss
        adversarial_loss = F.binary_cross_entropy_with_logits(fake_validity, torch.ones_like(fake_validity))

        # Distribution loss to match mean and standard deviation
        distribution_loss = self.distribution_loss(real_ratings, fake_ratings, fake_existence)

        # Latent factor loss to ensure alignment with item factors
        latent_loss = self.latent_factor_loss(fake_ratings, item_factors)

        # Combine all loss components with adjusted weights
        total_loss = adversarial_loss + 0.5 * distribution_loss + 0.5 * latent_loss
        return total_loss

    def clip_loss(self, loss, max_value=100.0):
        return torch.clamp(loss, max=max_value)

    def post_process(self, ratings, existence):
        ratings = ratings.to_dense() if ratings.is_sparse else ratings
        existence = existence.to_dense() if existence.is_sparse else existence

        ratings = torch.round(ratings * 2) / 2.0
        existence = (existence > 0.5).float()
        ratings = ratings * existence
        return ratings, existence

    def train_epoch(self, data_loader, num_epochs, item_factors):
        epoch_d_losses = []
        epoch_g_losses = []

        for epoch in range(num_epochs):
            for real_ratings, real_existence in data_loader:
                real_ratings = real_ratings.to(self.device).float()
                real_existence = real_existence.to(self.device).float()

                # Train Discriminator
                self.optimizer_d.zero_grad()
                real_validity = self.discriminator(real_ratings, real_existence)

                noise = torch.randn(real_ratings.size(0), self.generator.input_size).to(self.device)
                fake_ratings, fake_existence = self.generator(noise)

                # Post-process the generated data
                #fake_ratings, fake_existence = self.post_process(fake_ratings, fake_existence)

                fake_validity = self.discriminator(fake_ratings, fake_existence)

                real_validity_loss = F.binary_cross_entropy_with_logits(real_validity, torch.ones_like(real_validity))
                fake_validity_loss = F.binary_cross_entropy_with_logits(fake_validity, torch.zeros_like(fake_validity))

                d_loss = real_validity_loss + fake_validity_loss
                d_loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.optimizer_d.step()

                # Train Generator
                self.optimizer_g.zero_grad()
                noise = torch.randn(real_ratings.size(0), self.generator.input_size).to(self.device)
                fake_ratings, fake_existence = self.generator(noise)

                # Post-process the generated data
                #fake_ratings, fake_existence = self.post_process(fake_ratings, fake_existence)

                fake_validity = self.discriminator(fake_ratings, fake_existence)

                g_loss = self.custom_loss(fake_validity, real_ratings, fake_ratings, fake_existence, item_factors)
                g_loss.backward()
                nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                self.optimizer_g.step()

                d_loss = self.clip_loss(d_loss)
                g_loss = self.clip_loss(g_loss)

                epoch_d_losses.append(d_loss.item())
                epoch_g_losses.append(g_loss.item())

                # Log to wandb
                self.wandb.log({
                    "d_loss": d_loss.item(),
                    "g_loss": g_loss.item(),
                    "epoch": epoch
                })

            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs}, D Loss: {sum(epoch_d_losses) / len(epoch_d_losses):.4f}, G Loss: {sum(epoch_g_losses) / len(epoch_g_losses):.4f}")

            avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
            avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)

            # Log epoch averages
            self.wandb.log({
                "epoch_avg_d_loss": avg_d_loss,
                "epoch_avg_g_loss": avg_g_loss,
                "epoch": epoch
            })
        return epoch_d_losses, epoch_g_losses
