import torch
import torch.nn as nn
import torch.nn.functional as F

class GAN(nn.Module):
    def __init__(self, generator_r, generator_e, main_discriminator, distribution_discriminator, latent_factor_discriminator, device,
                 optimizer_g_r, optimizer_g_e, optimizer_d_main, optimizer_d_distribution, optimizer_d_latent, logger):
        super(GAN, self).__init__()
        self.generator_r = generator_r
        self.generator_e = generator_e
        self.main_discriminator = main_discriminator
        self.distribution_discriminator = distribution_discriminator
        self.latent_factor_discriminator = latent_factor_discriminator
        self.device = device
        self.optimizer_g_r = optimizer_g_r
        self.optimizer_g_e = optimizer_g_e
        self.optimizer_d_main = optimizer_d_main
        self.optimizer_d_distribution = optimizer_d_distribution
        self.optimizer_d_latent = optimizer_d_latent
        self.logger = logger

    def custom_loss(self, fake_validity_main, fake_validity_distribution, fake_validity_latent):
        # Adversarial losses
        adversarial_loss_main = F.binary_cross_entropy_with_logits(fake_validity_main, torch.ones_like(fake_validity_main))
        adversarial_loss_distribution = F.binary_cross_entropy_with_logits(fake_validity_distribution, torch.ones_like(fake_validity_distribution))
        adversarial_loss_latent = F.binary_cross_entropy_with_logits(fake_validity_latent, torch.ones_like(fake_validity_latent))

        # Combine adversarial losses
        total_loss = (2*adversarial_loss_main + adversarial_loss_distribution + adversarial_loss_latent)/4
        return total_loss

    def train_epoch(self, data_loader, num_epochs, item_factors):
        epoch_d_main_losses = []
        epoch_d_distribution_losses = []
        epoch_d_latent_losses = []
        epoch_g_r_losses = []
        epoch_g_e_losses = []

        for epoch in range(num_epochs):
            for real_ratings, real_existence in data_loader:
                real_ratings = real_ratings.to(self.device).float()
                real_existence = real_existence.to(self.device).float()

                # Log batch sizes
                #self.logger.info(f"Batch size - real_ratings: {real_ratings.size(0)}, real_existence: {real_existence.size(0)}")

                # Train Main Discriminator
                self.optimizer_d_main.zero_grad()
                real_validity_main = self.main_discriminator(real_ratings, real_existence)

                noise = torch.randn(real_ratings.size(0), self.generator_r.input_size).to(self.device)
                fake_ratings = self.generator_r(noise).detach()
                fake_existence = self.generator_e(noise).detach()

                # Log batch sizes for fake data
                #self.logger.info(f"Batch size - fake_ratings: {fake_ratings.size(0)}, fake_existence: {fake_existence.size(0)}")

                # Convert to dense and apply existence mask
                fake_ratings = fake_ratings.to_dense()
                fake_existence = fake_existence.to_dense()
                fake_ratings = fake_ratings * fake_existence

                fake_validity_main = self.main_discriminator(fake_ratings, fake_existence)

                real_validity_loss_main = F.binary_cross_entropy_with_logits(real_validity_main, torch.ones_like(real_validity_main))
                fake_validity_loss_main = F.binary_cross_entropy_with_logits(fake_validity_main, torch.zeros_like(fake_validity_main))

                d_loss_main = real_validity_loss_main + fake_validity_loss_main
                d_loss_main.backward()
                nn.utils.clip_grad_norm_(self.main_discriminator.parameters(), max_norm=1.0)
                self.optimizer_d_main.step()

                # Train Distribution Discriminator
                self.optimizer_d_distribution.zero_grad()
                real_validity_distribution = self.distribution_discriminator(real_ratings, fake_ratings)
                fake_validity_distribution = self.distribution_discriminator(real_ratings, fake_ratings)

                real_validity_loss_distribution = F.binary_cross_entropy_with_logits(real_validity_distribution, torch.ones_like(real_validity_distribution))
                fake_validity_loss_distribution = F.binary_cross_entropy_with_logits(fake_validity_distribution, torch.zeros_like(fake_validity_distribution))

                d_loss_distribution = real_validity_loss_distribution + fake_validity_loss_distribution
                d_loss_distribution.backward()
                nn.utils.clip_grad_norm_(self.distribution_discriminator.parameters(), max_norm=1.0)
                self.optimizer_d_distribution.step()

                # Train Latent Factor Discriminator
                self.optimizer_d_latent.zero_grad()
                real_validity_latent = self.latent_factor_discriminator(fake_ratings, item_factors)
                fake_validity_latent = self.latent_factor_discriminator(fake_ratings, item_factors)

                real_validity_loss_latent = F.binary_cross_entropy_with_logits(real_validity_latent, torch.ones_like(real_validity_latent))
                fake_validity_loss_latent = F.binary_cross_entropy_with_logits(fake_validity_latent, torch.zeros_like(fake_validity_latent))

                d_loss_latent = real_validity_loss_latent + fake_validity_loss_latent
                d_loss_latent.backward()
                nn.utils.clip_grad_norm_(self.latent_factor_discriminator.parameters(), max_norm=1.0)
                self.optimizer_d_latent.step()

                # Train Ratings Generator
                self.optimizer_g_r.zero_grad()
                noise_r = torch.randn(real_ratings.size(0), self.generator_r.input_size).to(self.device)
                fake_ratings = self.generator_r(noise_r)
                noise_e = torch.randn(real_ratings.size(0), self.generator_e.input_size).to(self.device)
                fake_existence = self.generator_e(noise_e)  # Generate new fake existence flags

                # Convert to dense and apply existence mask
                fake_ratings = fake_ratings.to_dense()
                fake_existence = fake_existence.to_dense()
                fake_ratings = fake_ratings * fake_existence

                fake_validity_main = self.main_discriminator(fake_ratings, fake_existence)
                fake_validity_distribution = self.distribution_discriminator(real_ratings, fake_ratings)
                fake_validity_latent = self.latent_factor_discriminator(fake_ratings, item_factors)

                g_loss_ratings = self.custom_loss(fake_validity_main, fake_validity_distribution, fake_validity_latent)
                g_loss_ratings.backward()
                nn.utils.clip_grad_norm_(self.generator_r.parameters(), max_norm=1.0)
                self.optimizer_g_r.step()

                # Train Existence Generator
                self.optimizer_g_e.zero_grad()
                noise_e = torch.randn(real_ratings.size(0), self.generator_e.input_size).to(self.device)
                fake_existence = self.generator_e(noise_e)
                noise_r = torch.randn(real_ratings.size(0), self.generator_r.input_size).to(self.device)
                fake_ratings = self.generator_r(noise_r)  # Generate new fake ratings

                # Convert to dense and apply existence mask
                fake_ratings = fake_ratings.to_dense()
                fake_existence = fake_existence.to_dense()
                fake_ratings = fake_ratings * fake_existence

                fake_validity_main = self.main_discriminator(fake_ratings, fake_existence)

                g_loss_existence = F.binary_cross_entropy_with_logits(fake_validity_main, torch.ones_like(fake_validity_main))  # Adversarial loss

                g_loss_existence.backward()
                nn.utils.clip_grad_norm_(self.generator_e.parameters(), max_norm=1.0)
                self.optimizer_g_e.step()

                epoch_d_main_losses.append(d_loss_main.item())
                epoch_d_distribution_losses.append(d_loss_distribution.item())
                epoch_d_latent_losses.append(d_loss_latent.item())
                epoch_g_r_losses.append(g_loss_ratings.item())
                epoch_g_e_losses.append(g_loss_existence.item())

            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs}, D Main Loss: {sum(epoch_d_main_losses) / len(epoch_d_main_losses):.4f}, "
                f"D Distribution Loss: {sum(epoch_d_distribution_losses) / len(epoch_d_distribution_losses):.4f}, "
                f"D Latent Loss: {sum(epoch_d_latent_losses) / len(epoch_d_latent_losses):.4f}, "
                f"G Ratings Loss: {sum(epoch_g_r_losses) / len(epoch_g_r_losses):.4f}, "
                f"G Existence Loss: {sum(epoch_g_e_losses) / len(epoch_g_e_losses):.4f}")

        return epoch_d_main_losses, epoch_d_distribution_losses, epoch_d_latent_losses, epoch_g_r_losses, epoch_g_e_losses
