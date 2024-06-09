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
        predictions = predictions.view(-1, 1)
        targets = targets.view(-1, 1)
        return nn.BCEWithLogitsLoss()(predictions, targets)

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
        fake_ratings = fake_ratings.to_dense()  # Ensure fake_ratings is a dense tensor
        #print(f"fake_ratings dtype: {fake_ratings.dtype}")  # Print dtype of fake_ratings
        #print(f"item_factors dtype: {item_factors.dtype}")  # Print dtype of item_factors

        # Project into latent space
        latent_representation = fake_ratings @ item_factors.T  # Shape: (256, 50)
        #print(f"latent_representation dtype: {latent_representation.dtype}")  # Print dtype of latent_representation

        # Project back to the original space
        reconstructed_ratings = latent_representation @ item_factors  # Shape: (256, 9724)
        #print(f"reconstructed_ratings dtype: {reconstructed_ratings.dtype}")  # Print dtype of reconstructed_ratings

        # Calculate MSE loss between original and reconstructed ratings
        return F.mse_loss(fake_ratings, reconstructed_ratings)

    def custom_loss(self, fake_validity, real_ratings, fake_ratings, fake_existence, item_factors):

        fake_ratings = fake_ratings.to_dense()
        fake_existence = fake_existence.to_dense()
        fake_ratings = fake_ratings * fake_existence

        adversarial_loss = self.adversarial_loss(fake_validity, torch.ones_like(fake_validity))
        distribution_loss = self.distribution_loss(real_ratings, fake_ratings, fake_existence)
        latent_loss = self.latent_factor_loss(fake_ratings, item_factors)

        return (adversarial_loss + distribution_loss + latent_loss) / 3

    def train_epoch(self, data_loader, num_epochs, item_factors):
        epoch_d_losses = []
        epoch_g_losses = []

        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs} started")

            for real_ratings, real_existence in data_loader:
                real_ratings = real_ratings.to(self.device).float()
                real_existence = real_existence.to(self.device).float()
                #print(f"real_ratings dtype: {real_ratings.dtype}")  # Print dtype of real_ratings
                #print(f"real_existence dtype: {real_existence.dtype}")

                # Train Discriminator
                self.optimizer_d.zero_grad()
                real_validity = self.discriminator(real_ratings, real_existence)

                noise = torch.randn(real_ratings.size(0), self.generator.input_size).to(self.device)
                fake_ratings, fake_existence = self.generator(noise)
                fake_ratings = fake_ratings.to_dense().to(self.device)
                fake_existence = fake_existence.to_dense().to(self.device)

                #print(f"fake_ratings dtype: {fake_ratings.dtype}")  # Print dtype of fake_ratings
                #print(f"fake_existence dtype: {fake_existence.dtype}")

                fake_validity = self.discriminator(fake_ratings, fake_existence)

                real_validity_loss = self.adversarial_loss(real_validity, torch.ones_like(real_validity))
                fake_validity_loss = self.adversarial_loss(fake_validity, torch.zeros_like(fake_validity))

                d_loss = real_validity_loss + fake_validity_loss
                d_loss.backward()
                self.optimizer_d.step()

                # Train Generator
                self.optimizer_g.zero_grad()
                noise = torch.randn(real_ratings.size(0), self.generator.input_size).to(self.device)
                fake_ratings, fake_existence = self.generator(noise)
                fake_ratings = fake_ratings.to_dense().to(self.device)
                fake_existence = fake_existence.to_dense().to(self.device)

                fake_validity = self.discriminator(fake_ratings, fake_existence)

                g_loss = self.custom_loss(fake_validity, real_ratings, fake_ratings, fake_existence, item_factors)
                g_loss.backward()
                self.optimizer_g.step()

                epoch_d_losses.append(d_loss.item())
                epoch_g_losses.append(g_loss.item())

            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, D Loss: {sum(epoch_d_losses) / len(epoch_d_losses):.4f}, G Loss: {sum(epoch_g_losses) / len(epoch_g_losses):.4f}")

        return epoch_d_losses, epoch_g_losses
