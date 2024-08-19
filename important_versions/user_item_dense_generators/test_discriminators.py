import torch
import torch.nn as nn
import torch.optim as optim
from generators import TrendGenerator, UserGenerator, MaskGenerator, InteractionGenerator
from discriminators import IndividualDiscriminator, GroupDiscriminator, InteractionDiscriminator

class GANTraining:
    def __init__(self, noise_dim, trend_dim, user_dim, item_dim, mask_dim, interaction_dim, window_size):
        # Initialize all models
        self.trend_generator = TrendGenerator(noise_dim, trend_dim)
        self.user_generator = UserGenerator(trend_dim, noise_dim, user_dim)
        self.mask_generator = MaskGenerator(user_dim, mask_dim)
        self.interaction_generator = InteractionGenerator(user_dim, item_dim, mask_dim, interaction_dim)

        self.individual_discriminator = IndividualDiscriminator(user_dim, item_dim, mask_dim, window_size)
        self.group_discriminator = GroupDiscriminator(user_dim, trend_dim)
        self.interaction_discriminator = InteractionDiscriminator(user_dim, item_dim, mask_dim, interaction_dim)

        # Optimizers for generators and discriminators
        self.g_optimizer = optim.Adam(
            list(self.trend_generator.parameters()) +
            list(self.user_generator.parameters()) +
            list(self.mask_generator.parameters()) +
            list(self.interaction_generator.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )

        self.d_optimizer = optim.Adam(
            list(self.individual_discriminator.parameters()) +
            list(self.group_discriminator.parameters()) +
            list(self.interaction_discriminator.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )

        # Loss function
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for GANs

    def train(self, real_data, epochs, batch_size, lambda_recon=10, lambda_sparsity=0.01, lambda_interaction=5):
        for epoch in range(epochs):
            for i in range(0, len(real_data), batch_size):
                # Prepare real and fake data
                real_batch = real_data[i:i+batch_size]
                noise = torch.randn(batch_size, noise_dim)

                # Generate trends and users
                trend = self.trend_generator(noise)
                user_latent = self.user_generator(trend, noise)
                mask = self.mask_generator(user_latent)

                # Generate interactions
                item_latent = torch.randn(batch_size, item_dim)  # Assuming random item latent factors
                interaction = self.interaction_generator(user_latent, item_latent, mask)

                # -------------------------
                #  Train Discriminators
                # -------------------------
                self.d_optimizer.zero_grad()

                # Real labels (1s) and fake labels (0s)
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                # Discriminator outputs for real data
                real_validity = self.individual_discriminator(real_batch['user'], real_batch['item'], real_batch['mask'])
                real_loss = self.criterion(real_validity, real_labels)

                # Discriminator outputs for fake data
                fake_validity = self.individual_discriminator(user_latent, item_latent, mask)
                fake_loss = self.criterion(fake_validity, fake_labels)

                # Total discriminator loss
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.d_optimizer.step()

                # ---------------------
                #  Train Generators
                # ---------------------
                self.g_optimizer.zero_grad()

                # Recompute fake validity for generator loss
                fake_validity = self.individual_discriminator(user_latent, item_latent, mask)
                g_loss_adv = self.criterion(fake_validity, real_labels)  # Generator wants to fool the discriminator

                # Reconstruction loss (Optional, depending on your specific setup)
                g_loss_recon = lambda_recon * nn.functional.mse_loss(user_latent, real_batch['user'])

                # Sparsity loss (Penalizing non-sparse mask)
                g_loss_sparsity = lambda_sparsity * torch.sum(torch.abs(mask))

                # Interaction consistency loss
                g_loss_interaction = lambda_interaction * nn.functional.mse_loss(interaction, real_batch['interaction'])

                # Total generator loss
                g_loss = g_loss_adv + g_loss_recon + g_loss_sparsity + g_loss_interaction
                g_loss.backward()
                self.g_optimizer.step()

                # Print losses every few steps
                if i % 100 == 0:
                    print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(real_data)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

if __name__ == '__main__':
    # Example parameters
    noise_dim = 100
    trend_dim = 50
    user_dim = 50
    item_dim = 50
    mask_dim = 10
    interaction_dim = 1
    window_size = 5
    epochs = 1000
    batch_size = 64

    # Assuming real_data is a dictionary containing real user, item, mask, and interaction matrices
    real_data = {
        'user': torch.randn(1000, user_dim),
        'item': torch.randn(1000, item_dim),
        'mask': torch.randn(1000, mask_dim),
        'interaction': torch.randn(1000, interaction_dim)
    }

    # Instantiate the GAN training class and start training
    gan_trainer = GANTraining(noise_dim, trend_dim, user_dim, item_dim, mask_dim, interaction_dim, window_size)
    gan_trainer.train(real_data, epochs, batch_size)
