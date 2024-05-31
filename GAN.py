import torch
import torch.nn.functional as F
import logging


class GAN:
    def __init__(self, generator, discriminator, learning_rate_G, learning_rate_D, betas):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=learning_rate_G, betas=betas)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate_D, betas=betas)
        self.logger = logging.getLogger()

    def train(self, data_loader, accumulation_steps, scaler, chunk_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i, (real_data, real_existence) in enumerate(data_loader):

            real_data = real_data.to(device)
            real_existence = real_existence.to(device)

            # Combine real_data and real_existence for Discriminator
            real_combined = torch.cat([real_data, real_existence], dim=1).to(device)

            # Pad the data if the shape is not as expected
            if real_combined.shape[1] < chunk_size * 2:
                padding_size = chunk_size * 2 - real_combined.shape[1]
                real_combined = F.pad(real_combined, (0, padding_size)).to(device)

            # Train Discriminator
            self.optimizer_D.zero_grad()
            real_rating_validity, real_existence_prediction = self.discriminator(real_combined)
            noise = torch.randn(real_data.size(0), 100, device=device)
            fake_data = self.generator(noise, chunk_size).detach().float()
            fake_rating_validity, fake_existence_prediction = self.discriminator(fake_data)

            d_loss_real = F.binary_cross_entropy(real_rating_validity, torch.ones_like(real_rating_validity, device=device))
            d_loss_fake = F.binary_cross_entropy(fake_rating_validity, torch.zeros_like(fake_rating_validity, device=device))
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            self.optimizer_D.step()

            # Train Generator
            self.optimizer_G.zero_grad()
            gen_data = self.generator(noise, chunk_size)
            validity, existence = self.discriminator(gen_data)
            g_loss = F.binary_cross_entropy(validity, torch.ones_like(validity, device=device))
            g_loss.backward()
            self.optimizer_G.step()

            # Print losses and batch duration
            if i % 200 == 0:
                print(f"Local Batch {i+1}/{len(data_loader)}: D_loss: {d_loss.item()}, G_loss: {g_loss.item()}")
                self.logger.info(f"Local Batch {i+1}/{len(data_loader)}: D_loss: {d_loss.item()}, G_loss: {g_loss.item()}")


