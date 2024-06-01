import torch
import torch.nn.functional as F
import logging

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, noise, real_data):
        fake_data = self.generator(noise)
        real_rating_validity, real_existence_prediction = self.discriminator(real_data)
        fake_rating_validity, fake_existence_prediction = self.discriminator(fake_data)
        return real_rating_validity, real_existence_prediction, fake_rating_validity, fake_existence_prediction

    def train(self, loader, accumulation_steps, scaler, chunk_size):
        for real_data, real_existence in loader:
            real_data, real_existence = real_data.to(self.device), real_existence.to(self.device)
            real_combined = torch.cat([real_data.view(real_data.size(0), -1), real_existence], dim=1).to(self.device)
            print(f"Real combined shape: {real_combined.shape}")

            self.optimizer_D.zero_grad()
            real_rating_validity, real_existence_prediction = self.discriminator(real_combined)
            noise = torch.randn(real_data.size(0), self.generator.noise_dim, device=device)
            fake_data = self.generator(noise, chunk_size).detach().float()
            print(f"Fake data shape: {fake_data.shape}")
            fake_combined = fake_data.view(fake_data.size(0), -1)  # Ensure same shape as real_combined
            print(f"Fake combined shape: {fake_combined.shape}")
            fake_rating_validity, fake_existence_prediction = self.discriminator(fake_combined)

            real_labels = torch.ones_like(real_rating_validity) * 0.9
            fake_labels = torch.zeros_like(fake_rating_validity) + 0.1

            d_loss_real = F.binary_cross_entropy(real_rating_validity, real_labels)
            d_loss_fake = F.binary_cross_entropy(fake_rating_validity, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            self.optimizer_D.step()

            self.optimizer_G.zero_grad()
            gen_data = self.generator(noise, chunk_size)
            gen_combined = gen_data.view(gen_data.size(0), -1)  # Ensure same shape as real_combined
            print(f"Gen combined shape: {gen_combined.shape}")
            validity, existence = self.discriminator(gen_combined)
            g_loss = F.binary_cross_entropy(validity, torch.ones_like(validity, device=device) * 0.9)
            g_loss.backward()
            self.optimizer_G.step()

            if i % 200 == 0:
                print(f"Local Batch {i+1}/{len(data_loader)}: D_loss: {d_loss.item()}, G_loss: {g_loss.item()}")
                self.logger.info(f"Local Batch {i+1}/{len(data_loader)}: D_loss: {d_loss.item()}, G_loss: {g_loss.item()}")

        print(f"Epoch complete")
        self.logger.info(f"Epoch complete")