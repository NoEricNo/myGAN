import torch
import torch.nn.functional as F

class GAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self, data_loader, accumulation_steps, scaler, chunk_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i, (real_data, real_existence) in enumerate(data_loader):
            #print(f"Training batch {i}")
            real_data = real_data.to(device)
            real_existence = real_existence.to(device)
            #print(f"Real data shape: {real_data.shape}")
            #print(f"Real existence shape: {real_existence.shape}")

            # Combine real_data and real_existence for Discriminator
            real_combined = torch.cat([real_data, real_existence], dim=1)
            #print(f"Combined real data shape for Discriminator: {real_combined.shape}")

            # Pad the data if the shape is not as expected
            if real_combined.shape[1] < chunk_size * 2:
                padding_size = chunk_size * 2 - real_combined.shape[1]
                real_combined = F.pad(real_combined, (0, padding_size))
                #print(f"Padded real_combined shape: {real_combined.shape}")

            # Train Discriminator
            self.optimizer_D.zero_grad()
            real_rating_validity, real_existence_prediction = self.discriminator(real_combined)
            #print(f"Discriminator output shapes: rating_validity: {real_rating_validity.shape}, existence_prediction: {real_existence_prediction.shape}")
            noise = torch.randn(real_data.size(0), 100, device=device)
            fake_data = self.generator(noise, chunk_size).detach().float()
            #print(f"Fake data shape from Generator: {fake_data.shape}")
            fake_rating_validity, fake_existence_prediction = self.discriminator(fake_data)
            #print(f"Discriminator output shapes for fake data: rating_validity: {fake_rating_validity.shape}, existence_prediction: {fake_existence_prediction.shape}")

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
