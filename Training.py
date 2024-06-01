import torch
import torch.nn as nn
import torch.optim as optim
import logging

class Training:
    def __init__(self, generator, discriminator, device):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train_epoch(self, loader, chunk_size, num_chunks, num_epochs=1, accumulation_steps=1, scaler=None, verbose=True):
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch + 1}/{num_epochs} started")
            print(f"Epoch {epoch + 1}/{num_epochs} started")
            for chunk_idx in range(num_chunks):
                logging.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
                print(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
                real_data, real_existence = loader.get_chunk(chunk_idx)
                real_data, real_existence = real_data.to(self.device), real_existence.to(self.device)

                self.train(real_data, real_existence, accumulation_steps, scaler, chunk_size)

    def train(self, real_data, real_existence, accumulation_steps, scaler, chunk_size):
        real_data_flat = real_data.view(real_data.size(0), -1)
        real_existence_flat = real_existence.view(real_existence.size(0), -1)

        logging.info(f"Initial real data shape: {real_data.shape}")
        logging.info(f"Initial real existence shape: {real_existence.shape}")
        logging.info(f"Real data shape after to(device): {real_data_flat.shape}")
        logging.info(f"Real existence shape after to(device): {real_existence_flat.shape}")

        real_combined = torch.cat([real_data_flat, real_existence_flat], dim=1)
        logging.info(f"Real combined shape: {real_combined.shape}")

        fake_combined = torch.cat([real_data_flat, real_existence_flat], dim=1)
        logging.info(f"Fake combined shape: {fake_combined.shape}")