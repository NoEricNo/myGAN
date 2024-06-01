import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, fc1_size, main_sizes, dropout_rate, existence_gen_size):
        super(Generator, self).__init__()
        noise_dim = 100  # Assuming a fixed noise dimension, adjust if needed
        print(f"Initializing Generator with fc1_size: {fc1_size}, main_sizes: {main_sizes}, dropout_rate: {dropout_rate}, existence_gen_size: {existence_gen_size}")
        self.fc1 = nn.Linear(noise_dim, fc1_size)
        self.main = nn.Sequential(
            nn.Linear(fc1_size, main_sizes[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(main_sizes[0], main_sizes[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(main_sizes[1], existence_gen_size)
        )

    def forward(self, noise):
        x = self.fc1(noise)
        x = self.main(x)
        ratings = x.view(-1, 5)
        existence = x[:, :1]
        return ratings, existence