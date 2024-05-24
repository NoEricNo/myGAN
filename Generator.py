import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, num_movies, noise_dim=100):
        super(Generator, self).__init__()
        self.num_movies = num_movies
        self.fc1 = nn.Linear(noise_dim, 64)  # Reduced size
        self.rating_gen = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(64, 128),  # Reduced size
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(128, num_movies * 5),
            nn.Softmax(dim=1)  # Use dim=1 for the correct dimension
        )
        self.existence_gen = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(64, num_movies),
            nn.Sigmoid()
        )

    def forward(self, noise):
        x = self.fc1(noise)
        ratings = self.rating_gen(x).reshape(-1, self.num_movies, 5)
        existence = self.existence_gen(x).reshape(-1, self.num_movies, 1)
        combined_output = torch.cat([ratings, existence], dim=2)
        return combined_output
