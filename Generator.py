import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, num_movies, noise_dim=100):
        super(Generator, self).__init__()
        self.num_movies = num_movies
        self.fc1 = nn.Linear(noise_dim, 128)
        self.rating_gen = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),  # Change inplace to False
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=False),  # Change inplace to False
            nn.Linear(256, num_movies * 5),
            nn.Softmax(dim=1)  # Change dim to 1
        )
        self.existence_gen = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),  # Change inplace to False
            nn.Linear(128, num_movies),
            nn.Sigmoid()
        )

    def forward(self, noise):
        x = self.fc1(noise)
        ratings = self.rating_gen(x).reshape(-1, self.num_movies, 5)
        existence = self.existence_gen(x).reshape(-1, self.num_movies, 1)
        combined_output = torch.cat([ratings, existence], dim=2)
        return combined_output
