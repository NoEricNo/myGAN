import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, noise_dim, fc1_size, rating_gen_sizes, existence_gen_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, fc1_size)
        self.rating_gen = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(fc1_size, rating_gen_sizes[0]),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(rating_gen_sizes[0], rating_gen_sizes[1]),
            nn.Softmax(dim=1)
        )
        self.existence_gen = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(fc1_size, existence_gen_size),
            nn.Sigmoid()
        )

    def forward(self, noise, chunk_size):
        x = self.fc1(noise)
        ratings = self.rating_gen(x)
        ratings = ratings[:, :chunk_size]  # Adjust to match the chunk size
        ratings = ratings.view(ratings.size(0), -1)  # Flatten to match Discriminator input
        existence = self.existence_gen(x)
        existence = existence[:, :chunk_size]  # Adjust to match the chunk size
        existence = existence.view(existence.size(0), -1)  # Flatten to match Discriminator input
        combined_output = torch.cat([ratings, existence], dim=1)  # Concatenate along the feature dimension
        return combined_output
