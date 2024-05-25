import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 128)
        self.rating_gen = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(256, 2000),  # Assuming maximum number of rating features
            nn.Softmax(dim=1)
        )
        self.existence_gen = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(128, 2000),  # Assuming maximum number of existence features
            nn.Sigmoid()
        )

    def forward(self, noise, chunk_size):
        x = self.fc1(noise)
        #print(f"Intermediate shape after fc1: {x.shape}")
        ratings = self.rating_gen(x)
        #print(f"Intermediate ratings shape before reshape: {ratings.shape}")
        ratings = ratings[:, :chunk_size]  # Adjust to match the chunk size
        ratings = ratings.view(ratings.size(0), -1)  # Flatten to match Discriminator input
        #print(f"Ratings shape after reshape: {ratings.shape}")
        existence = self.existence_gen(x)
        #print(f"Intermediate existence shape before reshape: {existence.shape}")
        existence = existence[:, :chunk_size]  # Adjust to match the chunk size
        existence = existence.view(existence.size(0), -1)  # Flatten to match Discriminator input
        #print(f"Existence shape after reshape: {existence.shape}")
        combined_output = torch.cat([ratings, existence], dim=1)  # Concatenate along the feature dimension
        #print(f"Generator output shape: {combined_output.shape}")
        return combined_output
