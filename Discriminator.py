import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, num_movies):
        super(Discriminator, self).__init__()
        self.num_movies = num_movies
        self.fc1 = nn.Linear(num_movies * 5, 512)  # Adjusted to num_movies * 5 to match the input dimension
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, ratings):
        x = self.fc1(ratings.reshape(ratings.size(0), -1))  # Flatten the input correctly using reshape
        validity = self.main(x)
        return validity
