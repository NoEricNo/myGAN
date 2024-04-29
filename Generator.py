import torch
from torch import nn
import torch.distributions as td

class Generator(nn.Module):
    def __init__(self, num_movies, noise_dim=100):
        super(Generator, self).__init__()
        self.num_movies = num_movies
        self.fc1 = nn.Linear(noise_dim, 256)
        self.rating_gen = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, num_movies * 5),
            nn.Softmax(dim=2)
        )
        # Existence of rating (binary output)
        self.existence_gen = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_movies),
            nn.Sigmoid()
        )

    def forward(self, noise):
        x = self.fc1(noise)
        ratings = self.rating_gen(x).view(-1, self.num_movies, 5)
        existence = self.existence_gen(x).view(-1, self.num_movies, 1)
        combined_output = torch.cat([ratings, existence], dim=2)
        return combined_output



#model = Generator()
#summary(model, input_size=(100,))
