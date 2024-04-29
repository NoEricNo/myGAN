import torch
from torch import nn
import torch.distributions as td

class Generator(nn.Module):
    def __init__(self, num_movies, noise_dim=100, use_sampling=False):
        super(Generator, self).__init__()
        self.num_movies = num_movies
        self.use_sampling = use_sampling  # Flag to choose between sampling and max probability
        self.fc1 = nn.Linear(noise_dim, 256)
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 5 * self.num_movies),  # Output probabilities for 5 ratings per movie
            nn.Softmax(dim=1)  # Ensures the output is distributed like probabilities
        )

    def forward(self, noise):
        x = self.fc1(noise)
        ratings_probs = self.main(x)
        ratings_probs = ratings_probs.view(-1, self.num_movies, 6)  #  Including a 'no rating' class

        if self.use_sampling:
            # Sample from the probability distribution to obtain discrete ratings
            ratings_dist = td.Categorical(ratings_probs)
            ratings = ratings_dist.sample().detach() + 1  # Add 1 to convert index to rating (1 to 5)
        else:
            # Map probabilities to discrete ratings based on maximum probability
            ratings = torch.argmax(ratings_probs, dim=2) + 1  # Add 1 to convert index to rating (1 to 5)

        return ratings



#model = Generator()
#summary(model, input_size=(100,))
