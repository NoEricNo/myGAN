import torch
from torch import nn
import torch
from torch import nn
import torch.distributions as td


class Discriminator(nn.Module):
    def __init__(self, num_movies):
        self.num_movies = num_movies
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(num_movies * 6, 512)  # Input layer to handle embedded rating vectors
        #self.fc2 = nn.Linear(512, 512)  # Additional layer for more nuanced feature extraction

        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        #self.aux_classifier = nn.Sequential(  # Auxiliary classifier for rated/not rated
        #    nn.Linear(512, num_movies),  # Output one logit per movie
        #    nn.Sigmoid()  # Sigmoid to classify rated vs. not rated
        #)

    def forward(self, ratings):
        x = self.fc1(ratings.view(-1, self.num_movies * 6))
        #x = self.fc1(ratings.view(-1, self.num_movies * 6))  # Flatten and input
        #x = self.fc2(x)
        validity = self.main(x)
        #rated_not_rated = self.aux_classifier(x)  # Auxiliary output
        return validity # , rated_not_rated




