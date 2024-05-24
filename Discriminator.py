import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, num_movies):
        super(Discriminator, self).__init__()
        self.num_movies = num_movies
        self.fc1 = nn.Linear(num_movies * 6, 256)  # Reduce size
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, 128),  # Reduce size
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),
        )
        self.rating_output = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.existence_output = nn.Sequential(
            nn.Linear(128, num_movies),
            nn.Sigmoid()
        )

    def forward(self, ratings):
        x = ratings.reshape(ratings.size(0), -1)
        x = self.fc1(x)
        x = self.main(x)
        rating_validity = self.rating_output(x)
        existence_prediction = self.existence_output(x)
        return rating_validity, existence_prediction
