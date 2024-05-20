import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, num_movies):
        super(Discriminator, self).__init__()
        self.num_movies = num_movies
        self.fc1 = nn.Linear(num_movies * 5, 512)
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),  # Change inplace to False
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=False),  # Change inplace to False
            nn.Dropout(0.3),
        )
        self.rating_output = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.existence_output = nn.Sequential(
            nn.Linear(256, num_movies),
            nn.Sigmoid()
        )

    def forward(self, ratings):
        #print(f"ratings shape before reshape: {ratings.shape}")
        x = ratings.reshape(ratings.size(0), -1)
        #print(f"x shape after reshape: {x.shape}")
        x = self.fc1(x)
        x = self.main(x)
        rating_validity = self.rating_output(x)
        existence_prediction = self.existence_output(x)
        return rating_validity, existence_prediction
