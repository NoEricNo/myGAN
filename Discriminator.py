import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_size, fc1_size, main_sizes, existence_output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),
            nn.Linear(fc1_size, main_sizes[0]),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),
            nn.Linear(main_sizes[0], main_sizes[1]),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),
        )
        self.rating_output = nn.Sequential(
            nn.Linear(main_sizes[1], 1),
            nn.Sigmoid()
        )
        self.existence_output = nn.Sequential(
            nn.Linear(main_sizes[1], existence_output_size),
            nn.Sigmoid()
        )

    def forward(self, ratings):
        x = ratings.view(ratings.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.main(x)
        rating_validity = self.rating_output(x)
        existence_prediction = self.existence_output(x)
        return rating_validity, existence_prediction
