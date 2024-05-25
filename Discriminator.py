import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),
        )
        self.rating_output = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.existence_output = nn.Sequential(
            nn.Linear(256, input_size // 2),  # Adjusted based on features
            nn.Sigmoid()
        )

    def forward(self, ratings):
        #print(f"Discriminator received input shape: {ratings.shape}")
        x = ratings.view(ratings.size(0), -1)  # Flatten the input
        #print(f"Discriminator input shape after flattening: {x.shape}")
        x = self.fc1(x)
        #print(f"Shape after fc1: {x.shape}")
        x = self.main(x)
        #print(f"Shape after main sequential: {x.shape}")
        rating_validity = self.rating_output(x)
        #print(f"Shape of rating_validity: {rating_validity.shape}")
        existence_prediction = self.existence_output(x)
        #print(f"Shape of existence_prediction: {existence_prediction.shape}")
        return rating_validity, existence_prediction
