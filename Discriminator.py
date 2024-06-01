import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_size, fc1_size, main_sizes, dropout_rate):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, fc1_size)
        self.main = nn.Sequential(
            nn.Linear(fc1_size, main_sizes[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(main_sizes[0], main_sizes[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        self.validity = nn.Linear(main_sizes[1], 1)
        self.existence = nn.Linear(main_sizes[1], input_size)  # Modify this line to match the shape

    def forward(self, ratings, existence):
        x = torch.cat((ratings, existence), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.main(x)
        validity = self.validity(x)
        existence_pred = self.existence(x)
        return validity, existence_pred
