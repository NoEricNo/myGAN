import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, dropout_rate):
        super(Discriminator, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(2, 128)  # Adjust this as needed
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.validity = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, rating_values, existence_flags, movie_ids):
        batch_size = rating_values.size(0)
        num_movies = rating_values.size(1)

        rating_values = rating_values.to_dense()
        existence_flags = existence_flags.to_dense()

        x = torch.stack([rating_values, existence_flags], dim=2).view(batch_size, -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        validity = self.validity(x)

        return validity
