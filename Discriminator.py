import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ratings_size, existence_size, user_id_size, movie_id_size, fc1_size, main_sizes, dropout_rate):
        super(Discriminator, self).__init__()
        input_size = ratings_size + existence_size + user_id_size + movie_id_size
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.main = nn.Sequential(
            nn.Linear(fc1_size, main_sizes[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(main_sizes[0], main_sizes[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        self.validity = nn.Linear(main_sizes[1], 1)
        self.existence = nn.Linear(main_sizes[1], ratings_size)  # Update to correct size

    def forward(self, ratings, existence, user_ids, movie_ids):
        ratings = ratings.float()
        existence = existence.float()
        user_ids = user_ids.float().view(-1, 1)  # Ensure 2D without repeating
        movie_ids = movie_ids.float().view(-1, 1)  # Ensure 2D without repeating
        x = torch.cat((ratings, existence, user_ids, movie_ids), dim=1)
        print(f"Concatenated input shape: {x.shape}")
        x = torch.relu(self.fc1(x))
        x = self.main(x)
        validity = self.validity(x)
        existence_pred = self.existence(x)
        return validity, existence_pred



