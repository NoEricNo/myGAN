import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_users, num_movies, fc1_size, dropout_rate):
        super(Discriminator, self).__init__()
        input_size = (num_movies * 2) + 2  # Two channels: rating values and rating existence, plus user_id and movie_id
        print(f"Discriminator input_size: {input_size}")
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, 128)  # Simplified to a smaller layer
        self.fc3 = nn.Linear(128, 64)  # Further simplified to a smaller layer
        self.validity = nn.Linear(64, 1)  # Output layer for validity
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, rating_values, rating_existence, user_ids, movie_ids):
        print(f"Input rating_values shape: {rating_values.shape}")
        print(f"Input rating_existence shape: {rating_existence.shape}")
        print(f"Input user_ids shape: {user_ids.shape}")
        print(f"Input movie_ids shape: {movie_ids.shape}")

        rating_values = rating_values.view(rating_values.size(0), -1).float()
        rating_existence = rating_existence.view(rating_existence.size(0), -1).float()
        user_ids = user_ids.view(user_ids.size(0), -1).float()
        movie_ids = movie_ids.view(movie_ids.size(0), -1).float()

        print(f"Reshaped rating_values shape: {rating_values.shape}")
        print(f"Reshaped rating_existence shape: {rating_existence.shape}")
        print(f"Reshaped user_ids shape: {user_ids.shape}")
        print(f"Reshaped movie_ids shape: {movie_ids.shape}")

        x = torch.cat((rating_values, rating_existence, user_ids, movie_ids), dim=1)
        print(f"Concatenated input shape: {x.shape}")

        x = torch.relu(self.fc1(x))
        print(f"After fc1 shape: {x.shape}")

        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        validity = self.validity(x)
        return validity
