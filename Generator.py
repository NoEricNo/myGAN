import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, fc1_size, main_sizes, num_users, num_movies, dropout_rate):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.num_users = num_users
        self.num_movies = num_movies
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.main_layers = nn.ModuleList()
        previous_size = fc1_size
        for size in main_sizes:
            self.main_layers.append(nn.Linear(previous_size, size))
            previous_size = size
        self.fc_ratings = nn.Linear(previous_size, num_movies)
        self.fc_existence = nn.Linear(previous_size, num_movies)

    def forward(self, x, movie_ids):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        for layer in self.main_layers:
            x = torch.relu(layer(x))

        rating_values = self.fc_ratings(x).view(-1, movie_ids.size(1))
        rating_existence = self.fc_existence(x).view(-1, movie_ids.size(1))

        return rating_values.to_sparse(), rating_existence.to_sparse()
