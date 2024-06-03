import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, fc1_size, dropout_rate):
        super(Discriminator, self).__init__()
        self.fc1_size = fc1_size
        self.dropout_rate = dropout_rate
        self.fc1 = None
        self.fc2 = nn.Linear(fc1_size, 128)
        self.fc3 = nn.Linear(128, 64)
        self.validity = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_fc1(self, input_size):
        self.fc1 = nn.Linear(input_size, self.fc1_size).to(self.device)

    def forward(self, rating_values, rating_existence, user_ids, movie_ids):
        # Flatten the inputs
        batch_size = rating_values.size(0)
        num_movies_chunk = rating_values.size(1)

        # Adjust the expected input size for the chunked input
        expected_input_size = (num_movies_chunk * 2) + 2

        rating_values = rating_values.view(batch_size, -1).float().to(self.device)
        rating_existence = rating_existence.view(batch_size, -1).float().to(self.device)
        user_ids = user_ids.view(batch_size, -1).float().to(self.device)
        movie_ids = movie_ids.view(batch_size, -1).float().to(self.device)

        # Debug prints
        print(f"Input rating_values shape: {rating_values.shape}")
        print(f"Input rating_existence shape: {rating_existence.shape}")
        print(f"Input user_ids shape: {user_ids.shape}")
        print(f"Input movie_ids shape: {movie_ids.shape}")

        # Concatenate all inputs to form the final input to the discriminator
        x = torch.cat((rating_values, rating_existence, user_ids, movie_ids), dim=1)
        print(f"Concatenated input shape: {x.shape}")

        # Initialize the first fully connected layer with the correct input size if not initialized or if size changed
        if self.fc1 is None or self.fc1.in_features != expected_input_size:
            self.initialize_fc1(expected_input_size)

        x = torch.relu(self.fc1(x))
        print(f"After fc1 shape: {x.shape}")

        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        validity = self.validity(x)
        return validity
