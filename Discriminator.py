import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_movies, dropout_rate, fc1_size, fc2_size, fc3_size):
        super(Discriminator, self).__init__()
        self.num_movies = num_movies
        self.dropout_rate = dropout_rate

        # The input size for fc1 is 2 * num_movies since we concatenate rating values and existence flags
        input_size = 2 * num_movies

        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.validity = nn.Linear(fc3_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, rating_values, existence_flags):
        # Ensure rating_values and existence_flags are dense tensors
        rating_values = rating_values.to_dense()
        existence_flags = existence_flags.to_dense()

        # Remove the extra dimension
        rating_values = rating_values.squeeze(1)
        existence_flags = existence_flags.squeeze(1)

        # Concatenate rating_values and existence_flags along the last dimension
        x = torch.cat([rating_values, existence_flags], dim=1)  # Shape: (batch_size, 2 * num_movies)

        # Print shapes for debugging
        #print("rating_values shape:", rating_values.shape)
        #print("existence_flags shape:", existence_flags.shape)
        #print("x shape:", x.shape)

        x = x.float()  # Ensure x is of type Float
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        validity = self.validity(x)

        return validity
