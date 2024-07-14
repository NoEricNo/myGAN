import torch
import torch.nn as nn

class MainDiscriminator(nn.Module):
    def __init__(self, num_movies, dropout_rate, fc1_size, fc2_size, fc3_size):
        super(MainDiscriminator, self).__init__()
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

        x = x.float()  # Ensure x is of type Float
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        validity = self.validity(x)

        return validity

import torch
import torch.nn as nn

class DistributionDiscriminator(nn.Module):
    def __init__(self, num_movies, dropout_rate, fc1_size, fc2_size, fc3_size):
        super(DistributionDiscriminator, self).__init__()
        self.num_movies = num_movies
        self.dropout_rate = dropout_rate

        # We'll use summary statistics as input
        input_size = 4  # mean, std, skewness, kurtosis

        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.validity = nn.Linear(fc3_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def calculate_stats(self, ratings):
        mean = torch.mean(ratings, dim=1)
        std = torch.std(ratings, dim=1, unbiased=False)
        skewness = torch.mean(((ratings - mean.unsqueeze(1)) / (std.unsqueeze(1) + 1e-6)) ** 3, dim=1)
        kurtosis = torch.mean(((ratings - mean.unsqueeze(1)) / (std.unsqueeze(1) + 1e-6)) ** 4, dim=1)
        return torch.stack([mean, std, skewness, kurtosis], dim=1)

    def forward(self, real_ratings, fake_ratings):
        real_ratings = real_ratings.squeeze(1)
        fake_ratings = fake_ratings.squeeze(1)

        real_stats = self.calculate_stats(real_ratings)
        fake_stats = self.calculate_stats(fake_ratings)

        # Ensure both tensors have the same number of dimensions
        if real_stats.dim() == 2 and fake_stats.dim() == 2:
            x = torch.cat([real_stats, fake_stats], dim=0)
        else:
            raise ValueError(f"Dimension mismatch: real_stats.dim()={real_stats.dim()}, fake_stats.dim()={fake_stats.dim()}")

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        validity = self.validity(x)

        return validity




class LatentFactorDiscriminator(nn.Module):
    def __init__(self, num_movies, latent_dim, dropout_rate, fc1_size, fc2_size, fc3_size):
        super(LatentFactorDiscriminator, self).__init__()
        self.num_movies = num_movies
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        input_size = latent_dim  # Ensure this matches the latent dimension size

        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.validity = nn.Linear(fc3_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, fake_ratings, item_factors):
        fake_ratings = fake_ratings.to_dense().squeeze(1)

        # Transpose item_factors for matrix multiplication
        item_factors = item_factors.T

        # Calculate the alignment with item factors
        alignment = fake_ratings @ item_factors

        x = alignment.float()
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        validity = self.validity(x)

        return validity

