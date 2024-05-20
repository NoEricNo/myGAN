import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MovieLensDataset(Dataset):
    def __init__(self, ratings_file):
        self.ratings_file = ratings_file
        self.rating_matrix, self.existence_matrix = self.load_data()
        self.num_movies = self.rating_matrix.shape[1]

    def load_data(self):
        # Load the ratings data
        ratings = pd.read_csv(self.ratings_file)

        # Create a user-item matrix
        user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

        # Fill missing values with 0 (no rating)
        user_item_matrix = user_item_matrix.fillna(0)

        # Extract the existence matrix (binary)
        existence_matrix = (user_item_matrix > 0).astype(float)

        # Extract the rating matrix
        rating_matrix = user_item_matrix

        return rating_matrix, existence_matrix

    def __len__(self):
        return len(self.rating_matrix)

    def __getitem__(self, idx):
        ratings = self.rating_matrix.iloc[idx].values
        existence = self.existence_matrix.iloc[idx].values

        # Convert to torch tensors
        ratings_tensor = torch.tensor(ratings, dtype=torch.float32).unsqueeze(-1)  # Shape: (num_movies, 1)
        existence_tensor = torch.tensor(existence, dtype=torch.float32).unsqueeze(-1)  # Shape: (num_movies, 1)

        # One-hot encode the ratings
        one_hot_ratings = torch.zeros((ratings_tensor.size(0), 5), dtype=torch.float32)
        for i, rating in enumerate(ratings_tensor.squeeze(-1)):
            if rating > 0:
                one_hot_ratings[i, int(rating) - 1] = 1.0  # Ratings are 1-5

        combined_tensor = torch.cat([one_hot_ratings, existence_tensor], dim=1)  # Shape: (num_movies, 6)

        # Add debug prints to verify shapes
        #print(f"ratings_tensor shape---: {ratings_tensor.shape}")
        #print(f"one_hot_ratings shape---: {one_hot_ratings.shape}")
        #print(f"combined_tensor shape---: {combined_tensor.shape}")

        return combined_tensor, existence_tensor


class MovieLensDataLoader:
    def __init__(self, ratings_file, batch_size):
        self.dataset = MovieLensDataset(ratings_file)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def get_loader(self):
        return self.dataloader

    def get_num_movies(self):
        return self.dataset.num_movies
