import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import coo_matrix

class MovieLensDataset(Dataset):
    def __init__(self, ratings_file):
        self.ratings_file = ratings_file
        self.rating_matrix, self.existence_matrix = self.load_data()
        self.num_movies = self.rating_matrix.shape[1]

    def load_data(self):
        # Load the ratings data
        ratings = pd.read_csv(self.ratings_file)

        # Create the user-item matrix in a sparse format
        user_item_matrix = coo_matrix((ratings['rating'], (ratings['userId'], ratings['movieId'])))

        # Extract the existence matrix (binary)
        existence_matrix = (user_item_matrix > 0).astype(float)

        # Create the existence matrix in a sparse format
        existence_matrix = coo_matrix((np.ones_like(ratings['rating']), (ratings['userId'], ratings['movieId'])))

        # Extract the rating matrix
        rating_matrix = user_item_matrix

        return rating_matrix, existence_matrix

    def __len__(self):
        return self.rating_matrix.shape[0]

    def __getitem__(self, idx):
        ratings = self.rating_matrix.getrow(idx).toarray().squeeze().astype(np.float32)  # Convert to float32
        existence = self.existence_matrix.getrow(idx).toarray().squeeze().astype(np.float32)  # Convert to float32

        # One-hot encode the ratings
        one_hot_ratings = np.zeros((len(ratings), 5), dtype=np.float32)
        for i, rating in enumerate(ratings):
            if rating > 0:
                one_hot_ratings[i, int(rating) - 1] = 1.0  # Ratings are 1-5

        # Convert to tensors
        ratings_tensor = torch.tensor(one_hot_ratings, dtype=torch.float32)  # Ensure float32
        existence_tensor = torch.tensor(existence, dtype=torch.float32).unsqueeze(-1)  # Ensure float32

        combined_tensor = torch.cat([ratings_tensor, existence_tensor], dim=1)

        return combined_tensor


class MovieLensDataLoader:
    def __init__(self, ratings_file, batch_size):
        self.dataset = MovieLensDataset(ratings_file)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def get_loader(self):
        return self.dataloader

    def get_num_movies(self):
        return self.dataset.rating_matrix.shape[1]
