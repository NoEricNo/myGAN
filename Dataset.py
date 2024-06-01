import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.sparse import coo_matrix

class MovieLensDataset(Dataset):
    def __init__(self, ratings_file, chunk_size):
        self.rating_matrix, self.existence_matrix, self.num_movies = self.load_data(ratings_file)
        self.chunk_size = chunk_size

    def load_data(self, ratings_file):
        print(f"Loading data from {ratings_file}")
        ratings = pd.read_csv(ratings_file)
        print(f"Initial ratings data shape: {ratings.shape}")

        user_indices = ratings['userId'].values
        original_movie_ids = ratings['movieId'].values
        rating_values = ratings['rating'].values

        num_users = ratings['userId'].nunique()
        num_movies = len(np.unique(original_movie_ids))

        print(f"Number of users: {num_users}")
        print(f"Number of movies: {num_movies}")

        user_id_mapping = {user_id: idx for idx, user_id in enumerate(np.unique(user_indices))}
        user_indices = [user_id_mapping[user_id] for user_id in user_indices]

        movie_id_mapping = {movie_id: idx for idx, movie_id in enumerate(np.unique(original_movie_ids))}
        movie_indices = [movie_id_mapping[movie_id] for movie_id in original_movie_ids]

        print(f"user_indices: {user_indices[:50]}")
        print(f"movie_indices: {movie_indices[:50]}")
        print(f"rating_values: {rating_values[:50]}")

        print(f"user_indices shape: {len(user_indices)}")
        print(f"movie_indices shape: {len(movie_indices)}")
        print(f"rating_values shape: {rating_values.shape}")

        max_user_id = max(user_indices)
        max_movie_id = max(movie_indices)
        print(f"Max user index: {max_user_id}, Max movie index: {max_movie_id}")

        assert len(user_indices) == len(movie_indices) == len(
            rating_values), "Lengths of indices and ratings do not match"
        assert max_user_id < num_users, "Max user index exceeds number of users"
        assert max_movie_id < num_movies, "Max movie index exceeds number of movies"

        user_item_matrix = coo_matrix((rating_values, (user_indices, movie_indices)),
                                      shape=(num_users, num_movies))
        print(f"user_item_matrix shape: {user_item_matrix.shape}")

        rating_matrix = torch.tensor(user_item_matrix.toarray(), dtype=torch.float32)
        existence_matrix = (rating_matrix > 0).float()

        print(f"rating_matrix shape: {rating_matrix.shape}")
        print(f"existence_matrix shape: {existence_matrix.shape}")

        return rating_matrix, existence_matrix, num_movies

    def __len__(self):
        return (self.rating_matrix.size(0) + self.chunk_size - 1) // self.chunk_size

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.rating_matrix.size(0))
        data_chunk = self.rating_matrix[start_idx:end_idx]
        existence_chunk = self.existence_matrix[start_idx:end_idx]
        return data_chunk, existence_chunk

class MovieLensDataLoader:
    def __init__(self, ratings_file, batch_size, chunk_size):
        self.dataset = MovieLensDataset(ratings_file, chunk_size)
        self.batch_size = batch_size

    def __iter__(self):
        for chunk_idx in range(len(self.dataset)):
            data_chunk, existence_chunk = self.dataset[chunk_idx]
            chunk_loader = DataLoader(list(zip(data_chunk, existence_chunk)), batch_size=self.batch_size, shuffle=True)
            yield chunk_loader

    def __len__(self):
        return len(self.dataset)
