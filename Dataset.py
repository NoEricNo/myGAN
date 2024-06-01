import torch
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

class MovieLensDataLoader:
    def __init__(self, ratings_file, batch_size, chunk_size):
        self.ratings_file = ratings_file
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.dataset = MovieLensDataset(ratings_file, chunk_size)
        self.num_chunks = self.dataset.num_chunks
        self.current_chunk = 0

    def get_chunk(self, chunk_idx):
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.dataset.rating_matrix.size(0))
        chunk = self.dataset.rating_matrix[start_idx:end_idx]
        existence_chunk = self.dataset.existence_matrix[start_idx:end_idx]
        return chunk, existence_chunk


class MovieLensDataset:
    def __init__(self, ratings_file, chunk_size):
        self.rating_matrix, self.existence_matrix, self.num_movies = self.load_data(ratings_file)
        self.chunk_size = chunk_size
        self.num_chunks = (self.rating_matrix.size(0) + chunk_size - 1) // chunk_size

    def load_data(self, ratings_file):
        print(f"Loading data from {ratings_file}")
        ratings = pd.read_csv(ratings_file)
        print(f"Initial ratings data shape: {ratings.shape}")

        user_indices = ratings['userId'].values - 1
        movie_indices = ratings['movieId'].values - 1
        rating_values = ratings['rating'].values

        print(f"user_indices: {user_indices[:50]}")  # Print the first 50 for debugging
        print(f"movie_indices: {movie_indices[:50]}")  # Print the first 50 for debugging
        print(f"rating_values: {rating_values[:50]}")  # Print the first 50 for debugging

        # Additional debug prints
        print(f"user_indices shape: {user_indices.shape}")
        print(f"movie_indices shape: {movie_indices.shape}")
        print(f"rating_values shape: {rating_values.shape}")

        assert len(user_indices) == len(movie_indices) == len(rating_values), "Lengths of indices and ratings do not match"

        user_item_matrix = coo_matrix((rating_values, (user_indices, movie_indices)),
                                      shape=(ratings['userId'].nunique(), ratings['movieId'].nunique()))
        print(f"user_item_matrix shape: {user_item_matrix.shape}")

        rating_matrix = torch.tensor(user_item_matrix.toarray(), dtype=torch.float32)
        existence_matrix = (rating_matrix > 0).float()

        print(f"rating_matrix shape: {rating_matrix.shape}")
        print(f"existence_matrix shape: {existence_matrix.shape}")

        return rating_matrix, existence_matrix, ratings['movieId'].nunique()
