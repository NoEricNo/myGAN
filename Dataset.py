import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.sparse import csr_matrix

class MovieLensDataset(Dataset):
    def __init__(self, ratings_file, chunk_size):
        self.rating_matrix, self.existence_matrix, self.num_movies, self.user_id_matrix, self.movie_id_matrix, self.num_users = self.load_data(ratings_file)
        self.chunk_size = chunk_size
        self.num_chunks = (self.rating_matrix.shape[0] + chunk_size - 1) // chunk_size

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

        rating_matrix = csr_matrix((rating_values, (user_indices, movie_indices)), shape=(num_users, num_movies)).toarray()
        existence_matrix = (rating_matrix > 0).astype(np.float32)

        print(f"rating_matrix shape: {rating_matrix.shape}")
        print(f"existence_matrix shape: {existence_matrix.shape}")

        rating_matrix[rating_matrix == 0] = -9999  # Replace missing ratings with -9999
        existence_matrix = (rating_matrix != -9999).astype(np.float32)  # Create existence matrix

        user_id_matrix = np.array(user_indices).reshape(-1, 1).astype(np.float32)
        movie_id_matrix = np.array(movie_indices).reshape(-1, 1).astype(np.float32)

        return rating_matrix, existence_matrix, num_movies, user_id_matrix, movie_id_matrix, num_users

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.rating_matrix.shape[0])
        data_chunk = self.rating_matrix[start_idx:end_idx]
        existence_chunk = self.existence_matrix[start_idx:end_idx]
        user_id_chunk = self.user_id_matrix[start_idx:end_idx]
        movie_id_chunk = self.movie_id_matrix[start_idx:end_idx]
        return data_chunk, existence_chunk, user_id_chunk, movie_id_chunk


class MovieLensDataLoader:
    def __init__(self, ratings_file, batch_size, chunk_size):
        self.dataset = MovieLensDataset(ratings_file, chunk_size)
        self.batch_size = batch_size

    def __iter__(self):
        for chunk_idx in range(len(self.dataset)):
            data_chunk, existence_chunk, user_id_chunk, movie_id_chunk = self.dataset[chunk_idx]
            chunk_loader = DataLoader(list(zip(data_chunk, existence_chunk, user_id_chunk, movie_id_chunk)),
                                      batch_size=self.batch_size, shuffle=True)
            yield chunk_loader

    def __len__(self):
        return len(self.dataset)
