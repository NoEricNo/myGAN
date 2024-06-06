import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import torch

class MovieLensDataset(Dataset):
    def __init__(self, ratings_file):
        self.ratings_file = ratings_file

        # Load data to get the number of users and movies
        df = pd.read_csv(self.ratings_file)
        self.num_users = df['userId'].nunique()
        self.num_movies = df['movieId'].nunique()

        user_indices = df['userId'].values
        original_movie_ids = df['movieId'].values
        rating_values = df['rating'].values

        user_id_mapping = {user_id: idx for idx, user_id in enumerate(np.unique(user_indices))}
        self.user_indices = [user_id_mapping[user_id] for user_id in user_indices]

        movie_id_mapping = {movie_id: idx for idx, movie_id in enumerate(np.unique(original_movie_ids))}
        self.movie_indices = [movie_id_mapping[movie_id] for movie_id in original_movie_ids]

        self.rating_matrix = csr_matrix((rating_values, (self.user_indices, self.movie_indices)),
                                        shape=(self.num_users, self.num_movies))
        self.existence_matrix = (self.rating_matrix > 0).astype(np.float32)

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        user_indices = np.array(self.user_indices)[idx]
        movie_indices = np.array(self.movie_indices)[idx]
        rating_values = self.rating_matrix[idx]
        existence_values = self.existence_matrix[idx]

        return rating_values, existence_values, user_indices, movie_indices

def sparse_collate(batch):
    data_list, existence_list, user_list, movie_list = zip(*batch)

    data_tensors = [torch.tensor(data.toarray(), dtype=torch.float32).to_sparse() for data in data_list]
    existence_tensors = [torch.tensor(existence.toarray(), dtype=torch.float32).to_sparse() for existence in existence_list]
    user_tensors = [torch.tensor(users, dtype=torch.float32) for users in user_list]
    movie_tensors = [torch.tensor(movies, dtype=torch.float32) for movies in movie_list]

    return (
        torch.stack(data_tensors),
        torch.stack(existence_tensors),
        torch.stack(user_tensors),
        torch.stack(movie_tensors)
    )
