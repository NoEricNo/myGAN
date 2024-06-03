import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix


def divide_users(num_users, num_groups):
    chunk_size = (num_users + num_groups - 1) // num_groups
    user_chunks = []
    for start in range(0, num_users, chunk_size):
        end = min(start + chunk_size, num_users)
        user_chunks.append((start, end))
    return user_chunks


def create_overlapping_chunks(num_items, num_groups, overlap_ratio):
    chunk_size = (num_items + num_groups - 1) // num_groups
    overlap_size = int(chunk_size * overlap_ratio)
    step_size = chunk_size - overlap_size
    chunks = []
    for start in range(0, num_items, step_size):
        end = min(start + chunk_size, num_items)
        chunks.append((start, end))
    return chunks


class MovieLensDataset(Dataset):
    def __init__(self, ratings_file, num_user_groups, num_movie_groups, movie_overlap_ratio):
        self.ratings_file = ratings_file
        self.num_user_groups = num_user_groups
        self.num_movie_groups = num_movie_groups
        self.movie_overlap_ratio = movie_overlap_ratio

        # Load data to get the number of users and movies
        df = pd.read_csv(self.ratings_file)
        self.num_users = df['userId'].nunique()
        self.num_movies = df['movieId'].nunique()

        self.user_chunks = self._create_user_chunks()
        self.movie_chunks = self._create_movie_chunks()

    def _create_user_chunks(self):
        return divide_users(self.num_users, self.num_user_groups)

    def _create_movie_chunks(self):
        return create_overlapping_chunks(self.num_movies, self.num_movie_groups, self.movie_overlap_ratio)

    def __len__(self):
        return len(self.user_chunks) * len(self.movie_chunks)

    def __getitem__(self, idx):
        user_chunk_idx = idx // len(self.movie_chunks)
        movie_chunk_idx = idx % len(self.movie_chunks)

        user_start, user_end = self.user_chunks[user_chunk_idx]
        movie_start, movie_end = self.movie_chunks[movie_chunk_idx]

        df = pd.read_csv(self.ratings_file)
        df = df[(df['userId'] >= user_start) & (df['userId'] < user_end) &
                (df['movieId'] >= movie_start) & (df['movieId'] < movie_end)]

        user_indices = df['userId'].values
        original_movie_ids = df['movieId'].values
        rating_values = df['rating'].values

        num_users = len(np.unique(user_indices))
        num_movies = len(np.unique(original_movie_ids))

        user_id_mapping = {user_id: idx for idx, user_id in enumerate(np.unique(user_indices))}
        user_indices = [user_id_mapping[user_id] for user_id in user_indices]

        movie_id_mapping = {movie_id: idx for idx, movie_id in enumerate(np.unique(original_movie_ids))}
        movie_indices = [movie_id_mapping[movie_id] for movie_id in original_movie_ids]

        rating_matrix = csr_matrix((rating_values, (user_indices, movie_indices)),
                                   shape=(num_users, num_movies)).toarray()
        existence_matrix = (rating_matrix > 0).astype(np.float32)

        rating_matrix[rating_matrix == 0] = -9999  # Replace missing ratings with -9999
        existence_matrix = (rating_matrix != -9999).astype(np.float32)  # Create existence matrix

        user_id_matrix = np.array(user_indices).reshape(-1, 1).astype(np.float32)
        movie_id_matrix = np.array(movie_indices).reshape(-1, 1).astype(np.float32)

        return rating_matrix, existence_matrix, user_id_matrix, movie_id_matrix


class MovieLensDataLoader:
    def __init__(self, ratings_file, batch_size, num_user_groups, num_movie_groups, movie_overlap_ratio):
        self.dataset = MovieLensDataset(ratings_file, num_user_groups, num_movie_groups, movie_overlap_ratio)
        self.batch_size = batch_size

    def __iter__(self):
        for block_idx in range(len(self.dataset)):
            data_chunk, existence_chunk, user_id_chunk, movie_id_chunk = self.dataset[block_idx]
            print(
                f"Chunk {block_idx} sizes - Data: {data_chunk.shape}, Existence: {existence_chunk.shape}, Users: {user_id_chunk.shape}, Movies: {movie_id_chunk.shape}")
            if data_chunk.size > 0:  # Ensure non-empty chunks
                chunk_loader = DataLoader(list(zip(data_chunk, existence_chunk, user_id_chunk, movie_id_chunk)),
                                          batch_size=self.batch_size, shuffle=True)
                yield chunk_loader

    def __len__(self):
        return len(self.dataset)
