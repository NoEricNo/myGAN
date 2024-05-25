import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import coo_matrix, csr_matrix

class MovieLensDataset(Dataset):
    def __init__(self, ratings_file, chunk_size):
        self.ratings_file = ratings_file
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rating_matrix, self.existence_matrix, self.num_movies = self.load_data()
        self.current_chunk_ratings = None
        self.current_chunk_existence = None

    def load_data(self):
        ratings = pd.read_csv(self.ratings_file)
        user_item_matrix = coo_matrix((ratings['rating'], (ratings['userId'] - 1, ratings['movieId'] - 1)),
                                      shape=(ratings['userId'].max(), ratings['movieId'].max())).tocsr()
        existence_matrix = coo_matrix((ratings['rating'] > 0, (ratings['userId'] - 1, ratings['movieId'] - 1)),
                                      shape=(ratings['userId'].max(), ratings['movieId'].max())).tocsr()
        return user_item_matrix, existence_matrix, ratings['movieId'].max()

    def set_chunk(self, chunk_idx):
        start_idx = chunk_idx * self.chunk_size
        end_idx = min((chunk_idx + 1) * self.chunk_size, self.num_movies)
        self.current_chunk_ratings = torch.tensor(self.rating_matrix[:, start_idx:end_idx].toarray(), dtype=torch.float32).to(self.device)
        self.current_chunk_existence = torch.tensor(self.existence_matrix[:, start_idx:end_idx].toarray(), dtype=torch.float32).to(self.device)
        #print(f"Set chunk {chunk_idx}: Ratings shape {self.current_chunk_ratings.shape}, Existence shape {self.current_chunk_existence.shape}")

    def __len__(self):
        return self.current_chunk_ratings.shape[0]

    def __getitem__(self, idx):
        return self.current_chunk_ratings[idx], self.current_chunk_existence[idx]

class MovieLensDataLoader:
    def __init__(self, ratings_file, batch_size, chunk_size):
        self.dataset = MovieLensDataset(ratings_file, chunk_size)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.num_chunks = (self.dataset.num_movies + chunk_size - 1) // chunk_size

    def set_chunk(self, chunk_idx):
        self.dataset.set_chunk(chunk_idx)

    def get_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
