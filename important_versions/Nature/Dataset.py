import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

class MovieLensDataset(Dataset):
    def __init__(self, ratings_file, num_clusters=5):
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

        # Sort movies based on popularity (number of ratings)
        movie_popularity = df.groupby('movieId').size().reset_index(name='count')
        movie_popularity = movie_popularity.sort_values(by='count', ascending=False)
        sorted_movie_ids = movie_popularity['movieId'].values

        movie_id_mapping = {movie_id: idx for idx, movie_id in enumerate(sorted_movie_ids)}
        self.movie_indices = [movie_id_mapping[movie_id] for movie_id in original_movie_ids]

        self.rating_matrix = csr_matrix((rating_values, (self.user_indices, self.movie_indices)),
                                        shape=(self.num_users, self.num_movies))
        self.existence_matrix = (self.rating_matrix > 0).astype(np.float32)

        # Perform user segmentation using KMeans
        self.num_clusters = num_clusters
        self.user_clusters = self.perform_clustering()

    def perform_clustering(self):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        user_profiles = self.rating_matrix.toarray()
        user_clusters = kmeans.fit_predict(user_profiles)
        return user_clusters

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):

        rating_values = self.rating_matrix[idx]
        existence_values = self.existence_matrix[idx]
        user_cluster = self.user_clusters[idx]

        return rating_values, existence_values, user_cluster

def sparse_collate(batch):
    data_list, existence_list, cluster_list = zip(*batch)

    # Ensure the data_list and existence_list are sparse tensors
    data_tensors = [torch.tensor(data.toarray()).to_sparse() for data in data_list]
    existence_tensors = [torch.tensor(existence.toarray()).to_sparse() for existence in existence_list]

    # Stack the sparse tensors without converting to dense format
    data_tensors = torch.stack(data_tensors)
    existence_tensors = torch.stack(existence_tensors)
    clusters = torch.tensor(cluster_list)

    return data_tensors, existence_tensors, clusters

