import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class UserItemInteractionDataset(Dataset):
    def __init__(self, ratings_file, num_users, num_items, user_col='user_id', item_col='item_id', rating_col='rating', sep='\t', header=None):
        """
        General-purpose dataset class for user-item interactions.

        Args:
            ratings_file (str): Path to the ratings file.
            num_users (int): Total number of users in the dataset.
            num_items (int): Total number of items in the dataset.
            user_col (str): Name of the column that contains user IDs.
            item_col (str): Name of the column that contains item IDs.
            rating_col (str): Name of the column that contains ratings.
            sep (str): Separator used in the ratings file (default: '\t').
            header (None or int): Row number to use as the column names, and the start of the data (default: None).
        """
        # Load the data
        self.ratings_df = pd.read_csv(ratings_file, sep=sep, header=header, names=[user_col, item_col, rating_col, 'timestamp'])

        # Initialize matrices
        self.num_users = num_users
        self.num_items = num_items
        self.user_matrix = np.zeros((self.num_users, self.num_items), dtype=np.float32)
        self.item_matrix = np.zeros((self.num_items, self.num_users), dtype=np.float32)
        self.mask_matrix = np.zeros((self.num_users, self.num_items), dtype=np.float32)
        self.interaction_matrix = np.zeros((self.num_users, self.num_items), dtype=np.float32)

        # Store column names
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

        # Fill matrices based on ratings
        self._preprocess()

    def _preprocess(self):
        for _, row in self.ratings_df.iterrows():
            user_id = int(row[self.user_col]) - 1  # Convert to 0-based index
            item_id = int(row[self.item_col]) - 1
            rating = row[self.rating_col]

            # Fill matrices
            self.user_matrix[user_id, item_id] = rating
            self.item_matrix[item_id, user_id] = rating
            self.mask_matrix[user_id, item_id] = 1  # Mask matrix indicates where interactions exist
            self.interaction_matrix[user_id, item_id] = rating  # Interaction matrix stores the actual ratings

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        user_data = self._get_user_data(idx)  # Collect data for the single user
        # Return 2D tensors without adding an extra batch dimension
        return user_data

    def _get_user_data(self, user_idx):
        return {
            'user': torch.tensor(self.user_matrix[user_idx, :], dtype=torch.float32),
            'item': torch.tensor(self.item_matrix[:, user_idx], dtype=torch.float32),
            'mask': torch.tensor(self.mask_matrix[user_idx, :], dtype=torch.float32),
            'interaction': torch.tensor(self.interaction_matrix[user_idx, :], dtype=torch.float32)
        }


def get_user_item_dataset(ratings_file, num_users, num_items, user_col='user_id', item_col='item_id', rating_col='rating', sep='\t', header=None):
    dataset = UserItemInteractionDataset(ratings_file, num_users, num_items, user_col, item_col, rating_col, sep, header)
    return dataset

if __name__ == '__main__':
    # Example for MovieLens 100k
    ratings_file = 'ml-100k/u.data'
    num_users = 943
    num_items = 1682

    dataset = get_user_item_dataset(ratings_file, num_users, num_items)
    print("Number of users:", len(dataset))
    print("Sample data:", dataset[0])  # Print the data for the first user
