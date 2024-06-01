import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


def preprocess_movie_lens_data(input_file, output_file):
    ratings = pd.read_csv(input_file)

    # Adjust user IDs to start from 0 and be continuous
    user_id_mapping = {user_id: idx for idx, user_id in enumerate(ratings['userId'].unique())}
    ratings['userId'] = ratings['userId'].map(user_id_mapping)

    # Adjust movie IDs to start from 0 and be continuous
    movie_id_mapping = {movie_id: idx for idx, movie_id in enumerate(ratings['movieId'].unique())}
    ratings['movieId'] = ratings['movieId'].map(movie_id_mapping)

    # Save the preprocessed data
    ratings.to_csv(output_file, index=False)





def verify_movielens_100k(ratings_file):
    # Load the ratings data
    ratings = pd.read_csv(ratings_file)

    # Check the number of rows
    expected_rows = 100000
    actual_rows = len(ratings)
    print(f"Number of rows: {actual_rows} (expected: {expected_rows})")

    # Check the columns
    expected_columns = ['userId', 'movieId', 'rating', 'timestamp']
    actual_columns = list(ratings.columns)
    print(f"Columns: {actual_columns}")

    # Check the data types
    expected_dtypes = {'userId': np.int64, 'movieId': np.int64, 'rating': np.float64, 'timestamp': np.int64}
    actual_dtypes = ratings.dtypes.to_dict()
    print(f"Data types: {actual_dtypes}")

    # Check the range of user IDs
    min_user_id = ratings['userId'].min()
    max_user_id = ratings['userId'].max()
    print(f"User ID range: {min_user_id} to {max_user_id}")

    # Check the range of movie IDs
    min_movie_id = ratings['movieId'].min()
    max_movie_id = ratings['movieId'].max()
    print(f"Movie ID range: {min_movie_id} to {max_movie_id}")

    # Check the range of ratings
    min_rating = ratings['rating'].min()
    max_rating = ratings['rating'].max()
    print(f"Rating range: {min_rating} to {max_rating}")

    # Check for missing values
    missing_values = ratings.isnull().sum()
    print(f"Missing values: {missing_values}")

    # Check for duplicate entries
    duplicate_entries = ratings.duplicated().sum()
    print(f"Duplicate entries: {duplicate_entries}")


# Specify the path to your MovieLens 100K ratings file
ratings_file = 'datasets/ratings_100k.csv'

# Verify the dataset
verify_movielens_100k(ratings_file)

# Specify the input and output file paths
input_file = 'datasets/ratings_100k.csv'
output_file = 'datasets/ratings_100k_preprocessed.csv'

# Preprocess the dataset
preprocess_movie_lens_data(input_file, output_file)

# Specify the path to your MovieLens 100K ratings file
ratings_file = 'datasets/ratings_100k_preprocessed.csv'

# Verify the dataset
verify_movielens_100k(ratings_file)