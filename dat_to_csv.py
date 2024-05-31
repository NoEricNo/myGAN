import pandas as pd

input_file = 'datasets/ratings_1m.dat'
output_file = 'datasets/ratings_1m.csv'
column_names = ['userId', 'movieId', 'rating', 'timestamp']
df = pd.read_csv(input_file, sep='::', engine='python', names=column_names)
df.to_csv(output_file, index=False)
print(f'File {output_file} has been created successfully.')
