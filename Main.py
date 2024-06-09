import torch
import logging
import os
from datetime import datetime
from torch.utils.data import DataLoader
from Dataset import MovieLensDataset, sparse_collate
from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import numpy as np

def setup_logging():
    log_dir = "Results"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename),
                            logging.StreamHandler()
                        ])

def perform_svd(real_ratings, latent_dim):
    svd = TruncatedSVD(n_components=latent_dim)
    item_factors = svd.fit_transform(real_ratings.T)
    return item_factors

# Set up logging
setup_logging()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Example logging
logger.info("This is a log message for a new run.")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset parameters
ratings_file = "datasets/ratings_100k_preprocessed.csv"
batch_size = 256

# Model parameters
input_size = 100
latent_dim = 50  # Latent dimension for SVD
dropout_rate = 0.3
fc1_size = 512  # Example size for the fully connected layer in the generator
main_sizes = [1024, 512, 256]  # Example sizes for the main layers in the generator

# Discriminator parameters
disc_fc1_size = 128
disc_fc2_size = 128
disc_fc3_size = 64

# Training parameters
num_epochs = 50
lr_g = 0.0002
lr_d = 0.0002
betas = (0.5, 0.999)

# Load data
dataset = MovieLensDataset(ratings_file)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=sparse_collate)

# Convert the entire dataset to a dense matrix for SVD
all_ratings = torch.tensor(dataset.rating_matrix.toarray(), dtype=torch.float32)
print(f"Shape of all_ratings: {all_ratings.shape}, dtype: {all_ratings.dtype}")

# Initialize models
generator = Generator(input_size=input_size, fc1_size=fc1_size, main_sizes=main_sizes, dropout_rate=dropout_rate,
                      num_users=dataset.num_users, num_movies=dataset.num_movies).to(device)

discriminator = Discriminator(num_movies=dataset.num_movies, dropout_rate=dropout_rate, fc1_size=disc_fc1_size, fc2_size=disc_fc2_size, fc3_size=disc_fc3_size).to(device)

# Print model parameters for verification
print(f"Generator num_users: {dataset.num_users}")
print(f"Generator num_movies: {dataset.num_movies}")
print(f"Discriminator num_users: {dataset.num_users}")
print(f"Discriminator num_movies: {dataset.num_movies}")

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas)

# Initialize GAN
gan = GAN(generator, discriminator, device, optimizer_g, optimizer_d, logger)

# Train the model
item_factors = perform_svd(all_ratings.numpy(), latent_dim)
print(f"Shape of item_factors (after SVD): {item_factors.shape}, dtype: {item_factors.dtype}")  # Print shape and dtype of item_factors after SVD
item_factors = item_factors.T
print(f"Shape of item_factors (after transpose): {item_factors.shape}, dtype: {item_factors.dtype}")  # Print shape and dtype of item_factors after transpose
item_factors = torch.tensor(item_factors, dtype=torch.float32).to(device)
print(f"Shape of item_factors (after conversion): {item_factors.shape}, dtype: {item_factors.dtype}")  # Print shape and dtype of item_factors after conversion
epoch_d_losses, epoch_g_losses = gan.train_epoch(data_loader, num_epochs=num_epochs, item_factors=item_factors)

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(epoch_d_losses, label='Discriminator Loss')
plt.plot(epoch_g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Results/loss_plot.png")
plt.show()
