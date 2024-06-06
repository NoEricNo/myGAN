import torch
import logging
import os
from datetime import datetime
from Dataset import MovieLensDataLoader
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
num_user_groups = 1
num_movie_groups = 1
movie_overlap_ratio = 0

# Model parameters
input_size = 100
latent_dim = 50  # Latent dimension for SVD
dropout_rate = 0.3
fc1_size = 512  # Example size for the fully connected layer in the generator
main_sizes = [1024, 512, 256]  # Example sizes for the main layers in the generator

# Training parameters
num_epochs = 50
lr_g = 0.0002
lr_d = 0.0002
betas = (0.5, 0.999)

# Load data
data_loader = MovieLensDataLoader(ratings_file, batch_size, num_user_groups, num_movie_groups, movie_overlap_ratio)

# Convert the entire dataset to a dense matrix for SVD
all_ratings = []
for i in range(len(data_loader.dataset)):
    ratings, _, _, _ = data_loader.dataset[i]
    all_ratings.append(torch.tensor(ratings.todense(), dtype=torch.float32))

all_ratings = torch.cat(all_ratings)
item_factors = perform_svd(all_ratings.numpy(), latent_dim)

# Initialize models
generator = Generator(input_size=input_size, fc1_size=fc1_size, main_sizes=main_sizes, dropout_rate=dropout_rate,
                      num_users=data_loader.dataset.num_users,
                      num_movies=data_loader.dataset.num_movies).to(device)

discriminator = Discriminator(dropout_rate=dropout_rate).to(device)

# Print model parameters for verification
print(f"Generator num_users: {data_loader.dataset.num_users}")
print(f"Generator num_movies: {data_loader.dataset.num_movies}")
print(f"Discriminator num_users: {data_loader.dataset.num_users}")
print(f"Discriminator num_movies: {data_loader.dataset.num_movies}")


# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas)

# Initialize GAN
gan = GAN(generator, discriminator, device, optimizer_g, optimizer_d, logger)

# Train the model
epoch_d_losses, epoch_g_losses = gan.train_epoch(data_loader, num_chunks=len(data_loader), num_epochs=num_epochs, item_factors=torch.tensor(item_factors, dtype=torch.float32).to(device))

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(epoch_d_losses, label='Discriminator Loss')
plt.plot(epoch_g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Results/loss_plot.png")
plt.show()
