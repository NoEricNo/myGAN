import torch
import psutil
import os
import logging
from Dataset import MovieLensDataLoader
from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# Function to print memory usage
def print_memory_usage(step):
    process = psutil.Process(os.getpid())
    print(f"{step} - Memory usage: {process.memory_info().rss / (1024 * 1024)} MB")

# Create a "Results" folder if it doesn't exist
os.makedirs("Results", exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset parameters
ratings_file = "datasets/ratings_100k_preprocessed.csv"
batch_size = 128
num_user_groups = 1
num_movie_groups = 1
movie_overlap_ratio = 0

# Model parameters
input_size = 100
fc1_size = 512
main_sizes = [256, 128]
dropout_rate = 0.3

# Training parameters
num_epochs = 50
lr_g = 0.0002
lr_d = 0.0002
betas = (0.5, 0.999)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a file handler for logging
file_handler = logging.FileHandler("Results/training_log.txt")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Load data
data_loader = MovieLensDataLoader(ratings_file, batch_size, num_user_groups, num_movie_groups, movie_overlap_ratio)

# Initialize models
print_memory_usage("Before Generator")
generator = Generator(input_size=input_size, fc1_size=fc1_size, main_sizes=main_sizes,
                      dropout_rate=dropout_rate, num_users=data_loader.dataset.num_users,
                      num_movies=data_loader.dataset.num_movies).to(device)
print_memory_usage("After Generator")

print_memory_usage("Before Discriminator")
discriminator = Discriminator(fc1_size=fc1_size, dropout_rate=dropout_rate).to(device)
print_memory_usage("After Discriminator")

# Print model parameters for verification
print(f"Generator num_users: {data_loader.dataset.num_users}")
print(f"Generator num_movies: {data_loader.dataset.num_movies}")
print(f"Discriminator num_users: {data_loader.dataset.num_users}")
print(f"Discriminator num_movies: {data_loader.dataset.num_movies}")

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas)

# Initialize GAN
gan = GAN(generator, discriminator, device, criterion, optimizer_g, optimizer_d, logger)

# Train the model
epoch_d_losses, epoch_g_losses = gan.train_epoch(data_loader, num_chunks=len(data_loader), num_epochs=num_epochs)

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(epoch_d_losses, label='Discriminator Loss')
plt.plot(epoch_g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Results/loss_plot.png")
plt.show()
