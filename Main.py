import torch
import os
import logging
from Dataset import MovieLensDataLoader
from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# Create a "Results" folder if it doesn't exist
os.makedirs("Results", exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset parameters
ratings_file = "datasets/ratings_100k_preprocessed.csv"
batch_size = 32
chunk_size = 100

# Model parameters
input_size = 100
fc1_size = 512
main_sizes = [256, 128]
dropout_rate = 0.3
existence_gen_size = 9724

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
data_loader = MovieLensDataLoader(ratings_file, batch_size, chunk_size)

# Initialize models
generator = Generator(input_size=input_size, fc1_size=fc1_size, main_sizes=main_sizes,
                      dropout_rate=dropout_rate, existence_gen_size=existence_gen_size).to(device)
discriminator = Discriminator(
    ratings_size=existence_gen_size,
    existence_size=existence_gen_size,
    user_id_size=1,  # Assuming each user ID is a single value
    movie_id_size=1,  # Assuming each movie ID is a single value
    fc1_size=fc1_size,
    main_sizes=main_sizes,
    dropout_rate=dropout_rate
).to(device)

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas)

# Initialize GAN
gan = GAN(generator, discriminator, device, criterion, optimizer_g, optimizer_d, logger)

# Train the model
epoch_d_losses, epoch_g_losses = gan.train_epoch(data_loader, chunk_size, num_chunks=1, num_epochs=num_epochs)

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(epoch_d_losses, label='Discriminator Loss')
plt.plot(epoch_g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Results/loss_plot.png")
plt.show()