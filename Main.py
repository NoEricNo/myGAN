import matplotlib.pyplot as plt
from setting_up import SettingUp
import torch

class Config:
    def __init__(self):
        # Dataset parameters
        self.ratings_file = "datasets/ratings_100k_preprocessed.csv"
        self.batch_size = 256

        # Model parameters
        self.input_size = 100
        self.latent_dim = 50  # Latent dimension for SVD
        self.dropout_rate = 0.3
        self.fc1_size = 2048  # Example size for the fully connected layer in the generator
        self.main_sizes = [2048, 2048]  # Example sizes for the main layers in the generator

        # Main Discriminator parameters
        self.main_disc_fc1_size = 128
        self.main_disc_fc2_size = 128
        self.main_disc_fc3_size = 64

        # Distribution Discriminator parameters
        self.dist_disc_fc1_size = 128
        self.dist_disc_fc2_size = 128
        self.dist_disc_fc3_size = 64

        # Latent Factor Discriminator parameters
        self.latent_disc_fc1_size = 128
        self.latent_disc_fc2_size = 128
        self.latent_disc_fc3_size = 64

        # Training parameters
        self.num_epochs = 1200

        # Separate learning rates for each component
        self.lr_g_ratings = 0.00005  # Learning rate for the ratings generator
        self.lr_g_existence = 0.000005  # Learning rate for the existence generator
        self.lr_d_main = 0.000005  # Learning rate for the main discriminator
        self.lr_d_distribution = 0.000005  # Learning rate for the distribution discriminator
        self.lr_d_latent = 0.00005  # Learning rate for the latent factor discriminator

        self.betas = (0.5, 0.999)

# Load configuration
config = Config()

# Initialize SettingUp
setup = SettingUp(config)

# Example logging
setup.logger.info("This is a log message for a new run.")

# Set device
device = setup.device

# Get item factors
item_factors = setup.get_item_factors()
#print(f"Shape of item_factors: {item_factors.shape}, dtype: {item_factors.dtype}")

# Train the model
epoch_d_main_losses, epoch_d_distribution_losses, epoch_d_latent_losses, epoch_g_r_losses, epoch_g_e_losses = setup.gan.train_epoch(
    setup.data_loader, num_epochs=config.num_epochs, item_factors=item_factors)

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(epoch_d_main_losses, label='Main Discriminator Loss')
plt.plot(epoch_d_distribution_losses, label='Distribution Discriminator Loss')
plt.plot(epoch_d_latent_losses, label='Latent Factor Discriminator Loss')
plt.plot(epoch_g_r_losses, label='Ratings Generator Loss')
plt.plot(epoch_g_e_losses, label='Existence Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Results/loss_plot.png")
plt.show()

def post_process(ratings, existence):
    # Round ratings to nearest 0.5
    ratings = torch.round(ratings * 2) / 2.0
    # Threshold existence to binary values
    existence = (existence > 0.5).float()
    # Apply existence mask
    ratings = ratings * existence
    return ratings, existence

def generate_samples(generator_r, generator_e, num_samples=5):
    # Generate noise
    noise = torch.randn(num_samples, generator_r.input_size).to(device)

    # Generate fake ratings and existence flags
    fake_ratings = generator_r(noise)
    fake_existence = generator_e(noise)
    fake_ratings = fake_ratings.to_dense()
    fake_existence = fake_existence.to_dense()

    # Post-process the generated data
    fake_ratings, fake_existence = post_process(fake_ratings, fake_existence)

    for i in range(num_samples):
        print(f"Sample {i + 1}:")
        print("Ratings:")
        print(fake_ratings[i].detach().cpu().numpy())  # Use detach() before converting to numpy
        print("Existence Flags:")
        print(fake_existence[i].detach().cpu().numpy())  # Use detach() before converting to numpy
        print("\n")

# Generate and print some samples
generate_samples(setup.gan.generator_r, setup.gan.generator_e)
