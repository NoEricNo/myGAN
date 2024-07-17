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
        self.main_sizes = [4096, 4096, 4096, 4096]  # Example sizes for the main layers in the generator

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
        self.num_epochs = 600

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

#print("GAN attributes:", dir(setup.gan))

# After training
torch.save({
    'generator_r_state_dict': setup.gan.generator_r.state_dict(),
    'generator_e_state_dict': setup.gan.generator_e.state_dict(),
    'main_discriminator_state_dict': setup.gan.main_discriminator.state_dict(),
    'distribution_discriminator_state_dict': setup.gan.distribution_discriminator.state_dict(),
    'latent_factor_discriminator_state_dict': setup.gan.latent_factor_discriminator.state_dict(),
    'optimizer_g_r_state_dict': setup.gan.optimizer_g_r.state_dict(),
    'optimizer_g_e_state_dict': setup.gan.optimizer_g_e.state_dict(),
    'optimizer_d_main_state_dict': setup.gan.optimizer_d_main.state_dict(),
    'optimizer_d_distribution_state_dict': setup.gan.optimizer_d_distribution.state_dict(),
    'optimizer_d_latent_state_dict': setup.gan.optimizer_d_latent.state_dict(),
}, 'gan_model.pth')

print("Model saved successfully.")

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

