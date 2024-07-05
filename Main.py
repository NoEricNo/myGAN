import matplotlib.pyplot as plt
from setting_up import SettingUp

class Config:
    def __init__(self):
        # Dataset parameters
        self.ratings_file = "datasets/ratings_100k_preprocessed.csv"
        self.batch_size = 256

        # Model parameters
        self.input_size = 100
        self.latent_dim = 50  # Latent dimension for SVD
        self.dropout_rate = 0.3
        self.fc1_size = 512  # Example size for the fully connected layer in the generator
        self.main_sizes = [1024, 512, 256]  # Example sizes for the main layers in the generator

        # Discriminator parameters
        self.disc_fc1_size = 128
        self.disc_fc2_size = 128
        self.disc_fc3_size = 64

        # Training parameters
        self.num_epochs = 200
        self.lr_g = 0.0004
        self.lr_d = 0.0002
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
print(f"Shape of item_factors: {item_factors.shape}, dtype: {item_factors.dtype}")

# Train the model
epoch_d_losses, epoch_g_losses = setup.gan.train_epoch(setup.data_loader, num_epochs=config.num_epochs, item_factors=item_factors)

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(epoch_d_losses, label='Discriminator Loss')
plt.plot(epoch_g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Results/loss_plot.png")
plt.show()
