import matplotlib.pyplot as plt
from setting_up import SettingUp
import torch


class Config:
    def __init__(self):
        # Dataset parameters
        self.ratings_file = "../../datasets/ratings_100k_preprocessed.csv"
        self.batch_size = 256

        # Model parameters
        self.input_size = 100
        self.latent_dim = 50  # Latent dimension for SVD
        self.dropout_rate = 0.3
        self.fc1_size = 2048  # Example size for the fully connected layer in the generator
        self.main_sizes = [2048, 4096, 4096, 2048]  # Example sizes for the main layers in the generator

        # Discriminator parameters
        self.disc_fc1_size = 128
        self.disc_fc2_size = 128
        self.disc_fc3_size = 64

        # Training parameters
        self.num_epochs = 1200
        self.lr_g = 0.00005  # Further lowered learning rate for the generator
        self.lr_d = 0.000005  # Further lowered learning rate for the discriminator
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
epoch_d_losses, epoch_g_r_losses, epoch_g_e_losses = setup.gan.train_epoch(setup.data_loader, num_epochs=config.num_epochs,
                                                       item_factors=item_factors)

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(epoch_d_losses, label='Discriminator Loss')
plt.plot(epoch_g_r_losses, label='Generator(R) Loss')
plt.plot(epoch_g_e_losses, label='Generator(E) Loss')
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


def generate_samples(generator, num_samples=5):
    # Generate noise
    noise = torch.randn(num_samples, generator.input_size).to(device)

    # Generate fake ratings and existence flags
    fake_ratings, fake_existence = generator(noise)
    fake_ratings = fake_ratings.to_dense()
    fake_existence = fake_existence.to_dense()

    # Post-process the generated data
    #fake_ratings, fake_existence = post_process(fake_ratings, fake_existence)

    for i in range(num_samples):
        print(f"Sample {i + 1}:")
        print("Ratings:")
        print(fake_ratings[i].detach().cpu().numpy())  # Use detach() before converting to numpy
        print("Existence Flags:")
        print(fake_existence[i].detach().cpu().numpy())  # Use detach() before converting to numpy
        print("\n")


# Generate and print some samples
generate_samples(setup.gan.generator)