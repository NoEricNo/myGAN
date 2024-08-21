import torch
from gan import WGAN
from dataset import get_user_item_dataset


def load_data(batch_size):
    # Load your dataset here
    dataset = get_user_item_dataset(ratings_file='./Datasets/ml-100k/u.data', num_users=943, num_items=1682)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Print a sample batch shape
    for batch in data_loader:
        print(f"Batch shape from DataLoader: {batch['user'].shape}")  # Print shape of a batch
        break  # We only need to check the shape once

    return data_loader


def main():
    # Directly set the parameters here
    config = {
        'noise_dim': 100,  # Dimensionality of the noise vector
        'num_groups': 10,  # Number of groups for latent preferences and behaviors
        'num_items': 1682,  # Number of items in the dataset
        'num_latent_factors': 20,  # Number of latent factors
        'window_size': 5,  # Window size for sliding over user ratings
        'lr': 0.0002  # Learning rate for the optimizers
    }

    num_epochs = 100  # Number of epochs for training
    batch_size = 512  # Batch size for training
    critic_steps = 1  # Number of critic updates per generator update
    save_model = True  # Flag to save the model after training
    save_path = 'wgan.pth'  # Path to save the model

    # Initialize the WGAN with the provided configuration
    wgan = WGAN(config)

    # Load the data
    real_data = load_data(batch_size)

    # Train the WGAN
    wgan.train(real_data, num_epochs=num_epochs, critic_steps=critic_steps)

    # Save the model if specified
    if save_model:
        torch.save(wgan, save_path)
        print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()
