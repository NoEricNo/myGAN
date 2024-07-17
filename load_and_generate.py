
import torch
import pandas as pd
from setting_up import SettingUp
from Main import Config
def post_process(ratings, existence):
    # Round ratings to nearest 0.5
    ratings = torch.round(ratings * 2) / 2.0
    # Threshold existence to binary values
    existence = (existence > 0.5).float()
    # Apply existence mask
    ratings = ratings * existence
    return ratings, existence

def generate_samples(device, generator_r, generator_e, num_samples=1000, batch_size=100, num_movies=9724):
    all_fake_ratings = []
    all_fake_existence = []

    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)
        noise = torch.randn(current_batch_size, generator_r.input_size).to(device)

        fake_ratings = generator_r(noise)
        fake_existence = generator_e(noise)
        fake_ratings = fake_ratings.to_dense()
        fake_existence = fake_existence.to_dense()

        fake_ratings, fake_existence = post_process(fake_ratings, fake_existence)

        all_fake_ratings.append(fake_ratings.detach().cpu())
        all_fake_existence.append(fake_existence.detach().cpu())

    # Concatenate all batches
    all_fake_ratings = torch.cat(all_fake_ratings, dim=0)
    all_fake_existence = torch.cat(all_fake_existence, dim=0)

    # Convert to numpy
    all_fake_ratings_np = all_fake_ratings.numpy()
    all_fake_existence_np = all_fake_existence.numpy()

    # Create intermediate DataFrame
    df_matrix = pd.DataFrame(all_fake_ratings_np)
    df_matrix.index.name = 'userId'
    df_matrix.columns.name = 'movieId'

    # Print summary statistics
    print("\nRating Matrix Summary:")
    print(f"Shape: {df_matrix.shape}")
    print(f"Non-null values: {df_matrix.count().sum()}")
    print(f"Sparsity: {1 - df_matrix.count().sum() / (df_matrix.shape[0] * df_matrix.shape[1]):.2%}")
    print("\nRating Distribution:")
    print(df_matrix.unstack().value_counts(normalize=True).sort_index())

    # Create a list to store all ratings
    ratings_list = []

    for user_id in range(num_samples):
        for movie_id in range(num_movies):
            if all_fake_existence_np[user_id, movie_id] > 0.5:  # Only include ratings where existence is True
                rating = all_fake_ratings_np[user_id, movie_id]
                ratings_list.append([user_id + 1, movie_id + 1, rating])

    # Create a DataFrame
    df = pd.DataFrame(ratings_list, columns=['userId', 'movieId', 'rating'])

    # Save to CSV
    df.to_csv('fake_ratings.csv', index=False)

    print(f"Generated and saved {len(df)} ratings for {num_samples} fake user profiles.")

    return df

def load_gan_model(setup, model_path):
    checkpoint = torch.load(model_path, map_location=setup.device)

    setup.gan.generator_r.load_state_dict(checkpoint['generator_r_state_dict'])
    setup.gan.generator_e.load_state_dict(checkpoint['generator_e_state_dict'])
    setup.gan.main_discriminator.load_state_dict(checkpoint['main_discriminator_state_dict'])
    setup.gan.distribution_discriminator.load_state_dict(checkpoint['distribution_discriminator_state_dict'])
    setup.gan.latent_factor_discriminator.load_state_dict(checkpoint['latent_factor_discriminator_state_dict'])

    setup.gan.eval()  # Set the model to evaluation mode

    # Disable gradient computation for evaluation
    for param in setup.gan.parameters():
        param.requires_grad = False

    print("Model loaded successfully and set to evaluation mode.")

def main():
    config = Config()

    # Initialize SettingUp without loading data or initializing models
    setup = SettingUp(config, load_data=False)

    # Load the pretrained model
    setup.load_pretrained_model('gan_model.pth')

    # Set a fixed seed for reproducibility
    torch.manual_seed(42)

    # Generate samples
    generate_samples(setup.device, setup.gan.generator_r, setup.gan.generator_e)

if __name__ == "__main__":
    main()