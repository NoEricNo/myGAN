
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

config = Config()

# Initialize SettingUp
setup = SettingUp(config)

checkpoint = torch.load('most_recent_gan_model.pth')
setup.gan.generator_r.load_state_dict(checkpoint['generator_r_state_dict'])
setup.gan.generator_e.load_state_dict(checkpoint['generator_e_state_dict'])
setup.gan.discriminator_main.load_state_dict(checkpoint['discriminator_main_state_dict'])
setup.gan.discriminator_distribution.load_state_dict(checkpoint['discriminator_distribution_state_dict'])
setup.gan.discriminator_latent.load_state_dict(checkpoint['discriminator_latent_state_dict'])
setup.gan.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
setup.gan.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

print("Model loaded successfully.")
# Generate and print some samples
generate_samples(setup.device, setup.gan.generator_r, setup.gan.generator_e)