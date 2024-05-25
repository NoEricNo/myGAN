import os
import torch
from Dataset import MovieLensDataLoader
from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ratings_file = 'ratings_100k.csv'
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"The file {ratings_file} does not exist. Please check the file path.")

    batch_size = 64
    chunk_size = 2000  # Adjusted to match the maximum chunk size

    # Tuneable parameters
    noise_dim = 100
    generator_fc1_size = 128
    generator_rating_gen_sizes = [256, 2000]  # [hidden_size, output_size]
    generator_existence_gen_size = 2000

    discriminator_input_size = chunk_size * 2
    discriminator_fc1_size = 512
    discriminator_main_sizes = [256, 256]  # [hidden_size1, hidden_size2]
    discriminator_existence_output_size = chunk_size

    data_loader = MovieLensDataLoader(ratings_file, batch_size, chunk_size)
    num_chunks = data_loader.num_chunks

    generator = Generator(noise_dim, generator_fc1_size, generator_rating_gen_sizes, generator_existence_gen_size).to(device)
    discriminator = Discriminator(discriminator_input_size, discriminator_fc1_size, discriminator_main_sizes, discriminator_existence_output_size).to(device)
    gan = GAN(generator, discriminator)

    num_epochs = 60

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for chunk_idx in range(num_chunks):
            print(f"Processing chunk {chunk_idx+1}/{num_chunks}")
            data_loader.set_chunk(chunk_idx)
            loader = data_loader.get_loader()
            for real_data, real_existence in loader:
                #print(f"Data loader output shapes: real_data: {real_data.shape}, real_existence: {real_existence.shape}")  # Debugging line
                break  # Print shapes for the first batch only
            gan.train(loader, accumulation_steps=1, scaler=None, chunk_size=chunk_size)
