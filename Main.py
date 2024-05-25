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

    data_loader = MovieLensDataLoader(ratings_file, batch_size, chunk_size)
    num_chunks = data_loader.num_chunks
    dynamic_input_size = chunk_size * 2  # Adjusted to match combined input size from Generator (ratings + existence)

    generator = Generator().to(device)
    discriminator = Discriminator(input_size=dynamic_input_size).to(device)  # Ensure this matches Generator output
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
