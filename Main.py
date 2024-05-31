import os
import torch
import logging
import time
from datetime import datetime
from Dataset import MovieLensDataLoader
from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ratings_file = 'datasets/ratings_100k.csv'
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"The file {ratings_file} does not exist. Please check the file path.")

    batch_size = 128
    chunk_size = 4000  # Adjusted to match the maximum chunk size

    # Tuneable parameters
    noise_dim = 100
    generator_fc1_size = 128
    generator_rating_gen_sizes = [256, 2000]  # [hidden_size, max_output_size]
    generator_existence_gen_size = 2000

    discriminator_fc1_size = 512
    discriminator_main_sizes = [256, 256]  # [hidden_size1, hidden_size2]

    data_loader = MovieLensDataLoader(ratings_file, batch_size, chunk_size)
    num_chunks = data_loader.num_chunks

    generator = Generator(noise_dim, generator_fc1_size, generator_rating_gen_sizes, generator_existence_gen_size).to(device)
    discriminator = Discriminator(discriminator_fc1_size, discriminator_main_sizes).to(device)
    gan = GAN(generator, discriminator)

    num_epochs = 60

    # Setup logger
    results_dir = 'Results'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(results_dir, f'training_log_{timestamp}.log')
    logger = setup_logger(log_file)

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_start_time = time.time()

        for chunk_idx in range(num_chunks):
            print(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
            chunk_start_time = time.time()

            data_loader.set_chunk(chunk_idx)
            loader = data_loader.get_loader()
            for real_data, real_existence in loader:
                chunk_size = real_data.shape[1]
                real_data = real_data.to(device)
                real_existence = real_existence.to(device)
                break  # Only print shapes for the first batch
            gan.train(loader, accumulation_steps=1, scaler=None, chunk_size=chunk_size)

            chunk_end_time = time.time()
            chunk_duration = chunk_end_time - chunk_start_time
            print(f"Chunk Time: {chunk_duration:.2f} seconds")
            logger.info(f"Chunk Time: {chunk_duration:.2f} seconds")
        epoch_end_time = time.time()  # End time for the epoch
        epoch_duration = epoch_end_time - epoch_start_time  # Calculate epoch duration
        print(f"Epoch Time: {epoch_duration:.2f} seconds")
        logger.info(f"Epoch Time: {epoch_duration:.2f} seconds")