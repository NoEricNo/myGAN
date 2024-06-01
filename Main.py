import torch
from Training import Training
from Generator import Generator
from Discriminator import Discriminator
from Dataset import MovieLensDataLoader

ratings_file = "datasets/ratings_100k.csv"
batch_size = 128
chunk_size = 610  # Adjust to the correct size based on your data
num_epochs = 1
discriminator_fc1_size = 512
discriminator_main_sizes = [256, 128]
generator_fc1_size = 512
generator_main_sizes = [256, 128]
existence_gen_size = 968045  # Adjust according to your needs
dropout_rate = 0.3
learning_rate = 0.0002
accumulation_steps = 1
scaler = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_loader = MovieLensDataLoader(ratings_file, batch_size, chunk_size)
num_chunks = data_loader.num_chunks

# Print statements to check the values
print(f"generator_fc1_size: {generator_fc1_size}")
print(f"existence_gen_size: {existence_gen_size}")

generator = Generator(generator_fc1_size, generator_main_sizes, dropout_rate, existence_gen_size).to(device)
discriminator = Discriminator(chunk_size * 5, discriminator_fc1_size, discriminator_main_sizes, dropout_rate).to(device)

training = Training(generator, discriminator, device)

training.train_epoch(data_loader, chunk_size, num_chunks, num_epochs=num_epochs, accumulation_steps=accumulation_steps, scaler=scaler, verbose=True)