import torch
from Dataset import MovieLensDataLoader
from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
ratings_file = "datasets/ratings_100k_preprocessed.csv"
batch_size = 32
chunk_size = 100
num_epochs = 50
input_size = 100
fc1_size = 512
main_sizes = [256, 128]
dropout_rate = 0.3

# Load data
data_loader = MovieLensDataLoader(ratings_file, batch_size, chunk_size)

# Initialize models
generator = Generator(input_size=input_size, fc1_size=fc1_size, main_sizes=main_sizes, dropout_rate=dropout_rate, existence_gen_size=9724).to(device)
discriminator = Discriminator(input_size=9724, fc1_size=fc1_size, main_sizes=main_sizes, dropout_rate=dropout_rate).to(device)

# Initialize GAN
gan = GAN(generator, discriminator, device)

# Train the model
epoch_d_losses, epoch_g_losses = gan.train_epoch(data_loader, chunk_size, num_chunks=1, num_epochs=num_epochs, verbose=True)