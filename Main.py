from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN
from Dataset import MovieLensDataLoader
import torch
# Configuration
batch_size = 64
ratings_file = 'ratings.csv'

# Initialize the dataset and dataloader
data_loader = MovieLensDataLoader(ratings_file, batch_size)
dataloader = data_loader.get_loader()
num_movies = data_loader.get_num_movies()

# Initialize the GAN model
device = torch.device("cpu")  # Force CPU for debugging
generator = Generator(num_movies=num_movies).to(device)
discriminator = Discriminator(num_movies=num_movies).to(device)
gan = GAN(generator, discriminator, device, num_movies)

# Train the GAN
num_epochs = 50
gan.train(dataloader, num_epochs)
