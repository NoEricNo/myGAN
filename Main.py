import torch
torch.autograd.set_detect_anomaly(True)

from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN
from Dataset import MovieLensDataLoader

# Configuration
batch_size = 64  # You can adjust the batch size as needed
ratings_file = 'ratings.csv'  # Update this with the actual path to your ratings file

# Initialize the dataset and dataloader
data_loader = MovieLensDataLoader(ratings_file, batch_size)
dataloader = data_loader.get_loader()
num_movies = data_loader.get_num_movies()

# Initialize the GAN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(num_movies=num_movies).to(device)
discriminator = Discriminator(num_movies=num_movies).to(device)
gan = GAN(generator, discriminator, device, num_movies)  # Pass num_movies

# Train the GAN
num_epochs = 50  # Set the number of epochs as needed
gan.train(dataloader, num_epochs)
