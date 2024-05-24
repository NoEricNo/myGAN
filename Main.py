from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN
from Dataset import MovieLensDataLoader
import torch
from torch.cuda.amp import GradScaler

# Configuration (Control Panel)
batch_size = 32 # Reduced batch size to fit GPU memory
ratings_file = 'ratings_25m.csv'  # Update this with the actual path to your ratings file
num_epochs = 60  # Set the number of epochs as needed
learning_rate_G = 0.0002  # Learning rate for the generator
learning_rate_D = 0.0001  # Learning rate for the discriminator
beta1 = 0.5  # Beta1 for Adam optimizer
beta2 = 0.999  # Beta2 for Adam optimizer
label_smoothing = 0.9  # Label smoothing for real data
noise_factor = 0.05  # Noise factor to add to real data
update_D_frequency = 2  # Update discriminator every N steps
accumulation_steps = 2  # Gradient accumulation steps for larger batch sizes

# Initialize the dataset and dataloader
data_loader = MovieLensDataLoader(ratings_file, batch_size)
dataloader = data_loader.get_loader()
num_movies = data_loader.get_num_movies()

# Initialize the GAN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(num_movies=num_movies).to(device)
discriminator = Discriminator(num_movies=num_movies).to(device)
scaler = GradScaler()  # Initialize the gradient scaler for mixed precision training
gan = GAN(generator, discriminator, device, num_movies, learning_rate_G, learning_rate_D, beta1, beta2, label_smoothing, noise_factor, update_D_frequency)

# Train the GAN
gan.train(dataloader, num_epochs, accumulation_steps, scaler)
