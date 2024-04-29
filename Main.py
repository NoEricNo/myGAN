from Generator import Generator
from Discreminator import Discriminator
from GAN import GAN

from Dataset import MNISTDataset
import torch


# Configuration
batch_size = 64  # You can adjust the batch size as needed

# Initialize the dataset
mnist_dataset = MNISTDataset(batch_size=batch_size)
dataloader = mnist_dataset.get_loader()

# Now pass `dataloader` to your GAN's train method
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gan = GAN(Generator().to(device), Discriminator().to(device), device)
num_epochs = 50  # Set the number of epochs as needed
gan.train(dataloader, num_epochs)

