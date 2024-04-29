from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNISTDataset:
    def __init__(self, batch_size, download_dir='mnist_data/'):
        # Transforms to apply to the images
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert a PIL Image or numpy.ndarray to tensor.
            transforms.Normalize((0.5,), (0.5,))  # Normalize a tensor image with mean and standard deviation.
        ])

        # Download and load the training data
        self.trainset = datasets.MNIST(download_dir, download=True, train=True, transform=self.transform)
        self.dataloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

    def get_loader(self):
        return self.dataloader
