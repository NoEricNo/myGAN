import torch
import logging
import os
from datetime import datetime
from torch.utils.data import DataLoader
from Dataset import MovieLensDataset, sparse_collate
from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN
import torch.optim as optim
from sklearn.decomposition import TruncatedSVD


class SettingUp:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self.setup_logging()
        self.dataset, self.data_loader, self.all_ratings = self.load_data()
        self.item_factors = self.perform_svd(self.all_ratings.numpy(), self.config.latent_dim)
        self.gan = self.initialize_models()

    def setup_logging(self):
        log_dir = "Results"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_filename),
                                logging.StreamHandler()
                            ])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        return logger

    def perform_svd(self, real_ratings, latent_dim):
        svd = TruncatedSVD(n_components=latent_dim)
        item_factors = svd.fit_transform(real_ratings.T)
        return item_factors.T

    def load_data(self):
        dataset = MovieLensDataset(self.config.ratings_file)
        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=sparse_collate)
        all_ratings = torch.tensor(dataset.rating_matrix.toarray(), dtype=torch.float32)
        return dataset, data_loader, all_ratings

    def initialize_models(self):
        generator = Generator(input_size=self.config.input_size, fc1_size=self.config.fc1_size,
                              main_sizes=self.config.main_sizes,
                              dropout_rate=self.config.dropout_rate, num_users=self.dataset.num_users,
                              num_movies=self.dataset.num_movies).to(self.device)

        discriminator = Discriminator(num_movies=self.dataset.num_movies, dropout_rate=self.config.dropout_rate,
                                      fc1_size=self.config.disc_fc1_size, fc2_size=self.config.disc_fc2_size,
                                      fc3_size=self.config.disc_fc3_size).to(self.device)

        optimizer_g = optim.Adam(generator.parameters(), lr=self.config.lr_g, betas=self.config.betas)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=self.config.lr_d, betas=self.config.betas)

        return GAN(generator, discriminator, self.device, optimizer_g, optimizer_d, self.logger)

    def get_item_factors(self):
        item_factors = torch.tensor(self.item_factors, dtype=torch.float32).to(self.device)
        return item_factors
