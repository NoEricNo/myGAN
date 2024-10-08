import torch
import logging
import os
from datetime import datetime
from torch.utils.data import DataLoader
from dataset import MovieLensDataset, sparse_collate
from SeparateGenerator import RatingsGenerator, ExistenceGenerator
from Discriminator import MainDiscriminator, DistributionDiscriminator, LatentFactorDiscriminator
from gan import GAN
from sklearn.decomposition import TruncatedSVD

class SettingUp:
    def __init__(self, config, load_data=True):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self.setup_logging()

        if load_data:
            self.dataset, self.data_loader, self.all_ratings = self.load_data()
            self.item_factors = self.perform_svd(self.all_ratings.numpy(), self.config.latent_dim)
            self.gan = self.initialize_models(self.dataset.num_movies)
        else:
            self.dataset = None
            self.data_loader = None
            self.all_ratings = None
            self.item_factors = None
            self.gan = None

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

    def initialize_models(self, num_movies):
        ratings_generator = RatingsGenerator(
            input_size=self.config.input_size,
            fc1_size=self.config.fc1_size,
            main_sizes=self.config.main_sizes,
            dropout_rate=self.config.dropout_rate,
            num_items=num_movies
        ).to(self.device)

        existence_generator = ExistenceGenerator(
            input_size=self.config.input_size,
            fc1_size=self.config.fc1_size,
            main_sizes=self.config.main_sizes,
            dropout_rate=self.config.dropout_rate,
            num_items=num_movies
        ).to(self.device)

        main_discriminator = MainDiscriminator(
            num_movies=num_movies,
            dropout_rate=self.config.dropout_rate,
            fc1_size=self.config.main_disc_fc1_size,
            fc2_size=self.config.main_disc_fc2_size,
            fc3_size=self.config.main_disc_fc3_size
        ).to(self.device)

        distribution_discriminator = DistributionDiscriminator(
            num_movies=num_movies,
            dropout_rate=self.config.dropout_rate,
            fc1_size=self.config.dist_disc_fc1_size,
            fc2_size=self.config.dist_disc_fc2_size,
            fc3_size=self.config.dist_disc_fc3_size
        ).to(self.device)

        latent_factor_discriminator = LatentFactorDiscriminator(
            num_movies=num_movies,
            latent_dim=self.config.latent_dim,
            dropout_rate=self.config.dropout_rate,
            fc1_size=self.config.latent_disc_fc1_size,
            fc2_size=self.config.latent_disc_fc2_size,
            fc3_size=self.config.latent_disc_fc3_size
        ).to(self.device)

        optimizer_g_ratings = torch.optim.Adam(ratings_generator.parameters(), lr=self.config.lr_g_ratings, betas=self.config.betas)
        optimizer_g_existence = torch.optim.Adam(existence_generator.parameters(), lr=self.config.lr_g_existence, betas=self.config.betas)
        optimizer_d_main = torch.optim.Adam(main_discriminator.parameters(), lr=self.config.lr_d_main, betas=self.config.betas)
        optimizer_d_distribution = torch.optim.Adam(distribution_discriminator.parameters(), lr=self.config.lr_d_distribution, betas=self.config.betas)
        optimizer_d_latent = torch.optim.Adam(latent_factor_discriminator.parameters(), lr=self.config.lr_d_latent, betas=self.config.betas)

        return GAN(
            ratings_generator, existence_generator, main_discriminator, distribution_discriminator, latent_factor_discriminator,
            self.device, optimizer_g_ratings, optimizer_g_existence, optimizer_d_main, optimizer_d_distribution, optimizer_d_latent,
            self.logger
        )

    def get_item_factors(self):
        item_factors = torch.tensor(self.item_factors, dtype=torch.float32).to(self.device)
        return item_factors

    def load_pretrained_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)

        # Determine the number of movies from the generator's final layer
        generator_r_state_dict = checkpoint['generator_r_state_dict']
        fc_final_weight = generator_r_state_dict['fc_final.weight']
        num_movies = fc_final_weight.size(0)  # This is directly the number of movies

        print(f"Detected number of movies: {num_movies}")

        self.gan = self.initialize_models(num_movies)  # Initialize the model structure

        self.gan.generator_r.load_state_dict(checkpoint['generator_r_state_dict'])
        self.gan.generator_e.load_state_dict(checkpoint['generator_e_state_dict'])
        self.gan.main_discriminator.load_state_dict(checkpoint['main_discriminator_state_dict'])
        self.gan.distribution_discriminator.load_state_dict(checkpoint['distribution_discriminator_state_dict'])
        self.gan.latent_factor_discriminator.load_state_dict(checkpoint['latent_factor_discriminator_state_dict'])

        self.gan.eval()  # Set the model to evaluation mode

        # Disable gradient computation for evaluation
        for param in self.gan.parameters():
            param.requires_grad = False

        print(f"Model loaded successfully and set to evaluation mode. Number of movies: {num_movies}")
