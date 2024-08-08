class Config:
    def __init__(self):
        # Dataset parameters
        self.ratings_file = "datasets/ratings_100k_preprocessed.csv"
        self.batch_size = 256

        # Model parameters
        self.input_size = 100
        self.latent_dim = 50  # Latent dimension for SVD
        self.dropout_rate = 0.3
        self.fc1_size = 2048  # Example size for the fully connected layer in the generator
        self.main_sizes = [2048, 2048]  # Example sizes for the main layers in the generator

        # Main Discriminator parameters
        self.main_disc_fc1_size = 128
        self.main_disc_fc2_size = 128
        self.main_disc_fc3_size = 64

        # Distribution Discriminator parameters
        self.dist_disc_fc1_size = 128
        self.dist_disc_fc2_size = 128
        self.dist_disc_fc3_size = 64

        # Latent Factor Discriminator parameters
        self.latent_disc_fc1_size = 128
        self.latent_disc_fc2_size = 128
        self.latent_disc_fc3_size = 64

        # Training parameters
        self.num_epochs = 1200

        # Separate learning rates for each component
        self.lr_g_ratings = 0.00005  # Learning rate for the ratings generator
        self.lr_g_existence = 0.000005  # Learning rate for the existence generator
        self.lr_d_main = 0.000005  # Learning rate for the main discriminator
        self.lr_d_distribution = 0.000005  # Learning rate for the distribution discriminator
        self.lr_d_latent = 0.00005  # Learning rate for the latent factor discriminator

        self.betas = (0.5, 0.999)
