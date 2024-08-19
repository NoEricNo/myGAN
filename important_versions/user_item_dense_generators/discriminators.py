import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class UserAttention(nn.Module):
    def __init__(self, num_items, embed_dim, num_heads):
        super(UserAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.embedding = nn.Linear(num_items, embed_dim)

    def forward(self, rating_matrix):
        # rating_matrix shape: (batch_size, num_users, num_items)
        batch_size, num_users, num_items = rating_matrix.shape
        print(f"Shape of rating_matrix: {rating_matrix.shape}")
        # Embed each user's rating vector
        embedded_users = self.embedding(rating_matrix)  # Shape: (batch_size, num_users, embed_dim)
        print(f"Shape of embedded_users: {embedded_users.shape}")
        # Transpose to (num_users, batch_size, embed_dim) for attention
        embedded_users = embedded_users.transpose(0, 1)

        # Apply self-attention across users
        attn_output, _ = self.attention(embedded_users, embedded_users, embedded_users)

        # Transpose back to (batch_size, num_users, embed_dim)
        attn_output = attn_output.transpose(0, 1)

        return attn_output


# Integration into UserRatingCritic
class UserRatingCritic(nn.Module):
    def __init__(self, num_items, window_size=5, embed_dim=32, num_heads=4):
        super(UserRatingCritic, self).__init__()
        self.window_size = window_size
        self.attention_layer = UserAttention(num_items, embed_dim, num_heads)
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(embed_dim * window_size, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 1))  # No Sigmoid, output raw score
        )

    def forward(self, rating_matrix):
        # Replace 0s with NaN
        rating_matrix = rating_matrix.clone()
        rating_matrix[rating_matrix == 0] = float('nan')

        batch_size, num_users, num_items = rating_matrix.size()
        sliding_windows = []

        # Apply attention across users
        attn_output = self.attention_layer(rating_matrix)

        # Slide over users rather than just ratings
        for i in range(num_users - self.window_size + 1):
            user_window = attn_output[:, i:i + self.window_size, :]
            user_window = user_window.view(batch_size, -1)  # Flatten the window
            sliding_windows.append(user_window)

        sliding_windows = torch.stack(sliding_windows)
        validity = self.fc(sliding_windows)
        return validity

class VAEBasedTrendCritic(nn.Module):
    def __init__(self, input_dim, latent_dim, vae_hidden_dim=128):
        super(VAEBasedTrendCritic, self).__init__()

        # VAE encoder
        self.fc1 = nn.Linear(input_dim, vae_hidden_dim)
        self.fc21 = nn.Linear(vae_hidden_dim, latent_dim)  # Mean vector
        self.fc22 = nn.Linear(vae_hidden_dim, latent_dim)  # Log-variance vector

        # VAE decoder
        self.fc3 = nn.Linear(latent_dim, vae_hidden_dim)
        self.fc4 = nn.Linear(vae_hidden_dim, input_dim)

        # Critic
        self.critic = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 32)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(32, 1))  # No Sigmoid, output raw score
        )

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, rating_matrix):
        # Replace 0s with NaN
        rating_matrix = rating_matrix.clone()
        rating_matrix[rating_matrix == 0] = float('nan')

        # Encode the ratings using VAE
        mu, logvar = self.encode(rating_matrix)
        z = self.reparameterize(mu, logvar)

        # Use DIANA for clustering (assume latent representations are the input)
        z_np = z.detach().cpu().numpy()
        clustering = AgglomerativeClustering(affinity='precomputed', linkage='complete')
        dist_matrix = 1 - cosine_similarity(z_np)
        cluster_labels = clustering.fit_predict(dist_matrix)

        # Feed the cluster centroids into the critic
        cluster_centroids = []
        for label in np.unique(cluster_labels):
            cluster_data = z_np[cluster_labels == label]
            centroid = np.nanmean(cluster_data, axis=0)
            cluster_centroids.append(centroid)

        cluster_centroids = torch.tensor(cluster_centroids, dtype=torch.float32)
        validity = self.critic(cluster_centroids)
        return validity
