import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class UserAttention(nn.Module):
    def __init__(self, num_items, embed_dim, num_heads, window_size):
        super(UserAttention, self).__init__()
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.embedding = nn.Linear(num_items, embed_dim)

    def forward(self, rating_matrix):
        # rating_matrix shape: (batch_size, num_items)
        batch_size, num_items = rating_matrix.shape

        # Embed each user's rating vector
        embedded_users = self.embedding(rating_matrix)  # Shape: (batch_size, embed_dim)
        print(f"Shape after embedding: {embedded_users.shape}")  # Shape: [batch_size, embed_dim]

        # Reshape to introduce window_size (sequence_length)
        # For attention, we want shape [window_size, batch_size, embed_dim]
        # For simplicity, assuming window_size <= batch_size
        embedded_users = embedded_users.view(self.window_size, batch_size // self.window_size, -1)
        print(
            f"Shape after introducing window size: {embedded_users.shape}")  # Shape: [window_size, num_windows, embed_dim]

        # Apply self-attention across users
        attn_output, _ = self.attention(embedded_users, embedded_users, embedded_users)
        print(f"Shape after attention: {attn_output.shape}")  # Shape: [window_size, num_windows, embed_dim]

        return attn_output


class UserRatingCritic(nn.Module):
    def __init__(self, num_items, embed_dim, num_heads, window_size):
        super(UserRatingCritic, self).__init__()
        self.window_size = window_size
        self.attention_layer = UserAttention(num_items, embed_dim, num_heads, window_size)
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(embed_dim * window_size, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 1))
        )

    def forward(self, rating_matrix):
        # rating_matrix shape: (batch_size, num_items)
        print(f"Initial shape: {rating_matrix.shape}")

        attn_output = self.attention_layer(rating_matrix)
        print(f"Shape after attention: {attn_output.shape}")  # [window_size, num_windows, embed_dim]

        # Flatten the attention output for feeding into the fully connected layers
        attn_output = attn_output.view(-1, self.window_size * attn_output.shape[-1])
        print(f"Shape after flattening: {attn_output.shape}")  # [num_windows, window_size * embed_dim]

        validity = self.fc(attn_output)
        print(f"Final output shape: {validity.shape}")  # [num_windows, 1]

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
        print(f"Shape after first layer (fc1): {h1.shape}")  # Print shape
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        print(f"Shape after reparameterization (z): {z.shape}")  # Print shape
        return z

    def forward(self, rating_matrix):
        # Replace 0s with NaN
        rating_matrix = rating_matrix.clone()
        rating_matrix[rating_matrix == 0] = float('nan')

        # Normalize the rating matrix
        rating_matrix = rating_matrix / torch.nanmax(rating_matrix)  # Normalize to [0, 1]

        # Encode the ratings using VAE
        mu, logvar = self.encode(rating_matrix)
        z = self.reparameterize(mu, logvar)

        # Convert to numpy
        z_np = z.detach().cpu().numpy()

        # Use DIANA for clustering (assume latent representations are the input)
        clustering = AgglomerativeClustering(metric='precomputed', linkage='complete')
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

