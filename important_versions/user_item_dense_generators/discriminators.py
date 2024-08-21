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
        self.embedding = nn.Linear(num_items, embed_dim)  # Ensure embed_dim is consistent

    def forward(self, rating_matrix):
        print(f"Shape before embedding: {rating_matrix.shape}")  # Print input shape
        embedded_users = self.embedding(rating_matrix)
        print(f"Shape after embedding: {embedded_users.shape}")  # Print output shape
        # Transpose to (num_users, batch_size, embed_dim) for attention
        embedded_users = embedded_users.transpose(0, 1)
        print(f"Shape after transpose: {embedded_users.shape}")  # Print shape after transpose
        # Apply self-attention across users
        attn_output, _ = self.attention(embedded_users, embedded_users, embedded_users)
        print(f"Shape after attention: {attn_output.shape}")  # Print shape after attention
        # Transpose back to (batch_size, num_users, embed_dim)
        attn_output = attn_output.transpose(0, 1)
        return attn_output


class UserRatingCritic(nn.Module):
    def __init__(self, num_items, window_size=5, embed_dim=64, num_heads=4):
        super(UserRatingCritic, self).__init__()
        self.window_size = window_size  # How many users to compare simultaneously
        self.attention_layer = UserAttention(num_items, embed_dim, num_heads)
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(embed_dim * window_size, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 1))  # No Sigmoid, output raw score
        )

    def forward(self, rating_matrix):
        print(f"Initial shape: {rating_matrix.shape}")  # Print initial shape

        # Apply attention across users
        attn_output = self.attention_layer(rating_matrix)
        print(f"Shape after attention: {attn_output.shape}")  # Print shape after attention

        # Slide over the batch dimension to compare groups of users
        sliding_windows = []
        batch_size = attn_output.size(0)

        # Adjusting the sliding window to correctly handle 2D data
        for i in range(batch_size - self.window_size + 1):  # Sliding over total users in batch
            user_window = attn_output[i:i + self.window_size, :]  # Index only on two dimensions
            #print(f"Shape of user_window: {user_window.shape}")  # Print shape of the sliding window
            user_window = user_window.view(self.window_size, -1)  # Flatten the window if needed
            sliding_windows.append(user_window)

        sliding_windows = torch.stack(sliding_windows)
        print(f"After sliding window: {sliding_windows.shape}")  # Print shape after stacking windows

        sliding_windows = sliding_windows.view(-1, self.window_size * attn_output.size(-1))  # Ensure correct flattening
        print(f"After flattening: {sliding_windows.shape}")  # Print shape after flattening

        validity = self.fc(sliding_windows)
        print(f"Final output shape: {validity.shape}")  # Print final output shape

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

