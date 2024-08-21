import torch
import torch.nn as nn
import torch.optim as optim
from generators import (
    GroupLatentPreferenceGenerator,
    ItemLatentCharacteristicGenerator,
    GroupLatentBehaviorGenerator,
    ItemLatentInteractionGenerator,
    UserDenseRatingGenerator,
    UserRatingBehaviorGenerator,
    FinalSparseUserRatingGenerator
)
from discriminators import UserRatingCritic, VAEBasedTrendCritic

class WGAN:
    def __init__(self, config):
        # Initialize generators
        self.group_latent_preference_generator = GroupLatentPreferenceGenerator(
            config['noise_dim'], config['num_groups'], config['num_latent_factors'])
        self.item_latent_characteristic_generator = ItemLatentCharacteristicGenerator(
            config['noise_dim'], config['num_items'], config['num_latent_factors'])
        self.group_latent_behavior_generator = GroupLatentBehaviorGenerator(
            config['noise_dim'], config['num_groups'], config['num_latent_factors'])
        self.item_latent_interaction_generator = ItemLatentInteractionGenerator(
            config['noise_dim'], config['num_items'], config['num_latent_factors'])
        self.user_dense_rating_generator = UserDenseRatingGenerator(
            config['num_latent_factors'], config['num_items'], config['noise_dim'])
        self.user_rating_behavior_generator = UserRatingBehaviorGenerator(
            config['num_latent_factors'], config['num_items'], config['noise_dim'])
        self.final_sparse_user_rating_generator = FinalSparseUserRatingGenerator(
            config['num_items'])

        # Initialize critics
        self.individual_critic = UserRatingCritic(
            config['num_items'], config['window_size'])
        self.group_critic = VAEBasedTrendCritic(
            config['num_items'], config['num_latent_factors'])

        # Optimizers
        self.optimizer_g = optim.RMSprop(list(self.group_latent_preference_generator.parameters()) +
                                         list(self.item_latent_characteristic_generator.parameters()) +
                                         list(self.group_latent_behavior_generator.parameters()) +
                                         list(self.item_latent_interaction_generator.parameters()) +
                                         list(self.user_dense_rating_generator.parameters()) +
                                         list(self.user_rating_behavior_generator.parameters()) +
                                         list(self.final_sparse_user_rating_generator.parameters()), lr=config['lr'])

        self.optimizer_d = optim.RMSprop(list(self.individual_critic.parameters()) +
                                         list(self.group_critic.parameters()), lr=config['lr'])

    def train(self, real_data, num_epochs, critic_steps=5):
        for epoch in range(num_epochs):
            for i, real_batch in enumerate(real_data):
                # Extract the interaction matrix or the specific tensor you need
                real_batch_tensor = real_batch['interaction']  # Adjust the key accordingly
                print(f"Shape of real_batch_tensor: {real_batch_tensor.shape}")
                batch_size = real_batch_tensor.size(0)

                # Train critics multiple times per generator update
                for _ in range(critic_steps):
                    # Training the critics with real data
                    real_individual_scores = self.individual_critic(real_batch_tensor)
                    print(f"Real individual scores shape: {real_individual_scores.shape}")
                    real_group_scores = self.group_critic(real_batch_tensor)

                    # Generate fake data
                    noise = torch.randn(batch_size, self.config['noise_dim'])

                    group_latent_preferences = self.group_latent_preference_generator(noise)
                    item_latent_characteristics = self.item_latent_characteristic_generator(noise)
                    group_latent_behaviors = self.group_latent_behavior_generator(noise)
                    item_latent_interactions = self.item_latent_interaction_generator(noise)

                    user_specific_noise = torch.randn(batch_size, self.config['noise_dim'])
                    dense_user_ratings = self.user_dense_rating_generator(
                        group_latent_preferences, item_latent_characteristics, user_specific_noise)
                    rating_behavior = self.user_rating_behavior_generator(
                        group_latent_behaviors, item_latent_interactions, user_specific_noise)
                    fake_user_ratings = self.final_sparse_user_rating_generator(
                        dense_user_ratings, rating_behavior)

                    fake_individual_scores = self.individual_critic(fake_user_ratings)
                    fake_group_scores = self.group_critic(fake_user_ratings)

                    # Critic loss
                    d_loss = -torch.mean(real_individual_scores) + torch.mean(fake_individual_scores) \
                             - torch.mean(real_group_scores) + torch.mean(fake_group_scores)

                    self.optimizer_d.zero_grad()
                    d_loss.backward()
                    self.optimizer_d.step()

                # Train generator once
                fake_individual_scores = self.individual_critic(fake_user_ratings)
                fake_group_scores = self.group_critic(fake_user_ratings)

                g_loss = -torch.mean(fake_individual_scores) - torch.mean(fake_group_scores)

                self.optimizer_g.zero_grad()
                g_loss.backward()
                self.optimizer_g.step()

                if i % 100 == 0:
                    print(f'Epoch [{epoch}/{num_epochs}], Step [{i}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')


