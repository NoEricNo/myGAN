import torch
import torch.nn as nn

class GroupLatentPreferenceGenerator(nn.Module):
    def __init__(self, noise_dim, num_groups, num_latent_factors):
        super(GroupLatentPreferenceGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_groups * num_latent_factors),
            nn.Tanh()  # Latent factors centered around 0
        )

    def forward(self, noise):
        group_latent_preferences = self.fc(noise)
        return group_latent_preferences.view(-1, num_groups, num_latent_factors)

class ItemLatentCharacteristicGenerator(nn.Module):
    def __init__(self, noise_dim, num_items, num_latent_factors):
        super(ItemLatentCharacteristicGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_items * num_latent_factors),
            nn.Tanh()  # Latent factors centered around 0
        )

    def forward(self, noise):
        item_latent_characteristics = self.fc(noise)
        return item_latent_characteristics.view(-1, num_items, num_latent_factors)

class GroupLatentBehaviorGenerator(nn.Module):
    def __init__(self, noise_dim, num_groups, num_latent_factors):
        super(GroupLatentBehaviorGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_groups * num_latent_factors),
            nn.Tanh()  # Latent factors centered around 0
        )

    def forward(self, noise):
        group_latent_behaviors = self.fc(noise)
        return group_latent_behaviors.view(-1, num_groups, num_latent_factors)

class ItemLatentInteractionGenerator(nn.Module):
    def __init__(self, noise_dim, num_items, num_latent_factors):
        super(ItemLatentInteractionGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_items * num_latent_factors),
            nn.Tanh()  # Latent factors centered around 0
        )

    def forward(self, noise):
        item_latent_interactions = self.fc(noise)
        return item_latent_interactions.view(-1, num_items, num_latent_factors)

class UserDenseRatingGenerator(nn.Module):
    def __init__(self, num_latent_factors, num_items, noise_dim):
        super(UserDenseRatingGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_items + noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_items),  # Generate dense user ratings
            nn.Tanh()  # Ratings centered around 0
        )

    def forward(self, group_latent_preferences, item_latent_characteristics, noise):
        # Explicit multiplication
        interaction = group_latent_preferences * item_latent_characteristics
        combined_input = torch.cat((interaction, noise), dim=1)
        dense_user_ratings = self.fc(combined_input)
        return dense_user_ratings


class UserRatingBehaviorGenerator(nn.Module):
    def __init__(self, num_latent_factors, num_items, noise_dim):
        super(UserRatingBehaviorGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_items + noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_items),  # Generate user rating behavior
            nn.Sigmoid()  # Output values between 0 and 1 to represent behavior
        )

    def forward(self, group_latent_behaviors, item_latent_interactions, noise):
        # Explicit multiplication
        interaction_behavior = group_latent_behaviors * item_latent_interactions
        combined_input = torch.cat((interaction_behavior, noise), dim=1)
        rating_behavior = self.fc(combined_input)
        return rating_behavior


class FinalSparseUserRatingGenerator(nn.Module):
    def __init__(self, num_items):
        super(FinalSparseUserRatingGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_items, 128),
            nn.ReLU(),
            nn.Linear(128, num_items),  # Generate final sparse user ratings
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, dense_user_ratings, rating_behavior):
        sparse_user_ratings = dense_user_ratings * rating_behavior
        final_user_ratings = self.fc(sparse_user_ratings) * 4 + 1  # Rescale to 1-5
        return final_user_ratings

if __name__ == '__main__':
    # Example usage and visualization
    noise_dim = 100
    num_groups = 10
    num_items = 50
    num_latent_factors = 20

    group_latent_preference_generator = GroupLatentPreferenceGenerator(noise_dim, num_groups, num_latent_factors)
    item_latent_characteristic_generator = ItemLatentCharacteristicGenerator(noise_dim, num_items, num_latent_factors)
    group_latent_behavior_generator = GroupLatentBehaviorGenerator(noise_dim, num_groups, num_latent_factors)
    item_latent_interaction_generator = ItemLatentInteractionGenerator(noise_dim, num_items, num_latent_factors)
    user_dense_rating_generator = UserDenseRatingGenerator(num_latent_factors, num_items, noise_dim)
    user_rating_behavior_generator = UserRatingBehaviorGenerator(num_latent_factors, num_items, noise_dim)
    final_sparse_user_rating_generator = FinalSparseUserRatingGenerator(num_items)

    noise = torch.randn(10, noise_dim)  # Generate a batch of noise vectors

    group_latent_preferences = group_latent_preference_generator(noise)  # Generate group latent preferences
    item_latent_characteristics = item_latent_characteristic_generator(noise)  # Generate item latent characteristics
    group_latent_behaviors = group_latent_behavior_generator(noise)  # Generate group latent behaviors
    item_latent_interactions = item_latent_interaction_generator(noise)  # Generate item latent interactions

    user_specific_noise = torch.randn(10, noise_dim)  # Generate user-specific noise
    dense_user_ratings = user_dense_rating_generator(group_latent_preferences, item_latent_characteristics, user_specific_noise)  # Generate dense user ratings
    rating_behavior = user_rating_behavior_generator(group_latent_behaviors, item_latent_interactions, user_specific_noise)  # Generate user rating behavior
    final_user_ratings = final_sparse_user_rating_generator(dense_user_ratings, rating_behavior)  # Generate final sparse user ratings

    print("Group Latent Preferences:", group_latent_preferences)
    print("Item Latent Characteristics:", item_latent_characteristics)
    print("Group Latent Behaviors:", group_latent_behaviors)
    print("Item Latent Interactions:", item_latent_interactions)
    print("Dense User Ratings:", dense_user_ratings)
    print("Rating Behavior:", rating_behavior)
    print("Final User Ratings:", final_user_ratings)

