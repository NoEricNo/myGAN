import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, dropout_rate):
        super(Discriminator, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = None
        self.fc2 = nn.Linear(128, 128)  # Adjust this as needed
        self.fc3 = nn.Linear(128, 64)
        self.validity = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_fc1(self, input_size):
        self.fc1 = nn.Linear(input_size, 128).to(self.device)  # Change 128 to your desired size

    def forward(self, rating_values, rating_existence, user_ids, movie_ids):
        batch_size = rating_values.size(0)
        num_movies_chunk = rating_values.size(1)

        rating_values = rating_values.to_dense().view(batch_size, -1).float().to(self.device)
        rating_existence = rating_existence.to_dense().view(batch_size, -1).float().to(self.device)
        user_ids = user_ids.view(batch_size, 1).float().to(self.device)
        movie_ids = movie_ids.view(batch_size, -1).float().to(self.device)

        x = torch.cat((rating_values, rating_existence, user_ids, movie_ids), dim=1)

        if self.fc1 is None or self.fc1.in_features != x.size(1):
            self.initialize_fc1(x.size(1))

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        validity = self.validity(x)

        return validity.view(batch_size, 1)
