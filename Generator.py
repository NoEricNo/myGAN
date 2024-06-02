import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_size, fc1_size, main_sizes, existence_gen_size, dropout_rate):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.main_layers = nn.ModuleList()
        previous_size = fc1_size
        for size in main_sizes:
            self.main_layers.append(nn.Linear(previous_size, size))
            previous_size = size
        self.fc_ratings = nn.Linear(previous_size, existence_gen_size)
        self.fc_existence = nn.Linear(previous_size, existence_gen_size)

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        for layer in self.main_layers:
            x = torch.relu(layer(x))
        ratings = self.fc_ratings(x)
        existence = self.fc_existence(x)
        return ratings, existence
