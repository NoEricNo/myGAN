import torch
import torch.nn as nn

class MultiChannelGenerator(nn.Module):
    def __init__(self, input_size, fc1_size, main_sizes, num_items, dropout_rate):
        super(MultiChannelGenerator, self).__init__()
        self.input_size = input_size
        self.num_items = num_items

        self.fc1 = nn.Linear(input_size, fc1_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.main_layers = nn.ModuleList()
        previous_size = fc1_size
        for size in main_sizes:
            self.main_layers.append(nn.Linear(previous_size, size))
            previous_size = size
        self.fc_final = nn.Linear(previous_size, num_items * 2)  # Output 2 channels (ratings and existence)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        for layer in self.main_layers:
            x = torch.relu(layer(x))

        output = self.fc_final(x)
        output = output.view(-1, 2, self.num_items)  # Reshape to (batch_size, 2 channels, num_items)

        rating_values, existence_flags = torch.chunk(output, 2, dim=1)  # Split into two channels
        rating_values = torch.sigmoid(self.fc_final(x)) * 4.5 + 0.5  # Ratings in the range [0.5, 5]  # Ratings in the range [0, 5]
        existence_flags = torch.sigmoid(existence_flags)  # Existence flags in the range [0, 1]

        return rating_values.squeeze(1).to_sparse(), existence_flags.squeeze(1).to_sparse()
