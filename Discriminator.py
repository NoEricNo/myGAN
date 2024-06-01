import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_size, fc1_size, main_sizes, dropout_rate=0.3):
        super(Discriminator, self).__init__()
        self.fc1_size = fc1_size
        self.main_sizes = main_sizes
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.main = nn.Sequential(
            nn.Linear(fc1_size, main_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(main_sizes[0], main_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.rating_output = nn.Linear(main_sizes[1], 1)
        self.existence_output_size = main_sizes[1] // 2
        self.existence_output = nn.Sequential(
            nn.Linear(main_sizes[1], self.existence_output_size),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        print(f"Discriminator input shape: {input_data.shape}")
        x = self.fc1(input_data)
        print(f"After fc1 shape: {x.shape}")
        x = self.main(x)
        print(f"After main shape: {x.shape}")
        rating_validity = self.rating_output(x)
        print(f"Rating validity shape: {rating_validity.shape}")
        existence_prediction = self.existence_output(x)
        print(f"Existence prediction shape: {existence_prediction.shape}")
        return rating_validity, existence_prediction