from .NO_basemodel import NO_basemodel
import torch
import torch.nn as nn


class CNN_model(NO_basemodel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = quickCNN(1, 1, 3, 4)
        self.learning_rate = self.param['lr']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.1)
        

class quickCNN(nn.Module):
    def __init__(self, N_input, N_output, N_Hidden, N_layers, kernel_size=3, num_filters=16):
        """

        Args:
            N_input (int): Input feature size.
            N_output (int): Output size.
            N_Hidden (int): Number of hidden neurons (used for FC layer after CNN).
            N_layers (int): Number of hidden layers.
            kernel_size (int): Size of convolutional kernel.
            num_filters (int): Number of filters in convolution layers.
        """
        super(quickCNN, self).__init__()
        activation = nn.Tanh()

        # Initial Convolutional layer (Mimics First Linear Layer)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=N_input, out_channels=num_filters, kernel_size=kernel_size, padding=1),
            activation
        )

        # Hidden Convolutional Layers (Mimics Hidden Fully Connected Layers)
        self.conv_hidden = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding=1),
                    activation
                ) for _ in range(N_layers)
            ]
        )

        # Fully Connected Layer (To map CNN features to output)
        self.fc = nn.Sequential(
            nn.Linear(num_filters, N_Hidden),  # Flattened CNN output
            activation,
            nn.Linear(N_Hidden, N_output)
        )

    def forward(self, x):
        """
        Forward pass through the CNN model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, N_input)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, N_output)
        """
        # Reshape input to fit CNN (Batch, Channels, Features)
        #x = x.unsqueeze(1)  # Convert (Batch, Features) -> (Batch, Channels=1, Features)

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv_hidden(x)

        # Flatten the CNN output for fully connected layer
        x = x.permute(0, 2, 3, 1)
        
        # Fully connected output mapping
        x = self.fc(x)
        x = x.permute(0, 3, 1, 2)

        return x