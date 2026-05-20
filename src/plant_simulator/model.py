import torch
import torch.nn as nn


class PlantSimulatorLSTM(nn.Module):
    """
    LSTM-based Neural Network for simulating plant dynamics.
    Takes a sequence of historical tags and control inputs, and predicts the next state.
    """

    def __init__(self, num_tags: int = 32, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.num_tags = num_tags
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=num_tags,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )

        self.fc = nn.Linear(hidden_size, num_tags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, sequence_length, num_tags)
        Returns: (batch_size, num_tags) representing the predicted state at t+1
        """
        # lstm_out: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]

        # Predict the next state
        out = self.fc(last_out)
        return out
