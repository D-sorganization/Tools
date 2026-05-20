import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import SCADADataset
from .model import PlantSimulatorLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(db_path: str, save_path: str, epochs: int = 10, batch_size: int = 32) -> None:
    """Train the plant simulator Neural Network."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = SCADADataset(db_path=db_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PlantSimulatorLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for _batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural Network Plant Simulator")
    parser.add_argument(
        "--db-path", type=str, default="dcs_scada.db", help="Path to SCADA DB"
    )
    parser.add_argument(
        "--save-path", type=str, default="plant_model.pt", help="Path to save model"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    args = parser.parse_args()

    train(args.db_path, args.save_path, args.epochs)
