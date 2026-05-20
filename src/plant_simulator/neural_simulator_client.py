import logging
from typing import Optional

import torch

from p1am_control_system.backend.models import RoutingConfig
from p1am_control_system.backend.plc_interface import BasePLCClient

from .model import PlantSimulatorLSTM

logger = logging.getLogger(__name__)


class NeuralSimulatorClient(BasePLCClient):
    """
    Concrete PLC client simulating plant process dynamics using a Neural Network.
    Adheres to the BasePLCClient interface.
    """

    def __init__(
        self, model_path: str = "plant_model.pt", sequence_length: int = 10
    ) -> None:
        super().__init__()
        self._connected = True
        self.simulated_tags: list[float] = [0.0] * 32
        self.active_config = RoutingConfig(
            input_routing=[0, 1, 2, 3, 4, 5],
            output_routing=[20, 21],
            pids=[
                {
                    "pv_tag_id": 5,
                    "cv_tag_id": 6,
                    "setpoint": 50.0,
                    "kp": 1.0,
                    "ki": 0.5,
                    "kd": 0.1,
                },
                {
                    "pv_tag_id": 15,
                    "cv_tag_id": 16,
                    "setpoint": 50.0,
                    "kp": 1.0,
                    "ki": 0.5,
                    "kd": 0.1,
                },
                {
                    "pv_tag_id": 25,
                    "cv_tag_id": 26,
                    "setpoint": 50.0,
                    "kp": 1.0,
                    "ki": 0.5,
                    "kd": 0.1,
                },
                {
                    "pv_tag_id": 27,
                    "cv_tag_id": 28,
                    "setpoint": 50.0,
                    "kp": 1.0,
                    "ki": 0.5,
                    "kd": 0.1,
                },
            ],
            interlocks=[
                {
                    "lolo_limit": -10.0,
                    "low_limit": 0.0,
                    "high_limit": 100.0,
                    "hihi_limit": 110.0,
                }
                for _ in range(32)
            ],
        )

        self.sequence_length = sequence_length
        self.history = []

        # Load the PyTorch Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PlantSimulatorLSTM().to(self.device)
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            logger.info(f"Loaded Neural Simulator Model from {model_path}")
        except FileNotFoundError:
            logger.warning(
                f"Model file {model_path} not found. "
                "Running with uninitialized weights."
            )

        self.model.eval()

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    async def read_tags(self) -> list[float] | None:
        """Perform a single step of the plant simulation using the Neural Network."""
        async with self.lock:
            if not self._connected:
                return None

            # Keep a sliding window of recent tags
            self.history.append(list(self.simulated_tags))
            if len(self.history) > self.sequence_length:
                self.history.pop(0)

            if len(self.history) == self.sequence_length:
                # Predict next state
                with torch.no_grad():
                    x = torch.tensor([self.history], dtype=torch.float32).to(
                        self.device
                    )
                    y_pred = self.model(x)
                    next_tags = y_pred.cpu().numpy()[0]

                # Update tags with prediction (basic PID control inputs can be
                # overridden by users elsewhere)
                # by users elsewhere)
                for i in range(len(self.simulated_tags)):
                    self.simulated_tags[i] = float(next_tags[i])
            else:
                # Not enough history yet, return current tags
                pass

            return list(self.simulated_tags)

    async def read_routing(self) -> Optional["RoutingConfig"]:
        async with self.lock:
            return self.active_config

    async def write_routing(self, config: "RoutingConfig") -> bool:
        async with self.lock:
            self.active_config = config
            return True

    async def save_to_flash(self) -> bool:
        return True

    async def trigger_estop(self) -> bool:
        async with self.lock:
            self.simulated_tags = [0.0] * 32
            return True

    async def write_tag(self, tag_id: int, value: float) -> bool:
        async with self.lock:
            if 0 <= tag_id < len(self.simulated_tags):
                self.simulated_tags[tag_id] = value
                return True
            return False
