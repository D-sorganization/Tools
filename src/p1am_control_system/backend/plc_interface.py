import abc
import asyncio
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from models import RoutingConfig


class BasePLCClient(abc.ABC):
    """Abstract base class representing a generic PLC client interface.

    Allows the SCADA system to operate with different hardware controllers.
    """

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.tuning_sessions: dict[int, dict] = {}

    @property
    @abc.abstractmethod
    def connected(self) -> bool:
        """Indicate whether the client is currently connected to the PLC."""
        pass

    @abc.abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the PLC.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        pass

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the PLC."""
        pass

    @abc.abstractmethod
    async def read_tags(self) -> list[float] | None:
        """Read all SCADA tags (typically 32 floats) from the PLC.

        Returns:
            Optional[list[float]]: The tag values if successful, or None on error.
        """
        pass

    @abc.abstractmethod
    async def read_routing(self) -> Optional["RoutingConfig"]:
        """Read the DCS routing matrix configuration from the PLC.

        Returns:
            Optional[RoutingConfig]: The current routing config, or None on error.
        """
        pass

    @abc.abstractmethod
    async def write_routing(self, config: "RoutingConfig") -> bool:
        """Write the DCS routing matrix configuration to the PLC.

        Args:
            config: The routing configuration model to deploy.

        Returns:
            bool: True if write is successful, False otherwise.
        """
        pass

    @abc.abstractmethod
    async def save_to_flash(self) -> bool:
        """Command the PLC to save current configuration to flash/NVRAM.

        Returns:
            bool: True if successful, False otherwise.
        """
        pass

    @abc.abstractmethod
    async def trigger_estop(self) -> bool:
        """Send an emergency stop/shutdown command to the PLC.

        Returns:
            bool: True if successful, False otherwise.
        """
        pass

    @abc.abstractmethod
    async def write_tag(self, tag_id: int, value: float) -> bool:
        """Directly write or override a tag value on the PLC.

        Args:
            tag_id: The logical ID of the tag.
            value: The float value to write.

        Returns:
            bool: True if successful, False otherwise.
        """
        pass
