import logging

from dwsim_model.standalone.base import StandaloneBase
from dwsim_model.topology import build_trc_stage

logger = logging.getLogger(__name__)


class TRCStandaloneFlowsheet(StandaloneBase):
    """
    Standalone Thermal Reduction Chamber (TRC) model.
    Designed for isolated testing and configuration (DbC, DRY).
    """

    def build_flowsheet(self) -> None:
        """Constructs an isolated TRC block."""
        build_trc_stage(self.builder, "RCT_PFR", self._safe_connect)
        self._is_built = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Building Standalone TRC...")
    m = TRCStandaloneFlowsheet()
    m.setup_thermo()
    m.build_flowsheet()
    m.calculate()
    m.builder.save("Standalone_TRC.dwxml")
