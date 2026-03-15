import logging

from dwsim_model.standalone.base import StandaloneBase
from dwsim_model.topology import build_pem_stage

logger = logging.getLogger(__name__)


class PEMStandaloneFlowsheet(StandaloneBase):
    """
    Standalone Plasma Entrained Melting (PEM) model.
    Designed for isolated testing and configuration (DbC, DRY).
    """

    def build_flowsheet(self) -> None:
        """Constructs an isolated PEM block."""
        build_pem_stage(self.builder, "RCT_Equilibrium", self._safe_connect)
        self._is_built = True


def main() -> PEMStandaloneFlowsheet:
    logging.basicConfig(level=logging.INFO)
    logger.info("Building Standalone PEM...")
    m = PEMStandaloneFlowsheet()
    m.setup_thermo()
    m.build_flowsheet()
    m.calculate()
    m.builder.save("Standalone_PEM.dwxml")
    return m


if __name__ == "__main__":
    main()
