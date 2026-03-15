import logging

from dwsim_model.standalone.base import StandaloneBase
from dwsim_model.topology import build_gasifier_stage

logger = logging.getLogger(__name__)


class GasifierStandaloneFlowsheet(StandaloneBase):
    """
    Standalone Gasifier unit operation model using Downdraft concept.
    Designed for isolated testing and execution, following DbC and DRY.
    """

    def build_flowsheet(self) -> None:
        """Constructs an isolated Gasifier block with its immediate streams."""
        build_gasifier_stage(self.builder, "RCT_Conversion", self._safe_connect)

        # Basic reaction property assignment
        ops = self.builder.operations
        if "Downdraft_Gasifier" in ops:
            # DbC Placeholder: Users modify kinetics here.
            pass

        self._is_built = True


def main() -> GasifierStandaloneFlowsheet:
    logging.basicConfig(level=logging.INFO)
    logger.info("Building Standalone Gasifier...")
    m = GasifierStandaloneFlowsheet()
    m.setup_thermo()
    m.build_flowsheet()
    m.calculate()
    m.builder.save("Standalone_Gasifier.dwxml")
    return m


if __name__ == "__main__":
    main()
