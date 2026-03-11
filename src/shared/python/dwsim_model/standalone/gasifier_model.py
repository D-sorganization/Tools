import logging

from dwsim_model.constants import COMPOUNDS_STANDARD, DEFAULT_PROPERTY_PACKAGE
from dwsim_model.core import FlowsheetBuilder, get_automation
from dwsim_model.topology import build_gasifier_stage

logger = logging.getLogger(__name__)


class GasifierStandaloneFlowsheet:
    """
    Standalone Gasifier unit operation model using Downdraft concept.
    Designed for isolated testing and execution, following DbC and DRY.
    """

    def __init__(self, compound_set: list | None = None):
        self.automation = get_automation()
        self.builder = FlowsheetBuilder()
        self._is_built = False
        # Allow callers to pass a custom compound list; defaults to shared standard
        self._compounds = compound_set or COMPOUNDS_STANDARD

    def setup_thermo(self):
        """Configure standard PR properties and required components."""
        for c in self._compounds:
            self.builder.add_compound(c)
        self.builder.add_property_package(DEFAULT_PROPERTY_PACKAGE)

    def build_flowsheet(self):
        """Constructs an isolated Gasifier block with its immediate streams."""
        b = self.builder

        def safe_connect(src, tgt, p1=0, p2=0):
            try:
                b.connect(src, tgt, p1, p2)
            except Exception as e:
                logger.warning(f"Failed to connect isolated node: {src} -> {tgt} ({e})")

        build_gasifier_stage(b, "RCT_Conversion", safe_connect)

        # Basic reaction property assignment
        ops = self.builder.operations
        if "Downdraft_Gasifier" in ops:
            # DbC Placeholder: Users modify kinetics here.
            pass

        self._is_built = True

    def calculate(self):
        # AUTO-FIXED: Replaced assert with if-raise to prevent byte-code optimization removal
        if not self._is_built:
            raise RuntimeError("Flowsheet must be built before calculating.")
        self.builder.calculate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Building Standalone Gasifier...")
    m = GasifierStandaloneFlowsheet()
    m.setup_thermo()
    m.build_flowsheet()
    m.calculate()
    m.builder.save("Standalone_Gasifier.dwxml")
