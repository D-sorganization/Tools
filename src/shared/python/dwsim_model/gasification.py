import logging
from enum import Enum

from dwsim_model.config_loader import ConfigLoader
from dwsim_model.constants import COMPOUNDS_STANDARD, DEFAULT_PROPERTY_PACKAGE
from dwsim_model.core import FlowsheetBuilder
from dwsim_model.topology import (
    build_gasifier_stage,
    build_pem_stage,
    build_trc_stage,
)

logger = logging.getLogger(__name__)


class ReactorMode(str, Enum):
    MIXED = "mixed"  # Gasifier: Conversion, PEM: Equilibrium, TRC: PFR
    KINETIC = "kinetic"  # All: PFR
    EQUILIBRIUM = "equilibrium"  # All: Equilibrium
    CONVERSION = "conversion"  # All: Conversion
    CUSTOM = "custom"  # Defined via a dictionary


class GasificationFlowsheet:
    """
    Constructs the standard Gasification Process model in DWSIM.

    Includes:
    - Downdraft Gasifier (RCT_Conversion)
    - PEM / Plasma Entrained Melter (RCT_Equilibrium)
    - TRC / Thermal Reduction Chamber (RCT_PFR)
    - Quench Vessel (Mixer for evaporative cooling via water injection)
    - Baghouse (SolidSeparator)
    - Scrubber (ComponentSeparator)
    - Blower/Compressor (Compressor)

    Parameters
    ----------
    builder:
        Optional pre-constructed FlowsheetBuilder.  If None, one is created.
    mode:
        Reactor fidelity mode — see ReactorMode for options.
    custom_reactors:
        When mode=CUSTOM, provide a dict like
        {"gasifier": "RCT_Conversion", "pem": "RCT_Gibbs", "trc": "RCT_PFR"}.
    config_path:
        Path to the master config YAML (or legacy JSON).  Defaults to the
        project-root ``config/master_config.yaml`` if it exists, otherwise
        falls back to ``config/feed_conditions.json``.
    compound_set:
        List of DWSIM compound names.  Defaults to COMPOUNDS_STANDARD from
        constants.py — edit that file to add new species.
    """

    def __init__(
        self,
        builder: FlowsheetBuilder | None = None,
        mode: ReactorMode | str = ReactorMode.MIXED,
        custom_reactors: dict[str, str] | None = None,
        config_path: str | None = None,
        compound_set: list[str] | None = None,
        runtime_config: dict | None = None,
    ):
        self.builder = builder or FlowsheetBuilder()
        self.mode = ReactorMode(mode)
        self.custom_reactors = custom_reactors or {}
        self.config_path = config_path  # None → ConfigLoader picks sensible default
        self.compound_set = compound_set or list(COMPOUNDS_STANDARD)
        self.runtime_config = (
            dict(runtime_config) if runtime_config is not None else None
        )
        self._injected_config = self.runtime_config
        self._is_built = False

    # ──────────────────────────────────────────────────────────────────────────
    # Reactor type selection
    # ──────────────────────────────────────────────────────────────────────────

    def _get_reactor_types(self) -> dict[str, str]:
        """Returns the DWSIM ObjectType string for each reactor based on mode."""
        if self.mode == ReactorMode.CUSTOM:
            return {
                "gasifier": self.custom_reactors.get("gasifier", "RCT_Conversion"),
                "pem": self.custom_reactors.get("pem", "RCT_Equilibrium"),
                "trc": self.custom_reactors.get("trc", "RCT_PFR"),
            }
        elif self.mode == ReactorMode.KINETIC:
            return {"gasifier": "RCT_PFR", "pem": "RCT_PFR", "trc": "RCT_PFR"}
        elif self.mode == ReactorMode.EQUILIBRIUM:
            return {
                "gasifier": "RCT_Equilibrium",
                "pem": "RCT_Equilibrium",
                "trc": "RCT_Equilibrium",
            }
        elif self.mode == ReactorMode.CONVERSION:
            return {
                "gasifier": "RCT_Conversion",
                "pem": "RCT_Conversion",
                "trc": "RCT_Conversion",
            }
        else:  # MIXED (default)
            return {
                "gasifier": "RCT_Conversion",
                "pem": "RCT_Equilibrium",
                "trc": "RCT_PFR",
            }

    # ──────────────────────────────────────────────────────────────────────────
    # Thermodynamics setup
    # ──────────────────────────────────────────────────────────────────────────

    def setup_thermo(self) -> None:
        """Sets up thermodynamics and compounds. DbC: Builder must exist."""
        # AUTO-FIXED: Replaced assert with if-raise to prevent byte-code optimization removal
        if self.builder is None:
            raise RuntimeError("Builder instance required")

        for compound in self.compound_set:
            self.builder.add_compound(compound)

        self.builder.add_property_package(DEFAULT_PROPERTY_PACKAGE)
        logger.info(
            f"Thermo setup complete: {len(self.compound_set)} compounds, "
            f"package={DEFAULT_PROPERTY_PACKAGE}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Flowsheet construction
    # ──────────────────────────────────────────────────────────────────────────

    def build_flowsheet(self) -> None:
        """Constructs all unit operations and streams then wires them together."""
        if self._is_built:
            logger.debug(
                "build_flowsheet() called on already-built flowsheet; skipping."
            )
            return

        b = self.builder
        rtypes = self._get_reactor_types()
        connection_failures: list[str] = []

        # ──────────────────────────────────────────────────────────────────────
        # 4. Quench Vessel
        # ──────────────────────────────────────────────────────────────────────
        water_inj = b.add_object("MaterialStream", "Quench_Water_Injection", 1950, 200)
        n2_inj = b.add_object("MaterialStream", "Quench_Nitrogen", 1950, 250)
        stm_inj = b.add_object("MaterialStream", "Quench_Steam", 1950, 300)

        quench = b.add_object("Mixer", "Quench_Vessel", 2100, 350)
        s4 = b.add_object("MaterialStream", "Syngas_Pre_Baghouse", 2200, 350)

        # ──────────────────────────────────────────────────────────────────────
        # 5. Baghouse
        # ──────────────────────────────────────────────────────────────────────
        baghouse = b.add_object("SolidSeparator", "Baghouse", 2350, 350)
        s5 = b.add_object("MaterialStream", "Clean_Syngas_Pre_Scrub", 2450, 350)
        baghouse_solids = b.add_object(
            "MaterialStream", "Baghouse_Solids_Out", 2450, 450
        )

        # ──────────────────────────────────────────────────────────────────────
        # 6. Scrubber
        # ──────────────────────────────────────────────────────────────────────
        scrubber = b.add_object("ComponentSeparator", "Scrubber", 2600, 350)
        s6 = b.add_object("MaterialStream", "Scrubbed_Syngas", 2700, 350)
        scrubber_blowdown = b.add_object(
            "MaterialStream", "Scrubber_Blowdown", 2700, 450
        )

        # ──────────────────────────────────────────────────────────────────────
        # 7. Blower
        # ──────────────────────────────────────────────────────────────────────
        blower = b.add_object("Compressor", "Blower", 2850, 350)
        product = b.add_object("MaterialStream", "Final_Syngas", 2950, 350)
        e_blower = b.add_object("EnergyStream", "E_Blower", 2850, 250)

        # ──────────────────────────────────────────────────────────────────────
        # Connections
        # ──────────────────────────────────────────────────────────────────────
        def safe_connect(src, tgt, p1=0, p2=0):
            try:
                b.connect(src, tgt, p1, p2)
            except Exception as exc:
                src_name = getattr(src, "Name", str(src))
                tgt_name = getattr(tgt, "Name", str(tgt))
                message = (
                    f"Failed to connect {src_name} -> {tgt_name} "
                    f"(ports {p1}->{p2}): {exc}"
                )
                connection_failures.append(message)
                logger.error(message)

        gasifier_stage = build_gasifier_stage(b, rtypes["gasifier"], safe_connect)
        s1 = gasifier_stage["syngas_out"]
        build_pem_stage(
            b,
            rtypes["pem"],
            safe_connect,
            syngas_inlet=s1,
        )
        s2 = b.materials["Syngas_Pre_TRC"]
        build_trc_stage(
            b,
            rtypes["trc"],
            safe_connect,
            syngas_inlet=s2,
        )
        s3 = b.materials["Syngas_Pre_Quench"]

        # Quench
        safe_connect(s3, quench, 0, 0)
        safe_connect(water_inj, quench, 0, 1)
        safe_connect(n2_inj, quench, 0, 2)
        safe_connect(stm_inj, quench, 0, 3)
        safe_connect(quench, s4, 0, 0)

        # Baghouse
        safe_connect(s4, baghouse, 0, 0)
        safe_connect(baghouse, s5, 0, 0)
        safe_connect(baghouse, baghouse_solids, 1, 0)

        # Scrubber
        safe_connect(s5, scrubber, 0, 0)
        safe_connect(scrubber, s6, 0, 0)
        safe_connect(scrubber, scrubber_blowdown, 1, 0)

        # Blower
        safe_connect(s6, blower, 0, 0)
        safe_connect(blower, product, 0, 0)
        safe_connect(e_blower, blower, 0, 1)

        if connection_failures:
            raise RuntimeError(
                "Critical flowsheet connections failed:\n"
                + "\n".join(f"- {failure}" for failure in connection_failures)
            )

        # ──────────────────────────────────────────────────────────────────────
        # Post-connection configuration
        # ──────────────────────────────────────────────────────────────────────
        self._configure_reactors()
        self._load_config()

        self._is_built = True
        logger.info("Flowsheet build complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # Reactor configuration
    # ──────────────────────────────────────────────────────────────────────────

    def _configure_reactors(self) -> None:
        """
        Wires reactor parameters from the chemistry module into DWSIM.

        The actual reaction definitions live in
        ``dwsim_model.chemistry.reactions`` so they can be edited without
        touching this file.  We import lazily here so the rest of the
        flowsheet can still build even if the chemistry module is missing.
        """
        from dwsim_model.chemistry.reactions import (
            configure_gasifier,
            configure_pem,
            configure_trc,
        )

        ops = self.builder.operations

        if "Downdraft_Gasifier" in ops:
            try:
                configure_gasifier(ops["Downdraft_Gasifier"], self.builder.sim)
                logger.info("Gasifier reactions configured.")
            except Exception as exc:
                raise RuntimeError(
                    f"Gasifier reactor configuration failed: {exc}"
                ) from exc

        if "PEM_Reactor" in ops:
            try:
                configure_pem(ops["PEM_Reactor"], self.builder.sim)
                logger.info("PEM reactions configured.")
            except Exception as exc:
                raise RuntimeError(f"PEM reactor configuration failed: {exc}") from exc

        if "TRC_Reactor" in ops:
            try:
                configure_trc(ops["TRC_Reactor"], self.builder.sim)
                logger.info("TRC reactor configured.")
            except Exception as exc:
                raise RuntimeError(f"TRC reactor configuration failed: {exc}") from exc

    # ──────────────────────────────────────────────────────────────────────────
    # Config loading  (BUG FIX: was passing b.energies, now b.energy_streams)
    # ──────────────────────────────────────────────────────────────────────────

    def _load_config(self) -> None:
        """Apply external config to the flowsheet after connections are built."""
        b = self.builder
        try:
            runtime_config = self.runtime_config or self._injected_config
            loader = ConfigLoader(
                config_path=self.config_path,
                config_data=runtime_config,
            )
            loader.load()
            loader.apply_to_flowsheet(b, b.materials, b.energy_streams)
            logger.info("External config applied successfully.")
        except Exception as exc:
            raise RuntimeError(f"Failed to apply external config: {exc}") from exc

    # ──────────────────────────────────────────────────────────────────────────
    # Run
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the configured flowsheet."""
        # AUTO-FIXED: Replaced assert with if-raise to prevent byte-code optimization removal
        if not self._is_built:
            raise RuntimeError("Flowsheet must be built before running")
        self.builder.calculate()
