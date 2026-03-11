import logging
import os
import sys

logger = logging.getLogger(__name__)

# AUTO-FIXED: Added basic logging

_interf = None
_ObjectType = None


def get_automation(dwsim_path: str | None = None):
    if dwsim_path is None:
        dwsim_path = os.environ.get("DWSIM_PATH", r"C:\Users\diete\AppData\Local\DWSIM")

    global _interf, _ObjectType
    if _interf is not None:
        return _interf, _ObjectType

    if dwsim_path not in sys.path:
        sys.path.append(dwsim_path)

    import clr

    try:
        # Load assemblies if not already loaded
        clr.AddReference(os.path.join(dwsim_path, "DWSIM.Automation.dll"))
        clr.AddReference(os.path.join(dwsim_path, "DWSIM.Interfaces.dll"))
        clr.AddReference(os.path.join(dwsim_path, "DWSIM.GlobalSettings.dll"))

        from DWSIM.Automation import Automation3
        from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType

        _interf = Automation3()
        _ObjectType = ObjectType
        logger.info("Successfully loaded DWSIM automation.")
    except Exception as e:
        logger.error(f"Failed to load DWSIM automation from {dwsim_path}: {e}")
        raise RuntimeError(f"Could not load DWSIM automation from {dwsim_path}: {e}")

    return _interf, _ObjectType


class FlowsheetBuilder:
    """
    Wrapper for DWSIM Automation3 to build the Gasification Process.
    """

    def __init__(self, dwsim_path: str | None = None):
        self.interf, self.ObjectType = get_automation(dwsim_path)

        self.sim = self.interf.CreateFlowsheet()
        self.materials: dict[str, object] = {}
        self.energy_streams: dict[str, object] = {}
        self.operations: dict[str, object] = {}

    def add_compound(self, name: str) -> None:
        """Add a compound to the simulation."""
        if not name:
            raise ValueError("Compound name cannot be empty")
        try:
            self.sim.AddCompound(name)
            logger.debug(f"Added compound '{name}' to the simulation.")
        except Exception as e:
            logger.error(f"Failed to add compound '{name}': {e}")
            raise ValueError(f"Failed to add compound '{name}': {e}")

    def add_property_package(self, package_name: str = "Peng-Robinson (PR)") -> object:
        """Add a property package to the flowsheet."""
        if not package_name:
            raise ValueError("Package name cannot be empty")
        try:
            pp_dict = getattr(self.interf, "AvailablePropertyPackages", {})
            if package_name not in pp_dict:
                raise ValueError(
                    f"Package {package_name} not found. Available: {list(pp_dict)}"
                )
            prop_pack = pp_dict[package_name]
            self.sim.AddPropertyPackage(prop_pack)
            logger.debug(f"Added property package '{package_name}'.")
            return prop_pack
        except Exception as e:
            logger.error(f"Failed to load property package {package_name}: {e}")
            raise ValueError(f"Failed to load property package {package_name}: {e}")

    def add_object(
        self, obj_type_name: str, name: str, x: int = 0, y: int = 0
    ) -> object:
        """Helper to add an object by enum name."""
        if not name or not obj_type_name:
            raise ValueError("Object type and name cannot be empty")
        try:
            ot = getattr(self.ObjectType, obj_type_name)
            obj = self.sim.AddObject(ot, x, y, name)
            if hasattr(obj, "GraphicObject") and obj.GraphicObject is not None:
                try:
                    obj.GraphicObject.ShowObjectData = True
                except Exception as exc:
                    logger.debug(
                        f"Exception setting GraphicObject.ShowObjectData: {exc}"
                    )  # AUTO-FIXED

            if obj_type_name == "MaterialStream":
                self.materials[name] = obj
            elif obj_type_name == "EnergyStream":
                self.energy_streams[name] = obj
            else:
                self.operations[name] = obj
            logger.debug(f"Added object '{name}' of type '{obj_type_name}'.")
            return obj
        except Exception as e:
            logger.error(
                f"Failed to add object '{name}' of type '{obj_type_name}': {e}"
            )
            raise ValueError(
                f"Failed to add object '{name}' of type '{obj_type_name}': {e}"
            )

    def connect(
        self, source_obj, target_obj, source_port: int = 0, target_port: int = 0
    ) -> None:
        """Connect two objects via GraphicObject interfaces."""
        if not source_obj or not target_obj:
            raise ValueError("Source and target objects cannot be None")
        try:
            source_graphic = source_obj.GraphicObject
            target_graphic = target_obj.GraphicObject
            self.sim.ConnectObjects(
                source_graphic, target_graphic, source_port, target_port
            )
            logger.debug(
                f"Connected {source_obj} to {target_obj} (ports {source_port}->{target_port})."
            )
        except Exception as e:
            logger.error(f"Failed to connect {source_obj} to {target_obj}: {e}")
            raise RuntimeError(f"Failed to connect {source_obj} to {target_obj}: {e}")

    def calculate(self) -> None:
        """Run the simulation flowsheet."""
        # CalculateFlowsheet2 handles IFlowsheet cleanly in older Pythonnet bindings
        try:
            logger.info("Starting flowsheet calculation.")
            self.interf.CalculateFlowsheet2(self.sim)
            logger.info("Flowsheet calculation finished successfully.")
        except Exception as e:
            # This handles DWSIM solver exceptions nicely
            logger.error(f"DWSIM solver returned an error: {e}")
            raise RuntimeError(f"DWSIM solver returned an error: {e}")

    def save(self, filepath: str) -> None:
        """Save the flowsheet to a .dwxml or .dwsjson file for GUI visualization."""
        try:
            # We use SaveFlowsheet. Ensure absolute path or use os.path.abspath
            abs_path = os.path.abspath(filepath)
            self.interf.SaveFlowsheet(self.sim, abs_path, True)
            logger.info(f"Saved flowsheet successfully to: {abs_path}")
        except Exception as e:
            logger.error(f"Failed to save flowsheet to {filepath}: {e}")
            raise RuntimeError(f"Failed to save flowsheet: {e}")
