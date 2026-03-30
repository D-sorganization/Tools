import logging
import sys

from .config import parse_args
from .physics import PhysicsEngine
from .renderer import PendulumRenderer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    """Execution pipeline."""
    phys_cfg, render_cfg = parse_args()

    logging.info("Calculating Lagrangian mechanics & Force Tensors...")
    engine = PhysicsEngine(phys_cfg)
    try:
        physics_data = engine.solve(render_cfg.duration, 1.0 / render_cfg.fps)
    except RuntimeError as str_err:
        logging.error(f"Solver Failure: {str_err}")
        sys.exit(1)

    logging.info("Building animation sequence...")
    renderer = PendulumRenderer(render_cfg, phys_cfg, physics_data)
    renderer.render()


if __name__ == "__main__":
    main()
