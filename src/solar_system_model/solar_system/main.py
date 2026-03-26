#!/usr/bin/env python3
"""
Solar System Simulation
=======================

A professional-grade, scientifically accurate solar system model.

This application provides:
- Real-time visualization of planetary orbits
- Accurate positions based on Keplerian mechanics
- Interplanetary trajectory planning
- Multiple camera perspectives
- Educational information overlays

Usage:
    python -m solar_system.main [options]

Options:
    --fullscreen    Start in fullscreen mode
    --width W       Window width (default: 1600)
    --height H      Window height (default: 900)
    --no-vsync      Disable vertical sync
    --start-date    Start date in YYYY-MM-DD format
    --help          Show this help message

Controls:
    SPACE       Pause/Resume simulation
    +/-         Speed up/slow down time
    R           Reverse time flow
    D           Toggle date picker (jump to any date)
    N           Toggle time navigation panel
    E           Toggle historical events panel
    [ / ]       Jump backward/forward 1 day
    0-9         Select celestial body
    F           Focus on selected body
    C           Cycle camera mode
    O           Toggle orbital paths
    L           Toggle labels
    I           Toggle info panel
    G           Toggle grid
    M           Toggle immersion checklist
    H           Toggle help overlay
    T           Plan trajectory to Mars
    .           Cycle through fun facts (when planet selected)
    HOME        Reset view
    ESC         Quit

Mouse:
    Left drag   Orbit camera
    Right drag  Pan camera
    Scroll      Zoom in/out

Educational Features:
    - Manual date navigation to see planetary positions at any time in history
    - Historical space exploration events panel (press E)
    - Educational fun facts about each celestial body
    - Accurate orbital mechanics visualization
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime

from .visualization.renderer import RenderSettings
from .visualization.scene import SolarSystemScene


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Solar System Simulation - A scientifically accurate model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")

    parser.add_argument("--width", type=int, default=1600, help="Window width (default: 1600)")

    parser.add_argument("--height", type=int, default=900, help="Window height (default: 900)")

    parser.add_argument("--no-vsync", action="store_true", help="Disable vertical sync")

    parser.add_argument(
        "--start-date", type=str, default=None, help="Start date in YYYY-MM-DD format"
    )

    parser.add_argument("--no-antialiasing", action="store_true", help="Disable antialiasing")

    parser.add_argument("--no-shaders", action="store_true", help="Disable shaders")

    return parser.parse_args()


def main() -> int:
    """Main entry point for the simulation."""
    args = parse_arguments()

    # Configure render settings
    settings = RenderSettings(
        window_width=args.width,
        window_height=args.height,
        fullscreen=args.fullscreen,
        vsync=not args.no_vsync,
        antialiasing=not args.no_antialiasing,
        use_shaders=not args.no_shaders,
    )

    # Create scene
    scene = SolarSystemScene(settings)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Initialize
    logging.info("\n" + "=" * 70)
    logging.info("  SOLAR SYSTEM SIMULATION")
    logging.info("  Professional-grade astronomical visualization")
    logging.info("=" * 70)
    logging.info("\nInitializing...")

    if not scene.initialize():
        logging.error("ERROR: Failed to initialize. Make sure PyGame and PyOpenGL are installed.")
        logging.error("Install with: pip install pygame PyOpenGL PyOpenGL_accelerate numpy")
        return 1

    # Set start date if specified
    if args.start_date:
        try:
            dt = datetime.strptime(args.start_date, "%Y-%m-%d")
            scene.time_manager.set_datetime(dt)
            logging.info(f"Starting at: {args.start_date}")
        except ValueError:
            logging.warning(f"Warning: Invalid date format '{args.start_date}', using current date")

    logging.info("\n✓ Initialization complete!")
    logging.info("\n" + "=" * 70)
    logging.info("  QUICK START GUIDE")
    logging.info("=" * 70)
    logging.info("\n  The help overlay is now visible in the simulation window.")
    logging.info("  Press 'H' to toggle it on/off.\n")
    logging.info("  KEY CONTROLS:")
    logging.info("  • SCROLL WHEEL  - Zoom in/out to see all planets")
    logging.info("  • LEFT DRAG     - Rotate camera")
    logging.info("  • RIGHT DRAG    - Pan camera")
    logging.info("  • SPACE         - Pause/Resume simulation")
    logging.info("  • +/-           - Speed up/slow down time")
    logging.info("  • D             - Toggle date picker (jump to any date)")
    logging.info("  • N             - Toggle time navigation panel")
    logging.info("  • E             - Toggle historical events")
    logging.info("  • [ / ]         - Jump backward/forward 1 day")
    logging.info("  • { / }         - Jump backward/forward 1 month")
    logging.info("  • T             - Plan trip to Mars from Earth")
    logging.info("  • 0-9           - Select planets (0=Sun, 3=Earth, 4=Mars, etc.)")
    logging.info("  • I             - Toggle info panel")
    logging.info("  • M             - Toggle immersion checklist")
    logging.info("  • H             - Toggle help overlay")
    logging.info("  • ESC           - Quit")
    logging.info("\n  EDUCATIONAL FEATURES:")
    logging.info("  • Navigate to any date in history (1800-2200)")
    logging.info("  • View space exploration events at different times")
    logging.info("  • Learn fun facts about each celestial body")
    logging.info("\n" + "=" * 70)
    logging.info("\nStarting simulation...\n")

    # Run the simulation
    try:
        scene.run()
    except KeyboardInterrupt:
        logging.info("\nSimulation interrupted by user")
    except (subprocess.SubprocessError, OSError) as e:
        logging.error(f"\nError: {e}")
        raise

    logging.info("Simulation ended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
