#!/usr/bin/env python3
"""Launch the Gasification Equilibrium Calculator.

Usage:
    python3 run_app.py              # Launch interactive GUI
    python3 run_app.py --headless   # Run calculation without GUI (for testing)
    python3 run_app.py --test       # Run all tests
"""

import os
import sys

# Ensure the parent directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    if "--test" in sys.argv:
        import subprocess

        test_dir = os.path.join(os.path.dirname(__file__), "tests")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", test_dir, "-v", "--tb=short"],
            cwd=os.path.dirname(__file__),
        )
        sys.exit(proc.returncode)

    elif "--headless" in sys.argv:
        from python.engine import GasificationEngine

        engine = GasificationEngine()

        print("=" * 60)  # noqa: T201
        print("  Gasification Equilibrium Calculator - Headless Mode")  # noqa: T201
        print("=" * 60)  # noqa: T201

        # Demo calculation
        feed = {"C": 1.0, "H": 1.0, "O": 0.5}
        result = engine.solve(temperature=1000, feed=feed)

        print(f"\nFeed: {feed}")  # noqa: T201
        print(  # noqa: T201
            f"T = {result.temperature:.0f} K, P = {result.pressure/101325:.1f} atm"
        )
        print(f"Converged: {result.converged}")  # noqa: T201
        print(f"H2/CO = {result.h2_co_ratio:.3f}")  # noqa: T201
        print(f"Carbon Conversion = {result.carbon_conversion*100:.1f}%")  # noqa: T201
        print("\nEquilibrium Composition (mol%):")  # noqa: T201
        for sp, frac in sorted(result.composition_dict().items(), key=lambda x: -x[1]):
            if frac > 0.001:
                print(f"  {sp:8s}: {frac*100:6.2f}%")  # noqa: T201

        # Demo sweep
        print(f"\n{'─' * 60}")  # noqa: T201
        print("Temperature Sweep: 500-1500 K")  # noqa: T201
        results = engine.temperature_sweep(500, 1500, n_points=20, feed=feed)
        print(  # noqa: T201
            f"  {sum(1 for r in results if r.converged)}/{len(results)} points converged"
        )
        print("  Done.")  # noqa: T201

    else:
        from python.app import main as app_main

        app_main()


if __name__ == "__main__":
    main()
