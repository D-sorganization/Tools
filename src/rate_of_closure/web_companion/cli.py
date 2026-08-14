"""Console entry point for the source production web companion."""

from __future__ import annotations

from .runtime import start_companion


def main() -> int:
    """Run the exact packaged companion until explicit process shutdown."""
    runtime = start_companion()
    try:
        runtime.wait()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
