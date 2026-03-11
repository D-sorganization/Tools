"""Launch the Programmatic PID generator CLI.

This module serves as the entry point for the Tools launcher
to invoke the P&ID generator command-line interface.
"""

from __future__ import annotations


def main() -> None:
    """Launch the PID generator CLI."""
    from programmatic_pid.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
