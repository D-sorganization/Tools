"""Support ``python -m agent_context`` without optional dependencies."""

from .cli import main

raise SystemExit(main())
