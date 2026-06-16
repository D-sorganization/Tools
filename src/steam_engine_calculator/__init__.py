"""Compatibility package for the steam engine calculator.

The implementation package lives under ``python/steam_engine_calculator``.
Extending ``__path__`` keeps imports such as ``steam_engine_calculator.ui``
working when the repository ``src`` directory is on ``sys.path``.
"""

from pathlib import Path

_implementation_package = Path(__file__).resolve().parent / "python" / __name__
if _implementation_package.is_dir():
    __path__.append(str(_implementation_package))
