from pathlib import Path

try:
    from utils.path_helpers import ensure_utils_in_path
except ImportError:

    def ensure_utils_in_path():
        pass


# Add src to sys.path
src_path = Path(__file__).resolve().parent.parent / "src"
ensure_utils_in_path()
