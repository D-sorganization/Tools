"""Utils package providing common utilities across the repository.

This package contains:
- logging_utils: Logging configuration and utilities
- error_handling: Error handling decorators and utilities
- test_utils: Testing utilities, fixtures, and assertions
- debug_utils: Debugging, profiling, and diagnostic tools
- integration_test_helpers: Integration test base classes and helpers
- path_setup: Repository root detection and Python path configuration
- file_utils: Safe file read/write operations
- validation: Input validation helpers
- csv_utils: CSV reading/writing utilities
- config_loader: Configuration loading from YAML/JSON
- env_utils: Environment variable helpers
- os_utils: OS-level helpers
- subprocess_utils: Safe subprocess execution

Import directly from submodules for explicit, lightweight imports::

    from utils.logging_utils import get_logger, setup_logging
    from utils.error_handling import safe_execute, exit_on_error
    from utils.path_setup import get_repo_root, setup_python_path
"""

# Intentionally empty: submodules should be imported directly
# to avoid loading unnecessary dependencies at package init time.
#
# Before (eager, loaded everything on `import utils`):
#     from utils.debug_utils import MemoryStats, ...
#     from utils.error_handling import exit_on_error, ...
#     from utils.logging_utils import get_logger, ...
#     from utils.test_utils import BaseTestCase, ...
#
# After (lazy, import what you need):
#     from utils.logging_utils import get_logger
#     from utils.file_utils import safe_read_text
