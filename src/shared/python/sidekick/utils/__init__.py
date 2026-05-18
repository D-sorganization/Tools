# ruff: noqa: E501
"""Upstream Drift Tools - Utility subpackage.

Submodules:
- paths: Canonical get_repo_root() for repository root discovery
- logging: Logger factory with file+console support
- state_manager: Save/load calculation states and sessions
- unit_constants: NIST-standard conversion factors and physical constants

Overlap with utils (src/python/src/utils):
----------------------------------------------
The following functions have duplicates across packages.  The *canonical*
location is listed first; the other copy is kept for backward compat.

+-------------------+---------------------------------------+-----------------------------+  # noqa: E501
| Function          | Canonical                             | Duplicate (compat)          |  # noqa: E501
+-------------------+---------------------------------------+-----------------------------+  # noqa: E501
| get_repo_root()   | upstream_drift_tools.utils.paths      | utils.path_setup            |  # noqa: E501
| get_logger()      | utils.logging_utils                   | upstream_drift_tools.utils  |  # noqa: E501
|                   |                                       |   .logging (wraps canonical)|  # noqa: E501
| safe_read_json()  | utils.file_utils                      | upstream_drift_tools.utils  |  # noqa: E501
|                   |                                       |   .state_manager (private)  |  # noqa: E501
| safe_write_json() | utils.file_utils                      | upstream_drift_tools.utils  |  # noqa: E501
|                   |                                       |   .state_manager (private)  |  # noqa: E501
+-------------------+---------------------------------------+-----------------------------+  # noqa: E501

New code should always import from the canonical location.
"""
