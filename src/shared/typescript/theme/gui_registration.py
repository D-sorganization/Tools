"""GUI registration for shared typescript theme package.

Declares web: False and catalog_visible: False so this shared library package.json
is recognized as not an unreachable launcher tile in the tool registry.
"""

GUI_INFO = {
    "catalog_visible": False,
    "web": False,
}
