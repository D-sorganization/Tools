"""Legacy Movement Optimizer GUI shim.

The vendored GUI package that used to live under ``src/optimizer_gui/python``
was consolidated into ``src/movement_optimizer`` (Tools #3983). This package
now only carries the compatibility registration and launcher shim:
``gui_registration.py`` redirects catalog metadata to the canonical
``movement_optimizer`` PyQt6 app and ``launch_pyqt6.py`` launches it.
"""
