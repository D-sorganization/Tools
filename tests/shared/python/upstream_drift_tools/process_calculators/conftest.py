"""conftest.py for process_calculators tests.

Sets headless matplotlib backend and limits thread counts so that scipy/numpy
don't stall on Windows during import. Also disables xdist parallelism for this
subdirectory since the Windows matplotlib/scipy backend detection is not safe
to run in parallel processes.
"""

import os

# Must be set before any numpy/scipy/matplotlib import
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
