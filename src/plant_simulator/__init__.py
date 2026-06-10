"""Neural Network Plant Simulator module (EXPERIMENTAL).

This package is experimental and not wired into any production launcher/tool
manifest. It trains an LSTM on logged SCADA ``TagLog`` data to predict plant
behaviour. Historically its dataset loader fabricated random data, so models
trained on noise (issue #3295); that silent path has been removed. The loader
now requires a populated SCADA database and raises ``ValueError`` on
insufficient data unless synthetic data is *explicitly* requested
(``SCADADataset(..., allow_synthetic=True)``).
"""

import warnings

warnings.warn(
    "plant_simulator is experimental: it is not a production tool and its "
    "predictions should not be relied upon. See issue #3295.",
    stacklevel=2,
)
