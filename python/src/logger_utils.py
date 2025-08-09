import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Seeds set: %%d", seed)
