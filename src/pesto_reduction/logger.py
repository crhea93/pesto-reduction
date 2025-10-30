import os
import logging
from datetime import datetime

def make_time_named_logger(level=logging.INFO,
                           fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
    """
    Create a logger whose *name* and *log-file* are the UTC timestamp
    of its creation (YYYYMMDD_HHMMSS).

    Parameters
    ----------
    level : int | str
        Logging level (e.g. logging.INFO, "DEBUG", etc.).
    fmt : str
        Format string for log records.

    Returns
    -------
    logging.Logger
        Fully configured, ready-to-use logger.
    """
    # 1️⃣ Timestamp string to use everywhere
    stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    # 2️⃣ Build the logger
    logger = logging.getLogger(stamp)
    logger.setLevel(level)
    logger.propagate = False        # keep messages from doubling in root logger

    # 3️⃣ Common formatter
    formatter = logging.Formatter(fmt)

    # 4️⃣ File handler -> `<timestamp>.log`
    os.makedirs('./logs', exist_ok=True)
    fh = logging.FileHandler(f"./logs/{stamp}.log")
    fh.setLevel(level)
    fh.setFormatter(formatter)

    # 5️⃣ Console handler (stream)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    # 6️⃣ Attach handlers (avoid duplicates on repeated calls)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
