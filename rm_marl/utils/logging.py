import logging
import os

def getLogger(name):
    logger = logging.getLogger(name.rsplit('.', 1)[-1])
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    return logger