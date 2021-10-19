import os, logging
from .configuration import opSpecs

def getLogger( master: bool, level = logging.DEBUG ):
    pid = os.getpid()
    lname = "master" if master else f"worker-{pid}"
    logger = logging.getLogger( lname )
    logger.setLevel(level)
    if not len(logger.handlers):
        log_dir = opSpecs.get("log_dir", "/tmp")
        log_file = f"{log_dir}/floodmap.{lname}.log"
        print(f" Creating log file: {log_file}")
        handler = logging.FileHandler(log_file,mode="w")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if master:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
    return logger


def getConsoleLogger( level = logging.DEBUG ):
    logger = logging.getLogger( f"console" )
    logger.setLevel(level)
    if not len(logger.handlers):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger