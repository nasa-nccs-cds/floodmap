import os, logging
from .configuration import opSpecs

def getLogFile( master: bool ):
    pid = os.getpid()
    lname = "master" if master else f"worker-{pid}"
    log_dir = opSpecs.get("log_dir", "/tmp")
    return (lname, f"{log_dir}/floodmap.{lname}.log")

def getLogger( master: bool, level = logging.DEBUG ):
    (lname,log_file) = getLogFile( master )
    logger = logging.getLogger( lname )
    logger.setLevel(level)
    if not len(logger.handlers):
        handler = logging.FileHandler( log_file, mode='w' )
        print( f"Creating log file: {log_file}")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
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