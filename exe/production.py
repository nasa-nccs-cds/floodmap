from multiprocessing import freeze_support
from floodmap.surfaceMapping.processing import LakeMaskProcessor

if __name__ == '__main__':
    freeze_support()
    lakeMaskProcessor = LakeMaskProcessor()
    lakeMaskProcessor.process_lakes()

