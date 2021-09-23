from multiprocessing import freeze_support
from floodmap.surfaceMapping.processing import LakeMaskProcessor

if __name__ == '__main__':
    freeze_support()
    skip_existing = False
    format="nc"
    lakeMaskProcessor = LakeMaskProcessor()
    lakeMaskProcessor.process_lakes( skip_existing=skip_existing, format=format )

