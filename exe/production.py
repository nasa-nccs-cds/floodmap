from multiprocessing import freeze_support
from floodmap_legacy.surfaceMapping.processing import LakeMaskProcessor

if __name__ == '__main__':
    freeze_support()
    lakeMaskProcessor = LakeMaskProcessor()
    lakeMaskProcessor.process_lakes( format="nc", skip_existing=False )



