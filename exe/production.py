from multiprocessing import freeze_support
from floodmap.surfaceMapping.processing import LakeMaskProcessor

if __name__ == '__main__':
    freeze_support()
    skip_existing = False
    save_diagnostics = True
    lakeMaskProcessor = LakeMaskProcessor()
    lakeMaskProcessor.process_lakes( format="tif", save_diagnostics=save_diagnostics, skip_existing=skip_existing )

