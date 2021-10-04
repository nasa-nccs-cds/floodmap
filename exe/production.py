from multiprocessing import freeze_support
from floodmap.surfaceMapping.processing import LakeMaskProcessor

if __name__ == '__main__':
    freeze_support()
    reproject_inputs = False
    lakeMaskProcessor = LakeMaskProcessor()
    lakeMaskProcessor.process_lakes( reproject_inputs, format="tif", skip_existing=False )

