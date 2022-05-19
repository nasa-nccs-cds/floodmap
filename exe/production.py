from multiprocessing import freeze_support
from floodmap.surfaceMapping.processing import LakeMaskProcessor
import time

if __name__ == '__main__':
    freeze_support()
    t0 = time.time()

    lakeMaskProcessor = LakeMaskProcessor()
    lakeMaskProcessor.process_lakes()

    print( f"Completed processing in {(time.time()-t0)/60} min." )