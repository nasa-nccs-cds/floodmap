import rioxarray as rxr
import os
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import matplotlib.pyplot as plt
import xarray as xa
from floodmap.util.plot import plot_array, floodmap_colors, plot_arrays, result_colors
from floodmap.util.configuration import opSpecs
from floodmap.surfaceMapping.processing import LakeMaskProcessor
from floodmap.util.configuration import opSpecs
from multiprocessing import freeze_support
results_dir = opSpecs.get( 'results_dir' )
water_maps_specs = opSpecs.get( 'water_maps' )
bin_size = water_maps_specs.get( 'bin_size', 8 )

#lake_index = 4                  #  [4, 5, 9, 11, 12, 14, 19, 21, 22, 26, 28, 37, 42, 43, 44, 51, 53, 60, 66, 67, 69, 73, 74, 76, 79, 81, 82, 85, 87, 88, 91, 93, 97, 99]
#opSpecs.get('lake_masks')['lake_index'] = lake_index
# run_cfg = {}

day_range = [ 6, 275 ]

if __name__ == '__main__':
    freeze_support()

    for day in range( day_range[0]+bin_size, day_range[1], bin_size ):
        print( f"\nProcessing day = {day}")
        lakeMaskProcessor = LakeMaskProcessor()
        lakeMaskProcessor.process_lakes( day=day )  # parallel=False )

