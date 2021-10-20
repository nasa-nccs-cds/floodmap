import csv
from typing import Dict, List, Tuple, Optional
import os, glob, sys
import xarray as xa
import numpy  as np
from floodmap.util.configuration import opSpecs

lake_indices = [ 26, 314, 333, 334, 336, 337 ]
results_dir = opSpecs.get( 'results_dir' )
results_files = dict( nrt='lake_{lake_index}_stats.csv', legacy='lake_{lake_index}_stats_legacy_alt.csv' )
# results_file = opSpecs.get( 'results_file' ).format( lake_index=lake_index )

for (fmtype, results_file) in results_files.items():
    pct_interp_values = []
    interp_area = []
    for lake_index in lake_indices:
        result_file_path = "/".join( [ results_dir, results_file.format( lake_index=lake_index ) ] )
        with open(result_file_path) as file:
            for iLine in range(35):
                line = file.readline()
                if iLine > 0:
                    (date, water_area_km2, percent_interploated) = line.split(",")
                    pct_interp_values.append( float(percent_interploated) )
                    interp_area.append( float(water_area_km2) * float(percent_interploated) / 100 )
    pct_interp_ave = np.array( pct_interp_values ).mean()
    mean_interp_area = np.array( interp_area ).mean()
    print( f" {fmtype}: {pct_interp_ave}, {mean_interp_area}" )

