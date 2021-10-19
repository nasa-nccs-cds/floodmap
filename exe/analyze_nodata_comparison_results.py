from typing import Dict, List, Tuple, Optional
import os, glob, sys
import xarray as xa
import numpy  as np
from floodmap.util.configuration import opSpecs

def pct_diff( x0: float, x1: float) -> float:
    sgn = 1 if (x1>x0) else -1
    return sgn * (abs(x1 - x0) * 100) / min(x0, x1)

data_loc = opSpecs.get('results_dir')
stats_filepaths = [ f"{data_loc}/nodata_stats_comparison_lake_masked.csv", f"{data_loc}/nodata_stats_comparison_lake_masked_1.csv" ]
# 520, 33.36, 52.17, 34.77, 58.31, 29.9457, 30.9996, 50.5665, 51.2417
legacy_nodata,legacy_size, nrt_nodata, nrt_size = [], [], [], []
for stats_filepath in stats_filepaths:
    with open( stats_filepath, "r" ) as stats_file:
        lines = stats_file.readlines()
        for line in lines:
            cols = line.split(",")
            lake_index = int(cols[0])
            tile_legacy_nodata = float(cols[1]); legacy_nodata.append( tile_legacy_nodata )
            tile_legacy_size = float(cols[2]); legacy_size.append( tile_legacy_size )
            tile_nrt_nodata = float(cols[3]); nrt_nodata.append( tile_nrt_nodata )
            tile_nrt_size = float(cols[4]); nrt_size.append( tile_nrt_size )

legacy_pct_nodata = ( np.ndarray(legacy_nodata).sum() / np.ndarray(legacy_size).sum() ) * 100
nrt_pct_nodata = ( np.ndarray(nrt_nodata).sum() / np.ndarray(nrt_size).sum() ) * 100
print( f" legacy_pct_nodata = {legacy_pct_nodata}" )
print( f" nrt_pct_nodata = {nrt_pct_nodata}" )
print( f" pct_diff = {pct_diff(legacy_pct_nodata,nrt_pct_nodata)}" )
