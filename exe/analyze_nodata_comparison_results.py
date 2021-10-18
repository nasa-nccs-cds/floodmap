from typing import Dict, List, Tuple, Optional
import os, glob, sys
import xarray as xa
import numpy  as np
from floodmap.util.configuration import opSpecs

data_loc = opSpecs.get('results_dir')
stats_filepath = f"{data_loc}/nodata_stats_comparison_lake.csv"

pct_diff_vals = []
with open( stats_filepath, "r" ) as stats_file:
    lines = stats_file.readlines()
    for line in lines:
        cols = line.split(",")
        pct_diff = float( cols[3] )
        pct_diff_vals.append( pct_diff )


ave_pct_diff = np.array( pct_diff_vals ).mean()
print( f" ave_pct_diff = {ave_pct_diff}" )
