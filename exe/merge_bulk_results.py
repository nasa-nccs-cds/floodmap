from glob import glob
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import csv
import numpy as np
from floodmap.util.configuration import opSpecs
results_dir = opSpecs.get('results_dir')
from datetime import datetime
import netCDF4 as nc

calendar = 'standard'
units = 'days since 1970-01-01 00:00'

def get_timestamp( tstr: str, fmversion: str ) -> datetime:
    if fmversion == "nrt": (m, d, y) = tstr.split("-")
    elif fmversion == "legacy": (y, m, d) = tstr.split("-")
    else: raise Exception( f"Unrecognized fmversion: {fmversion}")
    return datetime(int(y), int(m), int(d))

for fmversion in [ "legacy", "nrt" ]:
    print(f"\n **** Processing fmversion = {fmversion} ****\n ")
    stats_file_glob = "lake_*_stats.txt" if fmversion == "nrt" else "lake_*_stats_legacy.txt"
    result_name = f"floodmap_results_{fmversion}"
    result_file = f"{results_dir}/{result_name}.nc"
    lake_data = {}
    time_vals = set()
    dset = nc.Dataset(result_file, 'w', format='NETCDF4')

    for filepath in glob( f"{results_dir}/{stats_file_glob}"):
        with open(filepath, newline='') as csvfile:
            print( f"Processing file {filepath}")
            csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            lake_index = int( filepath.split('_')[1] )
            lake_spec = lake_data.setdefault( lake_index, {} )
            for row in csvreader:
                if "-" in row[0]:
                    ts = get_timestamp( row[0], fmversion )
                    time_vals.add(ts)
                    water_area = float( row[1] )
                    pct_interp = float( row[2] )
                    lake_spec[ts] = ( water_area, pct_interp )

    timeindex = sorted(time_vals)
    lakeindex = sorted(lake_data.keys())
    time = dset.createDimension( 'time', len(timeindex) )
    lake = dset.createDimension( 'lake', len( lakeindex ) )

    times = dset.createVariable( 'time', 'f4', ('time',) )
    times[:] = nc.date2num( timeindex, units=units, calendar=calendar )
    lakes = dset.createVariable('lake', 'i4', ('lake',))
    lakes[:] = np.array( lakeindex )

    water_area_var = dset.createVariable( 'water_area', 'f4', ('time', 'lake'), fill_value=float('nan') )
    pct_interp_var = dset.createVariable( 'pct_interp', 'f4', ('time', 'lake'), fill_value=float('nan') )

    for lindex, lake_id in enumerate(lakeindex):
        lake_spec = lake_data[lake_id]
        for tindex, ts in enumerate(timeindex):
            if ts in lake_spec.keys():
                (water_area, pct_interp) = lake_spec[ts]
                water_area_var[tindex, lindex] = water_area
                pct_interp_var[tindex, lindex] = pct_interp

    print( f"Saving floodmap data to {result_file}")
    dset.close()