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
nts = 31

def get_timestamp( tstr: str, fmversion: str ) -> int:
    if fmversion == "nrt": (m, d, y) = tstr.split("-")
    elif fmversion == "legacy": (y, m, d) = tstr.split("-")
    else: raise Exception( f"Unrecognized fmversion: {fmversion}")
    return int( nc.date2num( datetime(int(y), int(m), int(d)), units=units, calendar=calendar ) )

def get_rows( cvsreader, nrows  ):
    istart = -1
    rows = []
    for iR, row in enumerate(cvsreader):
        rows.append( row )
        if row[0].strip() == "date":  istart = iR + 1
    return [ rows[i] for i in range( istart, istart+nrows) ]

fmversion = "legacy" # "nrt"
print(f"\n **** Processing fmversion = {fmversion} ****\n ")
stats_file_glob = "lake_*_stats.csv" if fmversion == "nrt" else "lake_*_stats_legacy_alt.csv"
result_name = f"floodmap_results_{fmversion}_alt"
result_file = f"{results_dir}/{result_name}.nc"
lake_data = {}
timeindex = []
dset = nc.Dataset(result_file, 'w', format='NETCDF4')
file_list = glob( f"{results_dir}/{stats_file_glob}")

for filepath in file_list:
    with open(filepath, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        lake_index = int( filepath.split('_')[1] )
        lake_spec = ([],[])
        print(f"Processing file {filepath} for lake {lake_index}")
        rows = get_rows( csvreader, nts )
        for iR, row in enumerate(rows):
            print( f"Processing date[{iR}]: {row[0]}")
            ts: int = get_timestamp(row[0], fmversion)
            if len(lake_data) == 0:
                timeindex.append(ts)
            elif (ts != timeindex[iR-1]):
                print( f"Mismatched time value[{iR}] for lake {lake_index} ({ts} vs {timeindex[iR-1]})" )
                break
            water_area = float( row[1] )
            pct_interp = float( row[2] )
            lake_spec[0].append( water_area )
            lake_spec[1].append( pct_interp )

        if len( lake_spec[0] ) == len( timeindex ):
            lake_data[lake_index] = lake_spec
        else:
            print( f"Skipping lake {lake_index} due to faulty data")
            if len(lake_data) == 0: timeindex = []


lakeindex = sorted(lake_data.keys())
print( f"Created dimension time, len = {len( timeindex )} ")
time = dset.createDimension( 'time', len( timeindex ) )
print( f"Created lake time, len = {len( lakeindex )} ")
lake = dset.createDimension( 'lake', len( lakeindex ) )

times = dset.createVariable( 'time', 'f4', ('time',) )
times[:] = np.array( timeindex )
lakes = dset.createVariable('lake', 'i4', ('lake',))
lakes[:] = np.array( lakeindex )

water_area_var = dset.createVariable( 'water_area', 'f4', ('time', 'lake'), fill_value=float('nan') )
pct_interp_var = dset.createVariable( 'pct_interp', 'f4', ('time', 'lake'), fill_value=float('nan') )

for li, lake_id in enumerate(lakeindex):
    (water_area, pct_interp) = lake_data[lake_id]
    water_area_var[:,li] = np.array( water_area )
    pct_interp_var[:,li] = np.array( pct_interp )

print( f"Saving floodmap data to {result_file}")
dset.close()