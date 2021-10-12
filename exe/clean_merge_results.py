from glob import glob
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import csv, os
import numpy as np
from floodmap.util.configuration import opSpecs
results_dir = opSpecs.get('results_dir')
from datetime import datetime
import netCDF4 as nc

calendar = 'standard'
units = 'days since 1970-01-01 00:00'
nts = 31

def get_timestamp( tstr: str ) -> int:
    (y, m, d) = tstr.split("-")
    return int( nc.date2num( datetime(int(y), int(m), int(d)), units=units, calendar=calendar ) )

stats_file_glob = "lake_*_stats_legacy.txt.invalid"
lake_data = {}
timeindex = []
file_list = glob( f"{results_dir}/{stats_file_glob}")
invalid_list = []

for filepath in file_list:
    with open(filepath, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        lake_index = int( filepath.split('_')[1] )
        print(f"Processing file {filepath} for lake {lake_index}")
        for iR, row in enumerate(csvreader):
            if (iR > 0) and (iR <= nts):
                ts: int = get_timestamp(row[0])
                print( f"Lake-{lake_index} -> iR = {iR}, ts = {ts} ({row[0]})")
                if len(lake_data) == 1: timeindex.append(ts)
                elif ts != timeindex[iR-1]:
                    print( f"Mismatched time value[{iR}] for lake {lake_index} ({ts} vs {timeindex[iR-1]})" )
                    invalid_list.append( filepath )
                    break

for filepath in invalid_list:
    print( f"Invalid file {filepath}")
#    os.rename( filepath, filepath + ".invalid" )



