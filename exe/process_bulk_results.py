from glob import glob
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import xarray as xa
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

fmversion = "legacy"
result_name = f"floodmap_results_{fmversion}"
result_file = f"{results_dir}/{result_name}.nc"
dset = xa.open_dataset(result_file)
water_area = dset.data_vars['water_area']

for ilake in range(4):
    lake_index = dset.lake[ilake]
    print(f"Lake {lake_index}:")
    print( water_area.data[:,ilake] )

