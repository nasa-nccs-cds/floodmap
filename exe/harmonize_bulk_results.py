from glob import glob
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import csv
import numpy as np
import xarray as xa
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

def open_dset( fmversion ) -> xa.Dataset:
    result_name = f"floodmap_results_{fmversion}"
    result_file = f"{results_dir}/{result_name}.nc"
    return xa.open_dataset( result_file )


dset_legacy = open_dset( "legacy" )
dset_nrt = open_dset( "nrt" )

# dset_harmonized = get_dset( "nrt_harmonized", "w" )

for vname in ['water_area', 'pct_interp']:
    nrt_var: xa.DataArray = dset_nrt.data_vars[vname]
    legacy_var: xa.DataArray = dset_legacy.data_vars[vname]
    nrt_lake_filter: xa.DataArray = nrt_var.lake.isin( legacy_var.lake.values )
    legacy_lake_filter: xa.DataArray = legacy_var.lake.isin(nrt_var.lake.values)
    filtered_nrt_var = nrt_var.where( nrt_lake_filter )
    legacy_nrt_var = legacy_var.where( legacy_lake_filter )
    print(".")


