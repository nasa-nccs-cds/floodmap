from floodmap.util.configuration import opSpecs
from floodmap.util.anim import ArrayAnimation
from typing import Dict, List, Tuple
import os, glob, getpass
import xarray as xr, numpy as np
results_dir = opSpecs.get( 'results_dir' )

cats = [    ( 0, 'nodata', (0, 0, 0)),
            ( 1, 'land',   (0, 1, 0)),
            ( 2, 'water',  (0, 0, 1)),
            ( 3, 'interp land',   (0, 0.5, 0)),
            ( 4, 'interp water',  (0, 0, 0.5)),
            ( 5, 'mask', (0.25, 0.25, 0.25) ),
            ( 6, 'mask', (0.25, 0.25, 0.25) ),
            ( 7, 'mask', (0.25, 0.25, 0.25) ) ]
cmap = { cat[0]: cat[2] for cat in cats }

Lake_index = 462

file_glob = f"{results_dir}/{getpass.getuser( )}/lake_{Lake_index}_patched_water_map_*-geog.nc"
fpaths = glob.glob( file_glob )
data_arrays: List[xr.DataArray] = []
times: List[np.datetime64] = []
fps = 0.5

for fpath in fpaths:
    dset: xr.Dataset = xr.open_dataset( fpath )
    time: np.datetime64 = dset.time.values
    dvar: xr.DataArray = dset[f"Lake-{Lake_index}"]
    data_arrays.append( dvar )

# result: xr.DataArray = xr.concat( data_arrays, dim=np.array(times) )

roi = data_arrays[0].xgeo.bounds()
animator = ArrayAnimation( roi=roi, fps=fps )
anim = animator.create_animation( data_arrays, color_map=cmap )

