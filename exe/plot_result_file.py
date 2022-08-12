import rioxarray as rxr
import os
from floodmap.util.plot import create_cmap
import matplotlib.pyplot as plt
import xarray as xa

result_color_map = [
    (0, 'nodata', (0, 0, 0)),  # ,
    (1, 'land', (0, 1, 0)),  # , '
    (2, 'water', (0, 0, 1)),  # ,
    (3, 'int-land', (0, 0.6, 0)),  #
    (4, 'int-water', (0, 0, 0.6)),  #
    (5, 'mask', (1, 1, 0.7))  #
]

lake_index = 462
date = "04302021"
version = "dslay"
rasters = {}
data_dir= "/Users/tpmaxwel/Development/Data/floodmap"
data_layer = f"Lake-{lake_index}"

fname = f"lake_{lake_index}_patched_water_map_{date}_{version}"
input_file = f"{data_dir}/{fname}.nc"
dataset: xa.Dataset = xa.open_dataset( input_file )
raster: xa.DataArray = dataset.data_vars[ data_layer ]
tick_labels, cmap_specs = create_cmap( result_color_map )

figure, ax = plt.subplots()
raster.plot.imshow( ax=ax, **cmap_specs )
plt.show()