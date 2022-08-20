import rioxarray as rxr
import os, numpy as np
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

data_layer = "mpw-2272022"
rioplot = True
test_file = "/Volumes/Shared/Data/floodmap/test/Tile-h21v10-test1.nc"

dataset: xa.Dataset =  xa.open_dataset( test_file )  # rxr.open_rasterio(input_file)
raster: xa.DataArray = dataset.data_vars[ data_layer ]
raster = raster.where( raster < 5, 5 )
tick_labels, cmap_specs = create_cmap( result_color_map )

figure, ax = plt.subplots()
if rioplot:
    raster.plot.imshow( ax=ax, **cmap_specs )
else:
    raster_data: np.ndarray = raster.values
    img = ax.imshow( raster_data, cmap='jet', vmin=0.0, vmax=5.0 )
    ax.figure.colorbar(img, ax=ax )
plt.show()

def class_counts( raster ):
    print(f" Class counts: shape={raster.shape} size={raster.size}" )
    for iclass in range(0, 8):
        print(f" ** [{iclass}]: {np.count_nonzero(raster.isin([iclass]))}")

#> cd /Users/tpmaxwel/Development/Data/floodmap
#> scp tpmaxwel@adaptlogin.nccs.nasa.gov:/explore/nobackup/projects/ilab/projects/Birkett/MOD44W/results-mary-dslay/tpmaxwel/lake_462_patched_water_map_04292021-geog.nc ./lake_462_patched_water_map_04292021_dslay-new.nc