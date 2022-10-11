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
# /Volumes/Shared/Data/floodmap/Results/tpmaxwel/lake_462_patched_water_map_08292022.tif
lake_index = 316
date = "1007"
use_utm = True
use_input = False
year = 2022
rasters = {}
data_dir= "/Volumes/Shared/Data/floodmap/Results/tpmaxwel"
data_layer = 'mpw' if use_input else f"Lake-{lake_index}"
ext = ".tif" if use_utm else "-geog.nc"
if use_input:  fname = f"lake_{lake_index}_nrt_input_data_{date}{year}.nc"
else:          fname = f"lake_{lake_index}_patched_water_map_{date}{year}{ext}"
input_file = f"{data_dir}/{fname}"
rioplot = True

print( f"Plotting result: {input_file}")
if use_utm:
    raster: xa.DataArray = rxr.open_rasterio( input_file ).squeeze( drop=True )
else:
    dataset: xa.Dataset =  xa.open_dataset( input_file )  # rxr.open_rasterio(input_file)
    raster: xa.DataArray = dataset.data_vars[ data_layer ]
    if use_input: raster = raster[6,:,:].squeeze( drop=True )

tick_labels, cmap_specs = create_cmap( result_color_map )
nwater = np.count_nonzero( raster.isin([2,4]) )
dv = 16 if use_utm else 20
print(f" DAY-{date}: #water={nwater}, area={nwater/16} km2")

for iclass in  range(0,8):
    print( f" Class count[{iclass}]: {np.count_nonzero( raster.isin([iclass]) )}")

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