import rioxarray as rxr
import os
import matplotlib.pyplot as plt
import xarray as xa
from floodmap.util.plot import plot_array, floodmap_colors, plot_arrays

year = 2021
day_range = [ 119, 121 ]
tile = "h07v05"
rasters = {}
data_dir= "/Users/tpmaxwel/Development/Data/floodmap"

for day in range( *day_range ):
    fname = f"MCDWD_L3_F2_NRT.A{year}{day}.{tile}"
    input_file = f"{data_dir}/{fname}.061.tif"
    if os.path.isfile( input_file ):
        raster: xa.DataArray = rxr.open_rasterio( input_file ).squeeze( drop=True )
        rasters[day] = raster # .where( raster < 10, 4 )

figure, ax = plt.subplots()
plot_arrays( ax, rasters, title=f"Floodmap: tile={tile}", colors=floodmap_colors )
plt.show()
