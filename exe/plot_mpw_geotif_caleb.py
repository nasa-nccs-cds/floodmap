import numpy as np
import os
import rioxarray as rio
import matplotlib.pyplot as plt
from floodmap.util.xgeo import XGeo
import xarray as xa
from floodmap.util.plot import plot_array, floodmap_colors, plot_arrays

year = 2021
day_range = [ 119, 121 ]
type = "caleb"
tile = "h09v05"
rasters = {}
data_dir= "/Users/tpmaxwel/Development/Data/floodmap"
xbnds = [ -10007438,-8895719 ]
ybnds = [ 4183794, 4083947 ]

for day in range( *day_range ):
    fname = f"{year}-{day}-{tile}-MOD-randomforest"
    input_file = f"{data_dir}/{fname}.tif"
    if os.path.isfile( input_file ):
        raster: xa.DataArray = rio.open_rasterio( input_file ).squeeze( drop=True )
        print(f" ROI: x=[{raster.x.values[0]},{raster.x.values[-1]}]  y=[{raster.y.values[0]},{raster.y.values[-1]}]")
        raster = raster.sel( x=slice(*xbnds), y=slice(*ybnds) )
#        nwater = np.count_nonzero( raster == 1 )
#        print( f" #water={nwater} size={raster.size} %water={(nwater/raster.size)%100.0}")
        rasters[day] = raster

figure, ax = plt.subplots()
plot_arrays( ax, rasters, title=f"Floodmap: tile={tile}", colors=floodmap_colors )
plt.show()
