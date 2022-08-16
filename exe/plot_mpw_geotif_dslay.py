import numpy as np
import os
import rioxarray as rio
import matplotlib.pyplot as plt
from floodmap.util.xgeo import XGeo
import xarray as xa
from floodmap.util.plot import plot_array, plot_arrays

floodmap_colors = [ ( 0, 'land',         (0, 1, 0)),
                    ( 1, 'perm water',   (0, 0, 1)),
                    ( 2, 'flood water',  (0, 0, 0.5)),
                    ( 3, 'flood water',  (0, 0, 0.7)),
                    ( 4, 'nodata',       (0, 0, 0)) ]

year = 2021
day_range = [ 1, 100 ]
tile =  "h06v05"
rasters = {}
data_dir= "/Users/tpmaxwel/Development/Data/floodmap/h06v05"
xbnds = [ -111.6, -110.4 ]
ybnds = [ 37.8, 36.9 ]

for day in range( *day_range ):
    fname = f"MCDWD_L3_F2_NRT.A{year}{day:03d}.{tile}.061"
    input_file = f"{data_dir}/{fname}.tif"
    if os.path.isfile( input_file ):
        raster: xa.DataArray = rio.open_rasterio( input_file ).squeeze( drop=True )
        print(f" ROI: x=[{raster.x.values[0]},{raster.x.values[-1]}]  y=[{raster.y.values[0]},{raster.y.values[-1]}]")
        raster = raster.sel( x=slice(*xbnds), y=slice(*ybnds) )
        nwater = np.count_nonzero( raster == 1 )
        print( f" #water={nwater} size={raster.size} %water={(nwater/raster.size)%100.0}")
        rasters[day] = raster

figure, ax = plt.subplots()
plot_arrays( ax, rasters, title=f"Floodmap: tile={tile}", colors=floodmap_colors )
plt.show()
