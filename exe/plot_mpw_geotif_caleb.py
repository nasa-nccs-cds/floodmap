import numpy as np
import os
import rioxarray as rio
from floodmap.util.xrio import XRio
import matplotlib.pyplot as plt
from floodmap.util.xgeo import XGeo
import xarray as xa
from floodmap.util.plot import plot_array, floodmap_colors, plot_arrays

year = 2021
day = 119
type = "caleb"
tile = "h09v05"
rasters = {}
data_dir= "/Users/tpmaxwel/Development/Data/floodmap"
xbnds = [ -111.6, -110.4 ]
ybnds = [ 36.9, 37.8  ]
fnames = [ f"{year}-{day}-{tile}-MOD-randomforest" ]
file_paths = [ f"{data_dir}/{fname}.tif" for fname in fnames ]
mask_value = 5
roi_bounds = xbnds + ybnds
tile_raster: xa.DataArray = XRio.load(file_paths, mask=roi_bounds, band=0, mask_value=mask_value )
nodata = tile_raster.attrs['_FillValue']
tile_raster =  tile_raster.where( tile_raster >= 0, mask_value ).squeeze( drop=True )
nwater = np.count_nonzero(tile_raster == 1)
print(f" #water={nwater} size={tile_raster.size} %water={(nwater / tile_raster.size) % 100.0}")

# for day in range( *day_range ):
#     fname = f"{year}-{day}-{tile}-MOD-randomforest"
#     input_file = f"{data_dir}/{fname}.tif"
#     if os.path.isfile( input_file ):
#         raster: xa.DataArray = rio.open_rasterio( input_file ).squeeze( drop=True )
#         print(f" ROI: x=[{raster.x.values[0]},{raster.x.values[-1]}]  y=[{raster.y.values[0]},{raster.y.values[-1]}]")
#         raster = raster.sel( x=slice(*xbnds), y=slice(*ybnds) )
# #        nwater = np.count_nonzero( raster == 1 )
# #        print( f" #water={nwater} size={raster.size} %water={(nwater/raster.size)%100.0}")
#         rasters[day] = raster

print( f"Data range: [ {tile_raster.values.min()}, {tile_raster.values.max()} ]" )
figure, ax = plt.subplots()
plot_arrays( ax, { day: tile_raster }, title=f"Floodmap: tile={tile}", colors=floodmap_colors )
plt.show()
