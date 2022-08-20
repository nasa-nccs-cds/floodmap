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

year = 2022
day_range = [ 224, 231 ]
tile =  "h21v10"
rasters = {}
coords = None
data_dir= f"/Volumes/Shared/Data/floodmap/Results/{tile}/allData/61/MCDWD_L3_F2_NRT/Recent/"
test_file = f"/Volumes/Shared/Data/floodmap/test/Tile-{tile}-test1.nc"
ybnds = [ -9.47, -14.44 ]
xbnds = [ 33.68, 35.39 ]

for day in range( *day_range ):
    fname = f"MCDWD_L3_F2_NRT.A{year}{day:03d}.{tile}.061"
    input_file = f"{data_dir}/{fname}.tif"
    if os.path.isfile( input_file ):
        raster: xa.DataArray = rio.open_rasterio( input_file ).squeeze( drop=True )
        print(f" ROI: x=[{raster.x.values[0]},{raster.x.values[-1]}]  y=[{raster.y.values[0]},{raster.y.values[-1]}]")
        raster = raster.sel( x=slice(*xbnds), y=slice(*ybnds) )
        nwater = np.count_nonzero( raster == 1 )
        print( f" DAY-{day}: #water={nwater}, area={nwater/16} km2")
        rasters[day] = raster
        if coords is None: coords = dict( x = raster.x, y = raster.y )

data_vars = { f"mpw-{day}{year}": v for day,v in rasters.items() }
test_dataset = xa.Dataset( data_vars, coords )
test_dataset.to_netcdf( test_file )
print( f"Writing test file: {test_file}" )

figure, ax = plt.subplots()
plot_arrays( ax, rasters, title=f"Floodmap: tile={tile}", colors=floodmap_colors )
plt.show()
