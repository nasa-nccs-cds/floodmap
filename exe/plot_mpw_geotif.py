import rioxarray as rxr
import os
import xarray as xa
from floodmap.util.plot import plot_array, floodmap_colors, plot_arrays
from floodmap.util.configuration import opSpecs
results_dir = opSpecs.get('results_dir')

day_range = [ 100, 260 ]
tile = "h20v09"
rasters = {}

for day in range( *day_range ):
    fname = f"MCDWD_L3_F2_NRT.A2021{day:03}.{tile}"
    input_file = f"{results_dir}/{tile}/allData/61/MCDWD_L3_F2_NRT/Recent/{fname}.061.tif"
    if os.path.isfile( input_file ):
        raster: xa.DataArray = rxr.open_rasterio( input_file ).squeeze( drop=True )
        rasters[day] = raster.where( raster < 10, 4 )

plot_arrays( f"Floodmap: tile={tile}", rasters, floodmap_colors )


# MCDWD_L3_F2_NRT.A2021240.h20v09.061.tif