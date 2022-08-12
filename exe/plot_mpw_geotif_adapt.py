import rioxarray as rxr
import os
import matplotlib.pyplot as plt
import xarray as xa
from floodmap.util.plot import plot_array, floodmap_colors, plot_arrays

day_range = [ 120, 121 ]
year = 2021
tile = "h07v05"
rasters = {}
data_dir= "/explore/nobackup/projects/ilab/projects/Birkett/MOD44W/data"

if data_dir is None:
   from floodmap.util.configuration import opSpecs
   results_dir = opSpecs.get('results_dir')
   data_dir = f"{results_dir}/{tile}/allData/61/MCDWD_L3_F2_NRT/Recent"

for day in range( *day_range ):
    fname = f"MCDWD_L3_F2_NRT.A{year}{day:03}.{tile}"
    input_file = f"{data_dir}/{tile}/allData/61/MCDWD_L3_F2_NRT/Recent/{fname}.061.tif"
    if os.path.isfile( input_file ):
        raster: xa.DataArray = rxr.open_rasterio( input_file ).squeeze( drop=True )
        print( f"*RASTER: day={day}, attrs={raster.attrs}")
        print(f"  -> spatial_ref: '{raster.spatial_ref.crs_wkt}', attrs={raster.spatial_ref.attrs}")
#        print( dir(raster.spatial_ref) )
        rasters[day] = raster  # .where( raster < 10, 4 )

#figure, ax = plt.subplots()
#plot_arrays( ax, rasters, title=f"Floodmap: tile={tile}", colors=floodmap_colors )
#plt.show()
