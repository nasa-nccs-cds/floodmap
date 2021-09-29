import rioxarray as rxr
import xarray as xa
from floodmap.util.plot import plot_array, floodmap_colors, plot_arrays

day_range = [ 245, 250 ]
tile = "h20v09"
rasters = {}

for day in range( *day_range ):
    fname = f"MCDWD_L3_F2_NRT.A2021{day}.{tile}"
    input_file = f"/Users/tpmaxwel/Development/Data/WaterMapping/Results/{tile}/allData/61/MCDWD_L3_F2_NRT/Recent/{fname}.061.tif"
    raster: xa.DataArray = rxr.open_rasterio( input_file ).squeeze( drop=True )
    rasters[day] = raster.where( raster < 10, 4 )

plot_arrays( f"Floodmap: tile={tile}", rasters, floodmap_colors )
