import rioxarray as rxr
from pyhdf.SD import SD, SDC
import netCDF4
import xarray as xa
from floodmap.util.plot import plot_array, floodmap_colors, plot_arrays

day_range = [ 249, 250 ]
tile = "h20v09"
product = 'Flood 2-Day 250m'
rasters = {}

for day in range( *day_range ):
    fname = f"MCDWD_L3_NRT.A2021{day}.{tile}"
    fpath = f"/Users/tpmaxwel/Development/Data/WaterMapping/Results/{tile}/allData/61/MCDWD_L3_F2_NRT/Recent/{fname}.061.hdf"
    print( f"plotting: '{fpath}'")
    file: SD = SD( fpath, SDC.READ )
    datasets = file.datasets()
#    for idx, sds in enumerate( datasets.keys() ):
#        print( idx, sds )
    sds_obj = file.select(product)
    data = sds_obj.get()
    raster = xa.DataArray( data, dims = ["x","y"], name = fname )
    rasters[day] = raster
    file.end()

#        rasters[day] = raster.where( raster < 10, 4 )

plot_arrays( f"{product}: tile={tile}", rasters, floodmap_colors )


