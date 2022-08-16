import rioxarray as rxr
import os, numpy as np
import xarray as xa

day_range = [ 5, 200 ]
year = 2021
tile = "h06v05"
w = 0.1
data_dir= "/explore/nobackup/projects/ilab/projects/Birkett/MOD44W/data"
xbnds = [ -111.6, -110.4 ]
ybnds = [ 37.8, 36.9 ]
rave = None

for day in range( *day_range ):
    fname = f"MCDWD_L3_F2_NRT.A{year}{day:03}.{tile}"
    input_file = f"{data_dir}/{tile}/allData/61/MCDWD_L3_F2_NRT/Recent/{fname}.061.tif"
    if os.path.isfile( input_file ):
        raster: xa.DataArray = rxr.open_rasterio( input_file ).squeeze( drop=True )
        raster = raster.sel( x=slice(*xbnds), y=slice(*ybnds) )
        water_mask: xa.DataArray = raster.isin([1, 2, 3])
        nwater = np.count_nonzero( water_mask )
        rave = nwater if (rave is None) else (1-w)*rave + w*nwater
        print( f"Day-{day}: #water={nwater}, rave={rave}")

