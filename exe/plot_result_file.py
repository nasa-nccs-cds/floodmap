import rioxarray as rxr
import os
import matplotlib.pyplot as plt
import xarray as xa

file_index = 0
day_range = [ 334, 348 ]
tile = "h18v02"
rasters = {}
data_dir= "/Users/tpmaxwel/GDrive/Tom/Data/Birkitt/Results"
day = day_range[ file_index ]

fname = f"MCDWD_L3_F2_NRT.A2021{day:03}.{tile}"
input_file = f"{data_dir}/{fname}.061.tif"
raster: xa.DataArray = rxr.open_rasterio( input_file ).squeeze( drop=True )
print( f"Plotting file {input_file}, maxval = {raster.max()}")

figure, ax = plt.subplots()
raster.plot.imshow( ax=ax )
plt.show()