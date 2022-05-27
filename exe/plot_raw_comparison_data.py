from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import rioxarray, xarray as xa
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

def get_rect( extent, color='r' ) -> patches.Rectangle:
    w, h = (extent[1]-extent[0]), (extent[3]-extent[2])
    return patches.Rectangle((extent[0], extent[2]), w, h, linewidth=1, edgecolor=color, facecolor='none')

def subset( raster: xa.DataArray, extent ) -> xa.DataArray:
    return raster.loc[ extent[3]:extent[2], extent[0]:extent[1] ]

def get_color_map( colors: List ) -> Tuple[Normalize,LinearSegmentedColormap]:
    nc = len(colors)
    norm = Normalize(0, len(colors) - 1)
    color_map = LinearSegmentedColormap.from_list("lake-map", colors, N=nc)
    return (norm, color_map)

full_tile = True
tile_path = "simple/2021/h09v05_v2/2021-041-h09v05-MOD-simple.tif"
target_file_path = f'/Users/tpmaxwel/GDrive/Tom/Data/Birkitt/{tile_path}'
colors  = [ "green", "blue", "grey", "black" ]
norm, cmap = get_color_map( colors )

raw_raster: xa.DataArray = rioxarray.open_rasterio( target_file_path ).squeeze(drop=True)
fillval = raw_raster.attrs.get( "_FillValue", np.nan )
back_val = -999.00

raw_raster = raw_raster.where( raw_raster != fillval, 3.0 )
raw_raster = raw_raster.where( raw_raster != back_val, 2.0 )

figure, axs = plt.subplots()
raw_raster.plot.imshow( ax=axs, norm=norm, cmap=cmap )
plt.title(tile_path)

plt.show()