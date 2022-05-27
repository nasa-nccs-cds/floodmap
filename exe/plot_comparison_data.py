from typing import Dict, List, Tuple, Optional, Union
from floodmap.util.xrio import XRio
import matplotlib.pyplot as plt
from floodmap.surfaceMapping.processing import LakeMaskProcessor
import rioxarray, xarray as xa
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

def get_rect( extent, color='r' ) -> patches.Rectangle:
    w, h = (extent[1]-extent[0]), (extent[3]-extent[2])
    return patches.Rectangle((extent[0], extent[2]), w, h, linewidth=1, edgecolor=color, facecolor='none')

def subset( raster: xa.DataArray, extent ) -> xa.DataArray:
    return raster.loc[ extent[3]:extent[2], extent[0]:extent[1] ]

def get_color_map( colors: Dict ) -> Tuple[Normalize,LinearSegmentedColormap]:
    nc = len(colors)
    colors = [colors[ic] for ic in range(len(colors))]
    norm = Normalize(0, len(colors) - 1)
    color_map = LinearSegmentedColormap.from_list("lake-map", colors, N=nc)
    return (norm, color_map)

full_tile = True
target_file_path = '/Users/tpmaxwel/GDrive/Tom/Data/Birkitt/simple/2021/h09v05_v2/2021-041-h09v05-MOD-simple.tif'
tile_raster: Optional[xa.DataArray] =  XRio.load( target_file_path ).squeeze( drop=True )

if full_tile:
    figure, axs = plt.subplots()
    tile_raster.squeeze( drop=True ).plot.imshow( ax=axs )
else:
    lakeMaskProcessor = LakeMaskProcessor()
    lake_masks: Dict[int, Union[str, List[float]]] = lakeMaskProcessor.getLakeMasks()
    lake_ids = [368, 453, 461, 462, 468, 1261, 1273, 1278, 1279, 1602, 1603, 1604]
    idr = [8, 12]
    lakes = {lake_id: lakeMaskProcessor.read_lake_mask(lake_id, lake_masks[lake_id]) for lake_id in lake_ids[idr[0]:idr[1]]}

    figure, axs = plt.subplots(len(lakes), 2)
    fillval = tile_raster.attrs['_FillValue']
    backval = -999.0
    valid_mask = (tile_raster >= 0.0)
    tile_raster = tile_raster.where(valid_mask, np.nan)
    for iax, (lid, lspec) in enumerate(lakes.items()):
        subset( tile_raster, lspec['roi'] ).plot.imshow( ax=axs[iax,0] )
        lspec['mask'].squeeze( drop=True ).plot.imshow( ax=axs[iax,1] )

plt.show()