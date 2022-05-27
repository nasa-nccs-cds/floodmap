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

full_tile = False
plot_subtiles = False
tile_comparison = False
target_file_path = '/Users/tpmaxwel/GDrive/Tom/Data/Birkitt/simple/2021/h09v05_v2/2021-041-h09v05-MOD-simple.tif'
result_path = "/Volumes/Shared/Data/floodmap/comparison/randomforest"
tile_raster: Optional[xa.DataArray] =  XRio.load( target_file_path ).squeeze( drop=True )
lake_ids = [ 453, 462 ] # [368, 453, 461, 462, 468]  # 1261, 1273, 1278, 1279, 1602, 1603, 1604]

if full_tile:
    figure, axs = plt.subplots()
    tile_raster.squeeze( drop=True ).plot.imshow( ax=axs )
elif plot_subtiles:
    lakeMaskProcessor = LakeMaskProcessor()
    lake_masks: Dict[int, Union[str, List[float]]] = lakeMaskProcessor.getLakeMasks()
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
elif tile_comparison:
    day = 100
    figure, axs = plt.subplots(len(lake_ids), 2)
    for iax, lake_id in enumerate(lake_ids):
        result_file = f"{result_path}/lake_{lake_id}_patched_water_map_2021365.nc"
        data_file = f"{result_path}/lake_{lake_id}_nrt_input_data.nc"
        result_dset: xa.Dataset = xa.open_dataset(result_file)
        data_dset: xa.Dataset = xa.open_dataset(data_file)
        result_raster: Optional[xa.DataArray] = result_dset[f'Lake-{lake_id}']
        data_raster: Optional[xa.DataArray] = data_dset['mpw'][day].squeeze( drop=True )
        data_raster.plot.imshow( ax=axs[iax,0] )
        result_raster.plot.imshow(ax=axs[iax, 1])
else:
    day = 100
    lake_id = 462
    figure, axs = plt.subplots()
    result_file = f"{result_path}/lake_{lake_id}_patched_water_map_2021365.nc"
    result_dset: xa.Dataset = xa.open_dataset(result_file)
    result_raster: Optional[xa.DataArray] = result_dset[f'Lake-{lake_id}']
    result_raster.plot.imshow( ax=axs )

plt.show()