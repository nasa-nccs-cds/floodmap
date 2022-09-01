import rioxarray as rxr
import os
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import matplotlib.pyplot as plt
import xarray as xa
from floodmap.util.plot import plot_array, floodmap_colors, plot_arrays, result_colors
from floodmap.surfaceMapping.lakeExtentMapping import WaterMapGenerator
from floodmap.surfaceMapping.processing import LakeMaskProcessor
from  floodmap.surfaceMapping.mwp import MWPDataManager
from matplotlib.widgets import MultiCursor
from floodmap.util.configuration import opSpecs
from multiprocessing import freeze_support
results_dir = opSpecs.get( 'results_dir' )
water_maps_specs = opSpecs.get( 'water_maps' )
bin_size = water_maps_specs.get( 'bin_size', 8 )
rasters = {}

lake_index = 462                  #  [4, 5, 9, 11, 12, 14, 19, 21, 22, 26, 28, 37, 42, 43, 44, 51, 53, 60, 66, 67, 69, 73, 74, 76, 79, 81, 82, 85, 87, 88, 91, 93, 97, 99]
day = 255

run_cfg = { 'day': day, "lake_masks:lake_index": lake_index }
dataMgr = MWPDataManager.instance( **run_cfg )
( year, target_day ) = dataMgr.target_date()

count_water = [     ( 0, '0',  (0.0, 0.0, 0)),
                    ( 1, '1',  (0.0, 0.0, 0.5)),
                    ( 2, '2',  (0.2, 0.2, 0.6)),
                    ( 3, '3',  (0.4, 0.4, 0.7)),
                    ( 4, '4',  (0.6, 0.6, 0.8)),
                    ( 5, '5',  (0.7, 0.7, 0.9)),
                    ( 6, '6',  (0.8, 0.8, 1.0)) ]

count_land =  [     ( 0, '0',  (0.0, 0, 0.0)),
                    ( 1, '1',  (0.0, 0.5, 0.0)),
                    ( 2, '2',  (0.2, 0.6, 0.2)),
                    ( 3, '3',  (0.4, 0.7, 0.4)),
                    ( 4, '4',  (0.6, 0.8, 0.6)),
                    ( 5, '5',  (0.7, 0.9, 0.7)),
                    ( 6, '6',  (0.8, 1.0, 0.8)) ]

if __name__ == '__main__':
    freeze_support()
    lake_masks = LakeMaskProcessor.getLakeMasks()
    lake_mask_specs = LakeMaskProcessor.read_lake_mask( lake_index, lake_masks[lake_index] )
    roi: List[float] = lake_mask_specs['roi']
    mask: Optional[xa.DataArray] = lake_mask_specs['mask']
    locations = dataMgr.list_required_tiles(roi=roi, lake_mask=mask)
    tile= locations[0]

    water_map_file = f"{results_dir}/lake_{lake_index}_patched_water_map_2021{target_day}.nc"
    if not os.path.isfile( water_map_file ):
        lakeMaskProcessor = LakeMaskProcessor()
        lakeMaskProcessor.process_lakes( )
    water_map_dset: xa.Dataset = xa.open_dataset(water_map_file)

    plot_coords = dict( x="lon", y="lat" )
    for day in range( target_day-bin_size, target_day+1 ):
        fname = f"MCDWD_L3_F2_NRT.A2021{day:03}.{tile}"
        input_file = f"{results_dir}/{tile}/allData/61/MCDWD_L3_F2_NRT/Recent/{fname}.061.tif"
        if os.path.isfile( input_file ):
            raster: xa.DataArray = rxr.open_rasterio( input_file ).squeeze( drop=True )
            sub_raster = raster.sel( x=slice(roi[0],roi[1]), y=slice(roi[3],roi[2]) ).rename(  plot_coords )
            rasters[day] = WaterMapGenerator.update_classes( sub_raster )

    (water, land) =  WaterMapGenerator.get_class_count_layers( rasters )
    ext = water.xgeo.extent()
    floodmap: xa.DataArray = water_map_dset[ f"Lake-{lake_index}" ].sel( lat=slice(ext[3],ext[2]), lon=slice(ext[0],ext[1]) )

    figure, npaxs = plt.subplots( 2, 2, sharex='all', sharey='all' )
    axs = npaxs.flatten().tolist()
    multi = MultiCursor( figure.canvas, axs, color='r', lw=1, horizOn=True, vertOn=True )
    figure.suptitle( f"Floodmap: tile={tile}", fontsize=12 )
    probe_arrays = OrderedDict( water_counts=water, land_counts=land, result_class=floodmap )
    plot_arrays( axs[0], rasters,  colors=result_colors, cursor=multi, title=f"MWP Data for Lake {lake_index}", probe_arrays=probe_arrays, plot_coords=plot_coords )
    plot_array(  axs[1], floodmap, colors=result_colors, title=f"Water Map for {year}:{target_day}", plot_coords=plot_coords )
    plot_array(  axs[2], water, colors=count_water, title="Water counts", plot_coords=plot_coords )
    plot_array(  axs[3], land,  colors=count_land, title="Land counts", plot_coords=plot_coords  )

    plt.show()



