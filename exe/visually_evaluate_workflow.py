import rioxarray as rxr
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import xarray as xa
from floodmap.util.plot import plot_array, floodmap_colors, plot_arrays, result_colors
from floodmap.surfaceMapping.lakeExtentMapping import WaterMapGenerator
from floodmap.surfaceMapping.processing import LakeMaskProcessor
from  floodmap.surfaceMapping.mwp import MWPDataManager
from matplotlib.widgets import MultiCursor
from floodmap.util.configuration import opSpecs

results_dir = opSpecs.get( 'results_dir' )
water_maps_specs = opSpecs.get( 'water_maps' )
bin_size = water_maps_specs.get( 'bin_size', 8 )
rasters = {}

lake_index = 4

dataMgr = MWPDataManager.instance()
( year, target_day ) = dataMgr.target_date()

count_water = [     ( 0, '0',  (0.0, 0.0, 0)),
                    ( 1, '1',  (0.0, 0.0, 1)),
                    ( 2, '2',  (0.2, 0.2, 1)),
                    ( 3, '3',  (0.4, 0.4, 1)),
                    ( 4, '4',  (0.6, 0.6, 1)),
                    ( 5, '5',  (0.8, 0.8, 1)),
                    ( 6, '6',  (1.0, 1.0, 1)) ]

count_land =  [     ( 0, '0',  (0.0, 0, 0.0)),
                    ( 1, '1',  (0.0, 1, 0.0)),
                    ( 2, '2',  (0.2, 1, 0.2)),
                    ( 3, '3',  (0.4, 1, 0.4)),
                    ( 4, '4',  (0.6, 1, 0.6)),
                    ( 5, '5',  (0.8, 1, 0.8)),
                    ( 6, '6',  (1.0, 1, 1.0)) ]

lake_masks = LakeMaskProcessor.getLakeMasks()
lake_mask_specs = LakeMaskProcessor.read_lake_mask( lake_index, lake_masks[lake_index] )
roi: List[float] = lake_mask_specs['roi']
mask: Optional[xa.DataArray] = lake_mask_specs['mask']
locations = dataMgr.infer_tile_locations(roi=roi, lake_mask=mask)
tile= locations[0]

floodmap_result_file = f"{results_dir}/lake_{lake_index}_patched_water_map_2021{target_day}.nc"
floodmap_dset: xa.Dataset = xa.open_dataset(floodmap_result_file)
floodmap: xa.DataArray = floodmap_dset[ f"Lake-{lake_index}" ]

for day in range( target_day-bin_size, target_day+1 ):
    fname = f"MCDWD_L3_F2_NRT.A2021{day:03}.{tile}"
    input_file = f"{results_dir}/{tile}/allData/61/MCDWD_L3_F2_NRT/Recent/{fname}.061.tif"
    if os.path.isfile( input_file ):
        raster: xa.DataArray = rxr.open_rasterio( input_file ).squeeze( drop=True )
        sub_raster = raster.sel( x=slice(roi[0],roi[1]), y=slice(roi[3],roi[2]) ).rename(  dict( x="lat", y="lon" ) )
        rasters[day] = WaterMapGenerator.update_classes( sub_raster )

(water, land) =  WaterMapGenerator.get_class_count_layers( rasters )
figure, npaxs = plt.subplots( 2, 2, sharex='all', sharey='all' )
axs = npaxs.flatten().tolist()
multi = MultiCursor( figure.canvas, axs, color='r', lw=1, horizOn=True, vertOn=True )
figure.suptitle( f"Floodmap: tile={tile}", fontsize=12 )
plot_arrays( axs[0], rasters,  colors=result_colors, cursor=multi, title=f"MWP Data for Lake {lake_index}" )
plot_array(  axs[1], floodmap, colors=result_colors, title=f"Water Map for {year}:{target_day}" )
plot_array(  axs[2], water, colors=count_water, title="Water counts" )
plot_array(  axs[3], land,  colors=count_land, title="Land counts"  )

plt.show()



