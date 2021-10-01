from floodmap.util.plot import plot_array
import xarray as xa
from floodmap.util.configuration import opSpecs
results_dir = opSpecs.get('results_dir')

lake_index = 5
tile_file = f"{results_dir}/lake_{lake_index}_patched_water_map.tif"

raster: xa.DataArray = xa.open_rasterio( tile_file )
plot_array( f"Lake {lake_index}", raster[0] )
