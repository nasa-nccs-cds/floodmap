from floodmap.util.plot import plot_array
import xarray as xa

lake_index = 5
tile_file = f"/Users/tpmaxwel/Development/Data/WaterMapping/Results/lake_{lake_index}_patched_water_masks.tif"

raster: xa.DataArray = xa.open_rasterio( tile_file )
plot_array( f"Lake {lake_index}", raster[0] )
