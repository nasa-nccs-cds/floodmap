from floodmap.util.plot import plot_array
from floodmap.util.configuration import opSpecs
from  floodmap.surfaceMapping.mwp import MWPDataManager
import xarray as xa

result_color_map = {
    0: (0, 0, 0),  # , 'nodata',
    1: (0, 1, 0),  # , 'land',
    2: (0, 0, 1),  # , 'water',
    3: (0, 0.6, 0),  # , 'int-land',
    4: (0, 0, 0.6),  # , 'int-water',
    5: (1, 1, 0.7)  # 'mask',
}

lake_index = 4
fps =  0.5
stage = "raw"
plot_var = "water_map"
cmap = result_color_map

specs = opSpecs._defaults
floodmap_result_file = f"/Users/tpmaxwel/Development/Data/WaterMapping/Results/lake_{lake_index}_water_map.nc"

floodmap_dset: xa.Dataset = xa.open_dataset( floodmap_result_file )
floodmap_data: xa.DataArray = floodmap_dset.data_vars[plot_var]

plot_array( f"Floodmap {stage} result", floodmap_data )

