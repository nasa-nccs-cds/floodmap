import numpy as np

from floodmap.util.plot import plot_array
import rioxarray, xarray as xa
import matplotlib.pyplot as plt
from  floodmap.surfaceMapping.mwp import MWPDataManager
from floodmap.util.configuration import opSpecs
results_dir = '/Volumes/Shared/Data/floodmap/Results/tpmaxwel/'

result_color_map = {
    0: (0, 0, 0),  # , 'nodata',
    1: (0, 1, 0),  # , 'land',
    2: (0, 0, 1),  # , 'water',
    3: (0, 0.6, 0),  # , 'int-land',
    4: (0, 0, 0.6),  # , 'int-water',
    5: (1, 1, 0.7)  # 'mask',
}

lake_index = 1279
type = "nc"
day=41
year = 2021
plot_type = "patched_water" #  "water" "patched_water" "persistent_class"
dstr =    f"{year}{day:03}"

cmap = result_color_map
specs = opSpecs._defaults
floodmap_result_file = f"{results_dir}/lake_{lake_index}_{plot_type}_map_{dstr}.{type}"

if type=="tif":
    floodmap: xa.DataArray = rioxarray.open_rasterio( floodmap_result_file )
    print( f"Result vrange = {[np.nanmin(floodmap.values),np.nanmax(floodmap.values)]}")
else:
    floodmap_dset: xa.Dataset = xa.open_dataset(floodmap_result_file)
#    print( floodmap_dset.attrs )
    floodmap: xa.DataArray = floodmap_dset[ f"Lake-{lake_index}" ]

#(xc,yc) = ( floodmap.x.values[floodmap.x.size//2], floodmap.y.values[floodmap.y.size//2] )
#center = floodmap.xgeo.project_to_geographic( xc, yc )
#print( f"Lake center: {center}")
figure, ax = plt.subplots()
plot_array( ax, floodmap.squeeze(), title=f"Floodmap {plot_type} result"  )
plt.show()



