from floodmap_legacy.util.plot import plot_arrays
import xarray as xa
import matplotlib.pyplot as plt
from floodmap_legacy.util.configuration import opSpecs
results_dir = opSpecs.get('results_dir')

result_color_map = {
    0: (0, 0, 0),  # , 'nodata',
    1: (0, 1, 0),  # , 'land',
    2: (0, 0, 1),  # , 'water',
    3: (0, 0.6, 0),  # , 'int-land',
    4: (0, 0, 0.6),  # , 'int-water',
    5: (0.6, 0.6, 0.6)  # 'mask',
}

lake_index = 4       #  [4, 5, 9, 11, 12, 14, 19, 21, 22, 26, 28, 37, 42, 43, 44, 51, 53, 60, 66, 67, 69, 73, 74, 76, 79, 81, 82, 85, 87, 88, 91, 93, 97, 99]
type = "nc"

cmap = result_color_map
specs = opSpecs._defaults
floodmap_result_file = f"{results_dir}/lake_{lake_index}_water_maps.{type}"


if type=="tif":
    floodmap: xa.DataArray = xa.open_rasterio( floodmap_result_file )
else:
    floodmap_dset: xa.Dataset = xa.open_dataset(floodmap_result_file)
#    print( floodmap_dset.attrs )
    floodmap: xa.DataArray = floodmap_dset[ f"water_maps" ]

print("1 ")
floodmaps = { i: floodmap[i] for i in range( floodmap.shape[0]) }
figure, ax = plt.subplots()
plot_arrays( ax, floodmaps, title=f"Floodmap[{lake_index}] result"  )
plt.show()



