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

lake_index = 1604
dataMgr = MWPDataManager.instance()
cmap = result_color_map
specs = opSpecs._defaults
floodmap_data_file = f"{results_dir}/lake_{lake_index}_nrt_input_data.nc"
floodmap_dset: xa.Dataset = xa.open_dataset(floodmap_data_file)
mpw: xa.DataArray = floodmap_dset['mpw']
center = ( mpw.x.values[mpw.x.size//2], mpw.y.values[mpw.y.size//2] )
print( f"Lake {lake_index} location: {center}")
nplots = mpw.shape[0]

figure, axs = plt.subplots( 1, nplots, sharex='all', sharey='all' )
for day in range(nplots):
    ix,iy = day//4, day%4
    ax = axs[day] if nplots > 1 else axs
    plot_array( ax, mpw[day].squeeze(), title=f"WPW data: jday={day}"  )
plt.show()



