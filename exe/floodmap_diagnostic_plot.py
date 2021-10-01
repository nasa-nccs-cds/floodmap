from floodmap.util.plot import plot_array
from floodmap.util.configuration import opSpecs
import xarray as xa
import matplotlib.pyplot as plt
results_dir = opSpecs.get('results_dir')

lake_index = 4
diag_type = "probability"
cmap = "jet"
title = f"{diag_type}, Lake {lake_index}"

specs = opSpecs._defaults
diag_files = dict(
    reliability = ( f"{results_dir}/lake_{lake_index}_water_map.nc", "reliability" ),
    probability = ( f"{results_dir}/lake_{lake_index}_water_probability.nc", "water_probability" ),
)

diag_data = diag_files[diag_type]
floodmap_dset: xa.Dataset = xa.open_dataset( diag_data[0] )
floodmap_data: xa.DataArray = floodmap_dset.data_vars[ diag_data[1] ]

figure, axes = plt.subplots(1, 1)
figure.suptitle(title, fontsize=12)
floodmap_data.plot.imshow( ax=axes, cmap=cmap )
plt.show()

