from floodmap.util.plot import plot_floodmap_arrays
import xarray as xa
from floodmap.util.configuration import opSpecs

results_dir = opSpecs.get('results_dir')
lake_index = 4
floodmap_file = f"{results_dir}/lake_{lake_index}_floodmap_data.nc"

floodmap_dset: xa.Dataset = xa.open_dataset( floodmap_file )
for floodmap_data in floodmap_dset.data_vars.values():
    plot_floodmap_arrays( f"Lake {lake_index}", floodmap_data )
