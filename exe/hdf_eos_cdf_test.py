import os
from floodmap.util.plot import plot_array
import xarray as xa

data_dir = os.path.expanduser( "~/Development/Data/WaterMapping/Lance" )
data_file = "MCDWD_L3_NRT.A2020289.h00v02.061.nc"

dset: xa.Dataset = xa.open_dataset( os.path.join( data_dir, data_file ) )
flood_map: xa.DataArray = dset['Flood_3_Day_250m']

plot_array( data_file, flood_map )
