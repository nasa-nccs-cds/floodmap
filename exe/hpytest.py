import os
from pyhdf.SD import SD, SDC
from floodmap.util.plot import plot_array
import numpy as np

data_dir = os.path.expanduser( "~/Development/Data/WaterMapping/Lance" )
data_file = "MCDWD_L3_NRT.A2020289.h00v02.061.hdf"
data_file_path = os.path.join( data_dir, data_file )
print( f" OPENING File: {data_file_path}" )

hdf = SD( data_file_path, SDC.READ )

grid_attr = hdf.attributes().get( "StructMetadata.0")
print( f"\n hdf grid_attr methods: {dir(hdf.attributes())}:" )

# attr = hdf.attributes()
#
# print( f"\n hdf file attributes:" )
# for key, value in attr.items():
#     print( "\n--------------------------------------------------------------------------------------------------------")
#     print( f"**{key}: {value}")
#
# print( f"\n\n datasets: " )
# for dset in hdf.datasets():
#     print( f" --- {dset}" )

DATAFIELD_NAME='Flood 3-Day 250m'
flood_data = hdf.select(DATAFIELD_NAME)

print( f"\n dataset methods: {dir(flood_data)}" )
print( f"\n dataset dimensions: {flood_data.dimensions()}" )
print( f"\n dataset attributes: {flood_data.attributes()}" )
print( f"\n dataset datastrs: {flood_data.getdatastrs()}" )
print( f"\n dataset range: {flood_data.getrange()}" )
# print( f"\n dataset dim0 scale: {flood_data.dim(0).getscale()}" )
print( f"\n dataset dim0 attributes: {flood_data.dim(0).attributes()}" )


np_data: np.ndarray = np.array( flood_data.get() )