from floodmap.util.anim import ArrayAnimation
from floodmap.util.configuration import opSpecs
from  floodmap.surfaceMapping.mwp import MWPDataManager
import xarray as xa

lake_index = 4
fps =  0.5

color_map = {
    0: (0, 1, 0),  # , 'land',
    1: (0, 0, 1),  # , 'water',
    2: (0, 0.5, 1),  # , 'Flood',
    3: (0, 0.3, 1),  # , 'Perm Flood',
    4: (0, 0, 0)  # 'nodata',
}
result_color_map = {
    0: (0, 0, 0),  # , 'nodata',
    1: (0, 1, 0),  # , 'land',
    2: (0, 0, 1),  # , 'water',
    3: (0, 0.6, 0),  # , 'int-land',
    4: (0, 0, 0.6),  # , 'int-water',
    5: (1, 1, 0.7)  # 'mask',
}

specs = opSpecs._defaults
floodmap_data_file = f"/Users/tpmaxwel/Development/Data/WaterMapping/Results/lake_{lake_index}_floodmap_data.nc"
floodmap_data: xa.DataArray = xa.open_rasterio( floodmap_data_file )
data_arrays = [ floodmap_data[i] for i in range(floodmap_data.shape[0])]
roi = MWPDataManager.extent( floodmap_data.transform, floodmap_data.shape, "upper" )

animator = ArrayAnimation( roi=roi, fps=fps )
anim = animator.create_animation( data_arrays, color_map=color_map )
#    anim = animator.create_watermap_diag_animation( f"{product} @ {location}", data_arrays, savePath, True )
