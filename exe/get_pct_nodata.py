from  floodmap.surfaceMapping.mwp import MWPDataManager
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from floodmap.util.xrio import XRio
import xarray as xa
from floodmap.util.configuration import opSpecs

if __name__ == '__main__':
    lakeMaskSpecs: Dict = opSpecs.get("lake_masks", None)
    source_specs: Dict = opSpecs.get('source')
#    rbnds = [-75,30,-120,70]
    rbnds = [ -86.9, 47.6, -86.8, 47.7 ]

    dataMgr = MWPDataManager.instance()
    locations = dataMgr.infer_tile_locations( roi=rbnds )
    dataMgr.download_mpw_data( locations, **source_specs )
    for location in locations:
        tile_filespec: OrderedDict = dataMgr.get_tile(location)
        file_paths = list(tile_filespec.values())
        time_values = list(tile_filespec.keys())
        tile_raster: Optional[xa.DataArray] = XRio.load( file_paths, band=0, index=time_values )


