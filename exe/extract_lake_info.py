from multiprocessing import freeze_support
from floodmap.surfaceMapping.processing import LakeMaskProcessor
from floodmap.util.xgeo import XGeo
import xarray as xa
from typing import Dict, List, Tuple, Optional
from floodmap.util.configuration import opSpecs

if __name__ == '__main__':
    results_dir = opSpecs.get('results_dir')
    lakeMaskProcessor = LakeMaskProcessor()
    lake_locations_path = f"{results_dir}/lake_locations.csv"
    lake_masks: Dict[int,str] = lakeMaskProcessor.getLakeMasks()
    print( f"Writing lake locations: {lake_locations_path}")
    key_lakes = [26, 314, 333, 334, 336, 337]
    with open(lake_locations_path,"w") as lake_locations_file:
        lake_locations_file.write(f"index,size,lon,lat\n")
        for lake_index, lake_file in lake_masks.items():
            mask: xa.DataArray = XGeo.loadRasterFile( lake_file )
            umask = mask.xgeo.to_utm( (250,250) )
            lsize = (umask==1).sum().values/16.0
            centroid = mask.xgeo.centroid()
            lake_info = f"{lake_index}, {lsize}, {centroid[0]}, {centroid[1]}"
            lake_locations_file.write( f"{lake_info}\n" )
            print( lake_info )


