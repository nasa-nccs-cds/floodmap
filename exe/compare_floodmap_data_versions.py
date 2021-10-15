from floodmap.surfaceMapping.tiles import TileLocator
import os
from floodmap.util.configuration import opSpecs

if __name__ == '__main__':
    data_loc = opSpecs.get('results_dir')
    nrt_path = "allData/61/MCDWD_L3_F2_NRT/Recent"
    year = 2021
    days = [0,260]
    (xmin, xmax, ymin, ymax) = (-86.9, -86.8, 47.6, 47.7)

    for day in range(*days):
        legacy_tile = TileLocator.get_tiles_legacy(xmin, xmax, ymin, ymax)[0]
        legacy_data_file = f"{data_loc}/{legacy_tile}/MWP_{year}{day:03}_{legacy_tile}_2D2OT.tif"

        nrt_tile = TileLocator.get_tiles(xmin, xmax, ymin, ymax)[0]
        nrt_data_file = f"{data_loc}/{nrt_tile}/{nrt_path}/MCDWD_L3_F2_NRT.A{year}{day:03}.{nrt_tile}.061.tif"

        if not os.path.isfile(legacy_data_file): print( f"Legacy file does not exist: {legacy_data_file}" )
        elif not os.path.isfile(nrt_data_file):  print( f" NRT file does not exist: {nrt_data_file} " )
        else:
            print( f" -------------- Day: {day} -------------------------- " )
            print( f"Legacy: {legacy_data_file}" )
            print( f"NRT:    {nrt_data_file}" )