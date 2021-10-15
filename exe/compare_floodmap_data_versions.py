from floodmap.surfaceMapping.tiles import TileLocator
import os
import xarray as xa
import numpy  as np
from floodmap.util.configuration import opSpecs

if __name__ == '__main__':
    data_loc = opSpecs.get('results_dir')
    nrt_path = "allData/61/MCDWD_L3_F2_NRT/Recent"
    year = 2021
    days = [10,20]
    (xmin, xmax, ymin, ymax) = (-86.9, -86.8, 47.6, 47.7)

    for day in range(*days):
        legacy_tile = TileLocator.get_tiles_legacy(xmin, xmax, ymin, ymax)[0]
        legacy_data_file = f"{data_loc}/{legacy_tile}/MWP_{year}{day:03}_{legacy_tile}_2D2OT.tif"

        nrt_tile = TileLocator.get_tiles(xmin, xmax, ymin, ymax)[0]
        nrt_data_file = f"{data_loc}/{nrt_tile}/{nrt_path}/MCDWD_L3_F2_NRT.A{year}{day:03}.{nrt_tile}.061.tif"

        if not os.path.isfile(legacy_data_file): print( f"\nLegacy file does not exist: {legacy_data_file}\n" )
        elif not os.path.isfile(nrt_data_file):  print( f"\nNRT file does not exist: {nrt_data_file}\n" )
        else:
            print( f" -------------- Day: {day} -------------------------- " )
            legacy_data: xa.DataArray = xa.open_rasterio(legacy_data_file).squeeze(drop=True)
            nrt_data: xa.DataArray = xa.open_rasterio(nrt_data_file).squeeze(drop=True)
            legacy_nodata_mask = (legacy_data == 0)
            nrt_nodata_mask = (nrt_data == 255)
            legacy_nt = legacy_data.size
            legacy_nz = np.count_nonzero(legacy_nodata_mask.values)
            nrt_nt = nrt_data.size
            nrt_nz = np.count_nonzero(nrt_nodata_mask.values)
            print(f" LEGACY: {legacy_nz} / {legacy_nt}:  {(legacy_nz * 100.0) / legacy_nt:.2f} %")
            print(f" NRT:    {nrt_nz} / {nrt_nt}:  {(nrt_nz * 100.0) / nrt_nt:.2f} %")


