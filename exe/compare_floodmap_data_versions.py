from floodmap.surfaceMapping.tiles import TileLocator
from typing import Dict, List, Tuple, Optional
import os, glob
import xarray as xa
import numpy  as np
from floodmap.util.configuration import opSpecs

def get_centroid( tile: str ) -> Tuple[float,float]:
    x, xs = int(tile[0:3]), tile[4]
    y, ys = int(tile[4:7]), tile[7]
    if xs == 'W': x = -x
    if ys == 'S': y = -y
    return ( float(x+5), float(y-5) )

if __name__ == '__main__':
    data_loc = opSpecs.get('results_dir')
    nrt_path = "allData/61/MCDWD_L3_F2_NRT/Recent"
    legacy_tiles = [ os.path.basename(tpath) for tpath in glob.glob(f"{data_loc}/???[EW]???[NS]") ]
    for tile in legacy_tiles:
        pos = get_centroid( tile )
        tile_test = TileLocator.get_tiles_legacy(pos[0], pos[0], pos[1], pos[1])[0]
        nrt_tile = TileLocator.get_tiles(pos[0], pos[0], pos[1], pos[1])[0]
        print( f" {pos}-> {tile}={tile_test}: {nrt_tile}" )


    exit(0)
    scale = 0.00001
    year = 2021
    days = [10,260]
    (xmin, xmax, ymin, ymax) = (-86.9, -86.8, 47.6, 47.7)
    legacy_nodata, legacy_size = [], []
    nrt_nodata, nrt_size = [], []

    for day in range(*days):
        legacy_tile = TileLocator.get_tiles_legacy(xmin, xmax, ymin, ymax)[0]
        legacy_data_file = f"{data_loc}/{legacy_tile}/MWP_{year}{day:03}_{legacy_tile}_2D2OT.tif"

        nrt_tile = TileLocator.get_tiles(xmin, xmax, ymin, ymax)[0]
        nrt_data_file = f"{data_loc}/{nrt_tile}/{nrt_path}/MCDWD_L3_F2_NRT.A{year}{day:03}.{nrt_tile}.061.tif"

        if not os.path.isfile(legacy_data_file): print( f"\nLegacy file does not exist: {legacy_data_file}\n" )
        elif not os.path.isfile(nrt_data_file):  print( f"\nNRT file does not exist: {nrt_data_file}\n" )
        else:
#            print( f" -------------- Day: {day} -------------------------- " )
            legacy_data: xa.DataArray = xa.open_rasterio(legacy_data_file).squeeze(drop=True)
            nrt_data: xa.DataArray = xa.open_rasterio(nrt_data_file).squeeze(drop=True)
            legacy_nodata_mask = (legacy_data == 0)
            nrt_nodata_mask = (nrt_data == 255)
            legacy_nt = legacy_data.size
            legacy_nz = np.count_nonzero(legacy_nodata_mask.values)
            nrt_nt = nrt_data.size
            nrt_nz = np.count_nonzero(nrt_nodata_mask.values)
            legacy_nodata.append( legacy_nz * scale )
            legacy_size.append( legacy_nt * scale )
            nrt_nodata.append( nrt_nz * scale )
            nrt_size.append( nrt_nt * scale )

    legacy_total = np.array(legacy_nodata).sum() / np.array(legacy_size).sum()
    nrt_total = np.array(nrt_nodata).sum() / np.array(nrt_size).sum()

    print(f" LEGACY: {legacy_total*100} %")
    print(f" NRT:    {nrt_total*100} %")


