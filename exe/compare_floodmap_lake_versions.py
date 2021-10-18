from floodmap.surfaceMapping.tiles import TileLocator
from typing import Dict, List, Tuple, Optional
import os, glob, sys
import xarray as xa
import numpy  as np
from floodmap.util.configuration import opSpecs
from floodmap.surfaceMapping.processing import LakeMaskProcessor
from floodmap.surfaceMapping.lakeExtentMapping import WaterMapGenerator

def pct_diff( x0: float, x1: float) -> float:
    sgn = 1 if (x1>x0) else -1
    return sgn * (abs(x1 - x0) * 100) / min(x0, x1)

def get_centroid( tile: str ) -> Tuple[float,float]:
    x, xs = int(tile[0:3]), tile[3]
    y, ys = int(tile[4:7]), tile[7]
    if xs == 'W': x = -x
    if ys == 'S': y = -y
    return ( float(x+5), float(y-5) )

if __name__ == '__main__':
    data_loc = opSpecs.get('results_dir')
    nrt_path = "allData/61/MCDWD_L3_F2_NRT/Recent"
    scale = 0.00001
    year = 2021
    days = [10,260]
    tot_legacy_nodata, tot_legacy_size = [], []
    tot_nrt_nodata, tot_nrt_size = [], []
    output_file_path = f"{data_loc}/nodata_stats_comparison.csv"
    with open( output_file_path, "w" ) as output_file:
        lake_masks = LakeMaskProcessor.getLakeMasks()

        for ( lake_index, lake_mask_bounds ) in lake_masks.items():
            lake_mask_specs = LakeMaskProcessor.read_lake_mask( lake_index, lake_mask_bounds )

            (floodmap_data, time_values) = WaterMapGenerator.get_mpw_data( **lake_mask_specs )

            print( f"floodmap_data[{lake_index}]{floodmap_data.dims}, shape = {floodmap_data.shape} {floodmap_data.dims} ")
            exit(0)

            legacy_nodata, legacy_size = [], []
            nrt_nodata, nrt_size = [], []

            for day in range(*days):
                legacy_data_file = f"{data_loc}/{legacy_tile}/MWP_{year}{day:03}_{legacy_tile}_2D2OT.tif"
                nrt_data_file = f"{data_loc}/{nrt_tile}/{nrt_path}/MCDWD_L3_F2_NRT.A{year}{day:03}.{nrt_tile}.061.tif"

                if not os.path.isfile(legacy_data_file): print(f" LSKIP-{day}", end=''); sys.stdout.flush()
                elif not os.path.isfile(nrt_data_file):  print(f" NSKIP-{day}", end=''); sys.stdout.flush()
                else:
                    print(f" {day}", end=''); sys.stdout.flush()
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
            if len(legacy_size) == 0:
                print( f"\nSkipping tile {legacy_tile}: NO DATA")
            else:
                tile_legacy_nodata = np.array(legacy_nodata).sum()
                tile_legacy_size = np.array(legacy_size).sum()
                tile_nrt_nodata = np.array(nrt_nodata).sum()
                tile_nrt_size = np.array(nrt_size).sum()
                tile_legacy_pctn =  tile_legacy_nodata / tile_legacy_size
                tile_nrt_pctn = tile_nrt_nodata / tile_nrt_size
                print(f"\n\nTILE {legacy_tile}: LEGACY: {tile_legacy_pctn*100:.2f}%, NRT: {tile_nrt_pctn*100:.2f}%, pct_diff: {pct_diff(tile_legacy_pctn,tile_nrt_pctn):.2f}" )
                output_file.write( f"{legacy_tile}, {tile_legacy_pctn*100:.2f}, {tile_nrt_pctn*100:.2f}, {pct_diff(tile_legacy_pctn,tile_nrt_pctn):.2f}\n" )

            tot_legacy_nodata.append( tile_legacy_nodata )
            tot_legacy_size.append( tile_legacy_size )
            tot_nrt_nodata.append( tile_nrt_nodata )
            tot_nrt_size.append( tile_nrt_size )

        result_legacy_nodata = np.array(tot_legacy_nodata).sum()
        result_legacy_size = np.array(tot_legacy_size).sum()
        result_nrt_nodata = np.array(tot_nrt_nodata).sum()
        result_nrt_size = np.array(tot_nrt_size).sum()
        result_legacy_pctn = result_legacy_nodata / result_legacy_size
        result_nrt_pctn = result_nrt_nodata / result_nrt_size
        print(f"\n\nRESULT: LEGACY: {result_legacy_pctn * 100:.2f}%, NRT: {result_nrt_pctn * 100:.2f}%, pct_diff: {pct_diff(result_legacy_pctn,result_nrt_pctn):.2f}")
        output_file.write( f"TOTAL, {result_legacy_pctn * 100:.2f}, {result_nrt_pctn * 100:.2f}, {pct_diff(result_legacy_pctn, result_nrt_pctn):.2f}\n")





