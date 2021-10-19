from floodmap.surfaceMapping.tiles import TileLocator
from typing import Dict, List, Tuple, Optional
import os, glob, sys
import xarray as xa
import numpy  as np
from floodmap.util.configuration import opSpecs
from floodmap.surfaceMapping.processing import LakeMaskProcessor
from floodmap.surfaceMapping.lakeExtentMapping import WaterMapGenerator
from  floodmap.surfaceMapping.mwp import MWPDataManager

def pct_diff( x0: float, x1: float) -> float:
    sgn = 1 if (x1>x0) else -1
    return sgn * (abs(x1 - x0) * 100) / min(x0, x1)

def get_centroid( tile: str ) -> Tuple[float,float]:
    x, xs = int(tile[0:3]), tile[3]
    y, ys = int(tile[4:7]), tile[7]
    if xs == 'W': x = -x
    if ys == 'S': y = -y
    return ( float(x+5), float(y-5) )

def read_subset( legacy_data_file, lake_mask, apply_mask=False, **kwargs ):
    lake_mask_value = kwargs.get( 'lake_mask_value', 3 )
    mask_value = kwargs.get( 'result_mask_value', 100 )
    (x0, x1, y0, y1) = lake_mask.xgeo.extent()
    raster: xa.DataArray = xa.open_rasterio(legacy_data_file).squeeze(drop=True)
    tile_raster = raster.sel( x=slice(x0, x1), y= slice(y1, y0) )
    if apply_mask:
        lake_mask_interp: xa.DataArray = lake_mask.squeeze(drop=True).interp_like(tile_raster).fillna( lake_mask_value)
        tile_mask: xa.DataArray = (lake_mask_interp == lake_mask_value)
        tile_mask_data: np.ndarray = np.broadcast_to(tile_mask.values, tile_raster.shape).flatten()
        tile_raster_data: np.ndarray = tile_raster.values.flatten()
        tile_raster_data[tile_mask_data] = mask_value
        return tile_raster.copy( data=tile_raster_data.reshape(tile_raster.shape) )
    return tile_raster


if __name__ == '__main__':
    data_loc = opSpecs.get('results_dir')
    nrt_path = "allData/61/MCDWD_L3_F2_NRT/Recent"
    scale = 0.00001
    year = 2021
    days = [10,260]
    apply_mask = True
    tot_legacy_nodata, tot_legacy_size = [], []
    tot_nrt_nodata, tot_nrt_size = [], []
    mtype = "_masked" if apply_mask else ""
    output_file_path = f"{data_loc}/nodata_stats_comparison_lake{mtype}.csv"
    dataMgr = MWPDataManager.instance(day=260)
    waterMapGenerator = WaterMapGenerator()
    opSpecs.set('history_length', (days[1]-days[0]) )

    with open( output_file_path, "w" ) as output_file:
        print( f"Writing results to file {output_file_path}")
        lake_masks = LakeMaskProcessor.getLakeMasks()

        for ( lake_index, lake_mask_bounds ) in lake_masks.items():
            print( f"\nProcessing Lake {lake_index}: ", end='' )
            lake_mask_specs = LakeMaskProcessor.read_lake_mask( lake_index, lake_mask_bounds )
            lake_mask = lake_mask_specs.get( 'mask', None )
            [x0, x1, y0, y1] = lake_mask.xgeo.extent()
            locations = dataMgr.infer_tile_locations( lake_mask=lake_mask, legacy=True )
            for legacy_tile in locations:
                pos = get_centroid(legacy_tile)
                nrt_tile = TileLocator.get_tiles(pos[0], pos[0], pos[1], pos[1])[0]

                legacy_nodata, legacy_size = [], []
                nrt_nodata, nrt_size = [], []

                for day in range(*days):
                    legacy_data_file = f"{data_loc}/{legacy_tile}/MWP_{year}{day:03}_{legacy_tile}_2D2OT.tif"
                    nrt_data_file = f"{data_loc}/{nrt_tile}/{nrt_path}/MCDWD_L3_F2_NRT.A{year}{day:03}.{nrt_tile}.061.tif"

                    if not os.path.isfile(legacy_data_file):
                        print(f" LSKIP-{day}", end=''); sys.stdout.flush()
                    elif not os.path.isfile(nrt_data_file):
                        print(f" NSKIP-{day}", end=''); sys.stdout.flush()
                    else:
                        print(f" {day}", end=''); sys.stdout.flush()
                        legacy_data: xa.DataArray = read_subset( legacy_data_file, lake_mask, apply_mask )
                        nrt_data: xa.DataArray = read_subset( nrt_data_file, lake_mask, apply_mask )

                        legacy_nodata_mask: xa.DataArray = (legacy_data == 0)
                        nrt_nodata_mask: xa.DataArray = (nrt_data == 255)

                        legacy_ntot_mask: xa.DataArray = legacy_data.isin( [0, 2, 3] )
                        nrt_ntot_mask: xa.DataArray = nrt_data.isin( [1, 2, 3, 255] )

                        legacy_nodata_count = np.count_nonzero( legacy_nodata_mask.values )
                        nrt_nodata_count = np.count_nonzero( nrt_nodata_mask.values )

                        legacy_ntot_count = np.count_nonzero( legacy_ntot_mask.values )
                        nrt_ntot_count = np.count_nonzero( nrt_ntot_mask.values )

                        legacy_nodata.append( legacy_nodata_count * scale )
                        legacy_size.append( legacy_ntot_count * scale )
                        nrt_nodata.append( nrt_nodata_count * scale )
                        nrt_size.append( nrt_ntot_count * scale )

                if len(legacy_size) == 0:
                    print(f"\nSkipping tile {legacy_tile}: NO DATA")
                else:
                    tile_legacy_nodata = np.array(legacy_nodata).sum()
                    tile_legacy_size = np.array(legacy_size).sum()
                    tile_nrt_nodata = np.array(nrt_nodata).sum()
                    tile_nrt_size = np.array(nrt_size).sum()
                    output_file.write( f"{lake_index}, {tile_legacy_nodata:.2f}, {tile_legacy_size:.2f}, {tile_nrt_nodata:.2f}, {tile_nrt_size:.2f}, {x0:.4f}, {x1:.4f}, {y0:.4f}, {y1:.4f}\n" )
                    output_file.flush()

                tot_legacy_nodata.append(tile_legacy_nodata)
                tot_legacy_size.append(tile_legacy_size)
                tot_nrt_nodata.append(tile_nrt_nodata)
                tot_nrt_size.append(tile_nrt_size)

        result_legacy_nodata = np.array(tot_legacy_nodata).sum()
        result_legacy_size = np.array(tot_legacy_size).sum()
        result_nrt_nodata = np.array(tot_nrt_nodata).sum()
        result_nrt_size = np.array(tot_nrt_size).sum()
        result_legacy_pctn = result_legacy_nodata / result_legacy_size
        result_nrt_pctn = result_nrt_nodata / result_nrt_size
        print( f"\n\nRESULT: LEGACY: {result_legacy_pctn * 100:.2f}%, NRT: {result_nrt_pctn * 100:.2f}%, pct_diff: {pct_diff(result_legacy_pctn, result_nrt_pctn):.2f}")
        output_file.write( f"TOTAL, {result_legacy_pctn * 100:.2f}, {result_nrt_pctn * 100:.2f}, {pct_diff(result_legacy_pctn, result_nrt_pctn):.2f}\n")





