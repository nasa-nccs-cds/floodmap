from typing import Tuple, Dict
from floodmap.util.xrio import XRio
from multiprocessing import cpu_count, get_context, Pool
from functools import partial
import rioxarray as rio
from ..util.configuration import opSpecs
from datetime import datetime
import xarray as xr
import numpy as np
from ..util.logs import getLogger
import os, traceback, logging, atexit

def write_result_report( lake_index, report: str ):
    results_dir = opSpecs.get('results_dir')
    file_path = f"{results_dir}/lake_{lake_index}_task_report.txt"
    with open( file_path, "a" ) as file:
        file.write( report )

def get_date_from_year( year: int ):
    result = datetime( year, 1, 1 )
    return np.datetime64(result)

def process_lake_mask( lakeMaskSpecs: Dict, runSpecs: Dict, lake_mask_item: Tuple[int,str] ):
    from .lakeExtentMapping import WaterMapGenerator
    logger = getLogger( False, logging.DEBUG )
    (lake_index, lake_mask_file) = lake_mask_item
    try:
        lake_mask: xr.DataArray = rio.open_rasterio(lake_mask_file).astype(np.dtype('f4'))
        lake_mask.attrs.update(lakeMaskSpecs)
        lake_mask.name = f"Lake {lake_index} Mask"
        waterMapGenerator = WaterMapGenerator( {'lake_index': lake_index,  **opSpecs._defaults} )
        waterMapGenerator.generate_lake_water_map(lake_index, lake_mask, **runSpecs)
        logger.info(f"Completed processing lake {lake_index}")
        return lake_index
    except Exception as err:
        msg = f"Skipping lake {lake_index} due to error: {err} "
        logger.error(msg); print( msg )
        logger.error( traceback.format_exc() )
        write_result_report(lake_index, traceback.format_exc())

class LakeMaskProcessor:

    def __init__( self ):
        self.logger = getLogger( True, logging.DEBUG )
        self.pool: Pool = None
        atexit.register( self.shutdown )

    @classmethod
    def getLakeMasks(cls, opSpecs: Dict ) -> Dict:
        reproject_inputs = opSpecs.get( 'reproject', False )
        lakeMaskSpecs = opSpecs.get("lake_masks", None)
        data_dir = lakeMaskSpecs["basedir"]
        lake_index_range = lakeMaskSpecs.get( "lake_index_range", (0,1000) )
        files_spec = lakeMaskSpecs["file"]
        lake_masks = {}
        for lake_index in range(lake_index_range[0], lake_index_range[1] + 1):
            file_path = os.path.join( data_dir, files_spec.format(lake_index=lake_index) )
            if os.path.isfile(file_path):
                lake_masks[lake_index] = cls.convert(file_path) if reproject_inputs else file_path
                print(f"  Processing Lake-{lake_index} using lale file: {file_path}")
            else:
                print(f"Skipping Lake-{lake_index}, NO LAKE FILE")
        return lake_masks

    def process_lakes( self, **kwargs ):
        try:
            opSpecs.set( 'reproject', kwargs.get('reproject_inputs',False) )
            lake_masks = self.getLakeMasks( opSpecs )
            lakeMaskSpecs = opSpecs.get("lake_masks", None)
            nproc = opSpecs.get( 'ncores', cpu_count() )
            items = list(lake_masks.items())
            self.logger.info( f"Processing Lakes: {list(lake_masks.keys())}" )
            with get_context("spawn").Pool(processes=nproc) as p:
                self.pool = p
                results = p.map( partial( process_lake_mask, lakeMaskSpecs, kwargs ), items )

            self.logger.info( f"Processes completed- exiting.\n\n Processed lakes: {results}")
        except Exception as err:
            self.logger.error(f"Exception: {err}")
            self.logger.error( traceback.format_exc() )

    @classmethod
    def convert(cls, src_file: str, overwrite = True ) -> str:
        dest_file = src_file[:-4] + ".geo.tif"
        if overwrite or not os.path.exists(dest_file):
            XRio.convert( src_file, dest_file )
        return dest_file

    def fuzzy_where( cond: xr.DataArray, x, y, join="left" ) -> xr.DataArray:
        from xarray.core import duck_array_ops
        return xr.apply_ufunc( duck_array_ops.where, cond, x, y, join=join, dataset_join=join, dask="allowed" )

    def set_spatial_precision( self, array: xr.DataArray, precision: int ) -> xr.DataArray:
        if precision is None: return array
        sdims = [ array.dims[-2], array.dims[-1] ]
        rounded_coords = { dim: array.coords[dim].round( precision ) for dim in sdims }
        return array.assign_coords( rounded_coords )

    def shutdown(self):
        if self.pool is not None:
            self.pool.terminate()


