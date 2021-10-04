from typing import List, Union, Tuple, Dict, Optional
from ..xext.xrio import XRio
from multiprocessing import cpu_count, get_context, Pool, Event
from functools import partial
from ..util.configuration import opSpecs
from datetime import datetime
import xarray as xr
import numpy as np
from ..util.logs import getLogger
import os, time, collections, traceback, logging, atexit

def write_result_report( lake_index, report: str ):
    results_dir = opSpecs.get('results_dir')
    file_path = f"{results_dir}/lake_{lake_index}_task_report.txt"
    with open( file_path, "a" ) as file:
        file.write( report )

def get_date_from_year( year: int ):
    result = datetime( year, 1, 1 )
    return np.datetime64(result)

def process_lake_mask( lakeMaskSpecs: Dict, runSpecs: Dict, lake_mask_files: Tuple[int,Dict] ):
    from .lakeExtentMapping import WaterMapGenerator
    lake_index, sorted_file_paths = lake_mask_files
    logger = getLogger( False, logging.DEBUG )
    print( lake_index )
    try:
        time_values = np.array([ get_date_from_year(year) for year in sorted_file_paths.keys()], dtype='datetime64[ns]')
        yearly_lake_masks: xr.DataArray = XRio.load(list(sorted_file_paths.values()), band=0, index=time_values)
        yearly_lake_masks.attrs.update(lakeMaskSpecs)
        yearly_lake_masks.name = f"Lake {lake_index} Mask"
        nx, ny = yearly_lake_masks.shape[-1], yearly_lake_masks.shape[-2]
        waterMapGenerator = WaterMapGenerator( {'lake_index': lake_index,  **opSpecs._defaults} )
        waterMapGenerator.process_yearly_lake_masks( lake_index, yearly_lake_masks, **runSpecs )
        logger.info(f"Completed processing lake {lake_index}")
        return lake_index
    except Exception:
        logger.error(f"Skipping lake {lake_index} due to errors ")
        logger.error( traceback.format_exc() )
        write_result_report(lake_index, traceback.format_exc())

class LakeMaskProcessor:

    def __init__( self ):
        self.logger = getLogger( True, logging.DEBUG )
        self.pool: Pool = None
        atexit.register( self.shutdown )

    def process_lakes( self, reproject_inputs, **kwargs ):
        try:
            year_range = opSpecs.get('year_range')
            lakeMaskSpecs = opSpecs.get( "lake_masks", None )
            data_dir = lakeMaskSpecs["basedir"]
            lake_index_range = lakeMaskSpecs["lake_index_range"]
            directorys_spec = lakeMaskSpecs["subdir"]
            files_spec = lakeMaskSpecs["file"]
            lake_masks = {}
            lake_indices = []
            for year in range( int(year_range[0]), int(year_range[1]) + 1 ):
                year_dir = os.path.join( data_dir, directorys_spec.format( year=year ) )
                _lake_indices = lake_indices if year > year_range[0] else range(lake_index_range[0], lake_index_range[1] + 1)
                for lake_index in _lake_indices:
                    file_path = os.path.join(year_dir, files_spec.format( year=year, lake_index=lake_index ) )
                    if year == year_range[0]:
                        if os.path.isfile( file_path ):
                            lake_masks[lake_index] = collections.OrderedDict( )
                            lake_masks[lake_index][year] = self.convert( file_path ) if reproject_inputs else file_path
                            lake_indices.append( lake_index )
                    elif os.path.isfile( file_path ):
                        lake_masks[lake_index][year]= self.convert( file_path ) if reproject_inputs else file_path

            nproc = opSpecs.get( 'ncores', cpu_count() )
            items = list(lake_masks.items())
            self.logger.info( f"Processing Lakes: {list(lake_masks.keys())}")
            with get_context("spawn").Pool(processes=nproc) as p:
                self.pool = p
                results = p.map( partial( process_lake_mask, lakeMaskSpecs, kwargs ), items )

            self.logger.info( f"Processes completed- exiting.\n\n Processed lakes: {results}")
        except Exception as err:
            self.logger.error(f"Exception: {err}")
            self.logger.error( traceback.format_exc() )

    def convert(self, src_file: str, overwrite = True ) -> str:
        dest_file = src_file[:-4] + ".geo.tif"
        if overwrite or not os.path.exists(dest_file):
            self.logger.info( f"Saving converted input to {dest_file}")
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


