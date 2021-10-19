from typing import List, Union, Tuple, Dict, Optional
from ..xext.xrio import XRio
from multiprocessing import cpu_count, get_context, Pool, Event
from functools import partial
from ..util.configuration import opSpecs
from datetime import datetime
import xarray as xr
import numpy as np
import rioxarray as rio
from ..util.logs import getLogger
import os, time, collections, traceback, logging, atexit, csv

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

    @classmethod
    def read_lake_mask( cls, lake_index: int, lake_mask: Union[str,Tuple], **kwargs ) -> Dict:
        rv = dict( mask=None, roi=None, index=lake_index, **kwargs )
        if isinstance(lake_mask, str):
            lake_mask: xr.DataArray = rio.open_rasterio(lake_mask).astype(np.dtype('f4'))
            lake_mask.attrs.update( kwargs )
            lake_mask.attrs['mask'] = 3
            lake_mask.attrs['water'] = 1
            lake_mask.name = f"Lake {lake_index} Mask"
            rv['mask'] = lake_mask
            rv['roi'] = lake_mask.xgeo.extent()
        else:
            rv['roi'] = lake_mask
        return rv

    @classmethod
    def process_lake_mask(cls, runSpecs: Dict, lake_info: Tuple[int, str]):
        from .lakeExtentMapping import WaterMapGenerator
        logger = getLogger(False, logging.DEBUG)
        (lake_index, lake_mask_bounds) = lake_info
        try:
            lake_mask_specs = cls.read_lake_mask(lake_index, lake_mask_bounds, **runSpecs)
            waterMapGenerator = WaterMapGenerator({'lake_index': lake_index, **opSpecs._defaults})
            waterMapGenerator.process_yearly_lake_masks(lake_index, lake_mask_specs['mask'], **runSpecs)
            return lake_index
        except Exception as err:
            msg = f"Skipping lake {lake_index} due to error: {err}\n {traceback.format_exc()} "
            logger.error(msg);
            print(msg)
            write_result_report(lake_index, msg)

    @classmethod
    def getLakeMasks( cls ) -> Dict[int,str]:
        from floodmap_legacy.util.configuration import opSpecs
        lakeMaskSpecs: Dict = opSpecs.get("lake_masks", None)
        data_dir: str = lakeMaskSpecs.get("basedir", None)
        data_roi: str = lakeMaskSpecs.get( "roi", None )
        lake_index: int = int( lakeMaskSpecs.get( "lake_index", -1 ) )
        lake_indices: List[int] = [ int(lake_indx) for lake_indx in lakeMaskSpecs.get("lake_indices", [] ) ]
        lake_index_range: Tuple[int, int] = lakeMaskSpecs.get("lake_index_range", (0, 1000))
        files_spec: str = lakeMaskSpecs.get("file", "UNDEF" )
        lake_masks = {}
        if files_spec != "UNDEF":
            assert data_dir is not None, "Must define 'basedir' with 'file' parameter in 'lake_masks' config"
        if files_spec.endswith(".csv"):
            file_path = os.path.join(data_dir, files_spec )
            with open( file_path ) as csvfile:
                reader = csv.DictReader(csvfile, dialect="nasa")
                for row in reader:
                    index = int(row['index'])
                    if lake_index in [-1,index]:
                        lake_masks[ index ] = [ float(row['lon0']), float(row['lon1']), float(row['lat0']), float(row['lat1'])  ]
        elif files_spec.endswith(".tif"):
            if len(lake_indices) == 0:
                if ( lake_index >= 0 ): lake_indices = [ lake_index ]
                else: lake_indices = list( range(lake_index_range[0], lake_index_range[1] + 1) )
            for iLake in lake_indices:
                file_path = os.path.join( data_dir, files_spec.format(lake_index=iLake) )
                if os.path.isfile(file_path):
                    lake_masks[iLake] = file_path
                    print(f"  Processing Lake-{iLake} using lake file: {file_path}")
                else:
                    print(f"Skipping Lake-{iLake}, NO LAKE FILE")
        elif files_spec != "UNDEF":
            raise Exception( f"Unrecognized 'file' specification in 'lake_masks' config: '{files_spec}'")
        elif data_roi is not None:
            index = 0 if lake_index is None else lake_index
            lake_masks[ index ] = [ float(v) for v in data_roi.split(",") ]
        else:
            print( "No lakes configured in specs file." )
        return lake_masks

    def process_lakes1( self, **kwargs ):
        try:
            reproject_inputs = False
            lake_masks = self.getLakeMasks()
            year_range = opSpecs.get('year_range')
            lakeMaskSpecs = opSpecs.get( "lake_masks", None )
            data_dir = lakeMaskSpecs["basedir"]
            lake_index_range = lakeMaskSpecs.get("lake_index_range",None)
            lake_indices = lakeMaskSpecs.get("lake_indices",[])
            directorys_spec = lakeMaskSpecs["subdir"]
            files_spec = lakeMaskSpecs["file"]
            print( f" {files_spec}, {lakeMaskSpecs} ")
            lake_masks = {}
            lake_indices = set()
            for year in range( int(year_range[0]), int(year_range[1]) + 1 ):
                year_dir = os.path.join( data_dir, directorys_spec.format( year=year ) )
                _lake_indices = lake_indices if year > year_range[0] else range(lake_index_range[0], lake_index_range[1] + 1)
                for lake_index in _lake_indices:
                    file_path = os.path.join(year_dir, files_spec.format( year=year, lake_index=lake_index ) )
                    if os.path.isfile( file_path ):
                        if lake_index not in lake_masks: lake_masks[lake_index] = collections.OrderedDict( )
                        lake_masks[lake_index][year] = self.convert( file_path ) if reproject_inputs else file_path
                        lake_indices.add( lake_index )
                    else:
                        print( f"Lake file[{lake_index}] does not exist: {file_path}")

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

    def process_lakes( self, **kwargs ):
        try:
            lake_masks = self.getLakeMasks()
            parallel = kwargs.get( 'parallel', True )
            nproc = opSpecs.get( 'ncores', cpu_count() )
            lake_specs = list(lake_masks.items())
            self.logger.info( f"Processing Lakes: {list(lake_masks.keys())}" )
            if parallel:
                with get_context("spawn").Pool(processes=nproc) as p:
                    self.pool = p
                    results = p.map( partial( self.process_lake_mask, kwargs ), lake_specs )
            else:
                results = [ self.process_lake_mask(  kwargs, lake_spec ) for lake_spec in lake_specs ]
            self.logger.info( f"Processes completed- exiting.\n\n Processed lakes: {list(filter(None, results))}")
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


