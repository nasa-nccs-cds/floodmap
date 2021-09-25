from typing import Tuple, Dict, Optional, Union, List
from floodmap.util.xrio import XRio
from multiprocessing import cpu_count, get_context, Pool
from functools import partial
import rioxarray as rio
from ..util.configuration import opSpecs
from datetime import datetime
import xarray as xr
import numpy as np
from ..util.logs import getLogger
import os, traceback, logging, atexit, csv

class nasa_dialect(csv.Dialect):
    delimiter = ','
    quotechar = '"'
    doublequote = True
    skipinitialspace = True
    lineterminator = '\n'
    quoting = csv.QUOTE_ALL

csv.register_dialect("nasa", nasa_dialect)

def write_result_report( lake_index, report: str ):
    results_dir = opSpecs.get('results_dir')
    file_path = f"{results_dir}/lake_{lake_index}_task_report.txt"
    with open( file_path, "a" ) as file:
        file.write( report )

def get_date_from_year( year: int ):
    result = datetime( year, 1, 1 )
    return np.datetime64(result)

class LakeMaskProcessor:

    def __init__( self ):
        self.logger = getLogger( True, logging.DEBUG )
        self.pool: Pool = None
        atexit.register( self.shutdown )

    @classmethod
    def getLakeMasks( cls ) -> Dict:
        from floodmap.util.configuration import opSpecs
        lakeMaskSpecs: Dict = opSpecs.get("lake_masks", None)
        data_dir: str = lakeMaskSpecs.get("basedir", None)
        data_roi: str = lakeMaskSpecs.get( "roi", None )
        lake_index: int = int( lakeMaskSpecs.get( "lake_index", 0 ) )
        files_spec: str = lakeMaskSpecs.get("file", "UNDEF" )
        lake_masks = {}
        if files_spec != "UNDEF":
            assert data_dir is not None, "Must define 'basedir' with 'file' parameter in 'lake_masks' config"
        if files_spec.endswith(".csv"):
            file_path = os.path.join(data_dir, files_spec )
            with open( file_path ) as csvfile:
                reader = csv.DictReader(csvfile, dialect="nasa")
                for row in reader:
                    lake_masks[ int(row['index']) ] = [ float(row['lon0']), float(row['lon1']), float(row['lat0']), float(row['lat1'])  ]
        elif files_spec.endswith(".tif"):
            lake_index_range = lakeMaskSpecs.get( "lake_index_range", (0,1000) )
            for lake_index in range(lake_index_range[0], lake_index_range[1] + 1):
                file_path = os.path.join( data_dir, files_spec.format(lake_index=lake_index) )
                if os.path.isfile(file_path):
                    lake_masks[lake_index] = file_path
                    print(f"  Processing Lake-{lake_index} using lake file: {file_path}")
                else:
                    print(f"Skipping Lake-{lake_index}, NO LAKE FILE")
        elif files_spec != "UNDEF":
            raise Exception( f"Unrecognized 'file' specification in 'lake_masks' config: '{files_spec}'")
        elif data_roi is not None:
            lake_masks[ lake_index ] = [ float(v) for v in data_roi.split(",") ]
        else:
            print( "No lakes configured in specs file." )
        return lake_masks

    # def download_floodmap_data(self, **kwargs):
    #     from .mwp import MWPDataManager
    #     dataMgr = MWPDataManager.instance()
    #     dataMgr.download_current_mpw_data( **kwargs )

    def process_lakes( self, **kwargs ):
        try:
            lake_masks = self.getLakeMasks()
            nproc = opSpecs.get( 'ncores', cpu_count() )
            lake_specs = list(lake_masks.items())
#            self.download_floodmap_data( lake_masks )
            self.logger.info( f"Processing Lakes: {list(lake_masks.keys())}" )
            with get_context("spawn").Pool(processes=nproc) as p:
                self.pool = p
                results = p.map( partial( LakeMaskProcessor.process_lake_mask, kwargs ), lake_specs )
            self.logger.info( f"Processes completed- exiting.\n\n Processed lakes: {list(filter(None, results))}")
        except Exception as err:
            self.logger.error(f"Exception: {err}")
            self.logger.error( traceback.format_exc() )

    @classmethod
    def read_lake_mask( cls, lake_index: int, lake_mask: Union[str,Tuple], **kwargs ) -> Dict:
        rv = dict( mask=None, roi=None, index=lake_index, **kwargs )
        if isinstance(lake_mask, str):
            lake_mask: xr.DataArray = rio.open_rasterio(lake_mask).astype(np.dtype('f4'))
            lake_mask.attrs.update( kwargs )
            lake_mask.name = f"Lake {lake_index} Mask"
            rv['mask'] = lake_mask
            rv['roi'] = lake_mask.xgeo.extent()
        else:
            rv['roi'] = lake_mask
        return rv

    @classmethod
    def process_lake_mask( cls, runSpecs: Dict, lake_mask_spec: Tuple[int, str]):
        from .lakeExtentMapping import WaterMapGenerator
        logger = getLogger(False, logging.DEBUG)
        ( lake_index, lake_mask ) = lake_mask_spec
        try:
            lake_mask = cls.read_lake_mask( lake_index, lake_mask, **runSpecs )
            waterMapGenerator = WaterMapGenerator()
            waterMapGenerator.generate_lake_water_map( **lake_mask )
            return lake_index
        except Exception as err:
            msg = f"Skipping lake {lake_index} due to error: {err}\n {traceback.format_exc()} "
            logger.error(msg); print(msg)
            write_result_report(lake_index, msg)

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


