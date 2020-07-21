from typing import List, Union, Tuple, Optional
import pandas as pd
from .xextension import XExtension
from geopandas import GeoDataFrame
import os, warnings, ntpath
import numpy as np
from shapely.geometry import box, mapping
from ..util.configuration import argfilter
import rioxarray, traceback
import rasterio, logging
from ..util.logs import getLogger
from rasterio.warp import calculate_default_transform, reproject, Resampling
import xarray as xr

@xr.register_dataarray_accessor('xrio')
class XRio(XExtension):
    """  This is an extension for xarray to provide an interface to rasterio capabilities """

    def __init__(self, xarray_obj: xr.DataArray):
        XExtension.__init__( self, xarray_obj )

    @classmethod
    def open( cls, iFile: int, filename: str, **kwargs )-> Optional[xr.DataArray]:
        from ..surfaceMapping.tiles import TileLocator
        logger = getLogger(False)
        mask = kwargs.pop("mask", None)
        kill_zombies = kwargs.pop( "kill_zombies", False )
        oargs = argfilter( kwargs, parse_coordinates = None, chunks = None, cache = None, lock = None )
        try:
            result: xr.DataArray = rioxarray.open_rasterio( filename, **oargs ).astype( np.dtype('f4') )
            band = kwargs.pop( 'band', -1 )
            if band >= 0:
                result = result.isel( band=band, drop=True )
            result.encoding = dict( dtype = str(np.dtype('f4')) )
            if mask is None: return result
            elif isinstance( mask, list ):
                tile_bounds = TileLocator.get_bounds(result)
                invert_y = (tile_bounds[2] > tile_bounds[3])
                if iFile == 0: logger.info(f"Subsetting array with bounds {tile_bounds} by xbounds = {mask[:2]}, ybounds = {mask[2:]}")
                return result.xrio.subset( mask[:2], mask[2:], invert_y )
            elif isinstance( mask, GeoDataFrame ):
                return result.xrio.clip( mask, **kwargs )
            else:
                raise Exception( f"Unrecognized mask type: {mask.__class__.__name__}")
        except Exception as err:
            logger.error( f"XRio Error opening file {filename}: {err}")
            logger.error( traceback.format_exc() )
            if kill_zombies:
                print(f"Deleting erroneous file")
                os.remove( filename )
            return None

    def subset(self, xbounds: List, ybounds: List, invert_y: bool  )-> xr.DataArray:
        xbounds.sort(), ybounds.sort( reverse = invert_y )
        sel_args = { self._obj.dims[-1]: slice(*xbounds), self._obj.dims[-2]: slice(*ybounds) }
        return self._obj.sel(**sel_args)

    def clip(self, geodf: GeoDataFrame, **kwargs )-> xr.DataArray:
        cargs = argfilter( kwargs, all_touched = True, drop = True )
        mask_value = int( kwargs.pop( 'mask_value', 255  ) )
        self._obj.rio.set_nodata(mask_value)
        result = self._obj.rio.clip( geodf.geometry.apply(mapping), geodf.crs, **cargs )
        result.attrs['mask_value'] = mask_value
        result.encoding = self._obj.encoding
        return result

    @classmethod
    def print_array_dims( cls, filePaths: Union[ str, List[str] ], **kwargs ):
        if isinstance( filePaths, str ): filePaths = [ filePaths ]
        logger = getLogger( False )
        result: xr.DataArray = None
        logger.info(f" ARRAY DIMS " )
        for iF, file in enumerate(filePaths):
            data_array: xr.DataArray = cls.open( iF, file, **kwargs )
            if data_array is not None:
                time_values = np.array([ cls.get_date_from_filename(os.path.basename(file)) ], dtype='datetime64[ns]')
                data_array = data_array.expand_dims( { 'time': time_values }, 0 )
                logger.info( f"  ** Array[{iF}:{ntpath.basename(file)}]-> shape = {data_array.shape}")
        return result

    @classmethod
    def load( cls, filePaths: Union[ str, List[str] ], **kwargs ) -> Union[ List[xr.DataArray], xr.DataArray ]:
        if isinstance( filePaths, str ): filePaths = [ filePaths ]
        result: xr.DataArray = None
        for iF, file in enumerate(filePaths):
            data_array: xr.DataArray = cls.open( iF, file, **kwargs )
            if data_array is not None:
                time_values = np.array([ cls.get_date_from_filename(os.path.basename(file)) ], dtype='datetime64[ns]')
                data_array = data_array.expand_dims( { 'time': time_values }, 0 )
                result = data_array if result is None else cls.concat([result, data_array])
        return result

    @classmethod
    def get_date_from_filename(cls, filename: str):
        from datetime import datetime
        basename = filename[:-4] if filename.endswith(".tif") else filename
        toks = basename.split( "_")[1]
        try:    result = datetime.strptime(toks, '%Y%j').date()
        except: result = datetime.strptime(toks, '%Y' ).date()
        return np.datetime64(result)

    @classmethod
    def convert(self, source_file_path: str, dest_file_path: str, espg = 4236 ):
        dst_crs = f'EPSG:{espg}'

        with rasterio.open( source_file_path ) as src:
            print( f"PROFILE: {src.profile}" )
            src_crs = ''.join(src.crs.wkt.split())
            print(f" ---> CRS: {src_crs}")
            transform, width, height = calculate_default_transform( src_crs, dst_crs, src.width, src.height, *src.bounds )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(dest_file_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)

    @classmethod
    def merge( cls, data_arrays: List[xr.DataArray], **kwargs ) -> xr.DataArray:
        new_axis_name = kwargs.get('axis','time')
        index_mask = kwargs.get('index_mask', None )
        index = kwargs.get('index', None )
        new_axis_values = range( len(data_arrays) ) if index is None else index if index_mask is None else np.extract( index_mask, index )
        merge_coord = pd.Index( new_axis_values, name=new_axis_name )
        result: xr.DataArray =  xr.concat( data_arrays, merge_coord, compat='broadcast_equals', join='outer' )
        return result

    @classmethod
    def concat( cls, data_arrays: List[xr.DataArray] ) -> xr.DataArray:
        array0, dim0 = data_arrays[0], data_arrays[0].dims[0]
        result_data = np.concatenate( [ da.values for da in data_arrays ], axis=0 )
        coords = { key:data_arrays[0].coords[key] for key in array0.dims[1:] }
        coords[ dim0 ] = xr.concat( [ da.coords[dim0] for da in data_arrays ], dim=array0.coords[dim0].dims[0] )
        result: xr.DataArray =  xr.DataArray( result_data, dims=array0.dims, coords=coords )
#        print( f"Concat arrays along dim {array0.dims[0]}, input array dims = {array0.dims}, shape = {array0.shape}, Result array dims = {result.dims}, shape = {result.shape}")
        return result

    @classmethod
    def mergable(cls, arrays: List[xr.DataArray]) -> bool:
        for array in arrays:
            if array.shape != arrays[0].shape: return False
        return True
