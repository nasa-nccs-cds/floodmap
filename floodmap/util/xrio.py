from typing import List, Union, Optional
import pandas as pd
from .crs import CRS
from floodmap.util.xext import XExtension
from geopandas import GeoDataFrame
import os, ntpath
import numpy as np
from shapely.geometry import mapping
from floodmap.util.configuration import argfilter
import rioxarray, traceback
import rasterio
from floodmap.util.logs import getLogger
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
            if not os.path.exists(filename):
                msg = f"File {filename} does not exist, skipping..."
                print( msg )
                logger.info( msg )
                return None
            raster: xr.DataArray = rioxarray.open_rasterio( filename, **oargs )
            band = kwargs.pop( 'band', -1 )
            if band >= 0:
                raster = raster.isel( band=band, drop=True )
            raster.encoding = dict( dtype = str(np.dtype('f4')) )
            if mask is None:
                result = raster
            elif isinstance( mask, list ):
                transform = raster.xgeo.getTransform()
                if transform[1] > 10:
                    raster.attrs['crs'] = CRS.get_utm_proj4( mask[0], mask[2] )
                    raster = raster.xgeo.gdal_reproject()
                tile_bounds = TileLocator.get_bounds(raster)
                invert_y = (tile_bounds[2] > tile_bounds[3])
                if iFile == 0: logger.info(f"Subsetting array with bounds {tile_bounds} by xbounds = {mask[:2]}, ybounds = {mask[2:]}")
                result = raster.xrio.subset( mask[:2], mask[2:], invert_y )
            elif isinstance( mask, GeoDataFrame ):
                result = raster.xrio.clip( mask, **kwargs )
            else:
                raise Exception( f"Unrecognized mask type: {mask.__class__.__name__}")
            return result
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
        result = self._obj.sel(**sel_args)
        base_transform = [ *self._obj.transform ]
        base_transform[0] = xbounds[0]
        base_transform[3] = ybounds[0]
        result.attrs['transform'] = base_transform
        return result

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
    def load( cls, filePaths: Union[ str, List[str] ], **kwargs ) -> Optional[ xr.DataArray ]:
        if isinstance( filePaths, str ): filePaths = [ filePaths ]
        result: Optional[ xr.DataArray ] = None
        for iF, file in enumerate(filePaths):
            data_array: xr.DataArray = cls.open( iF, file, **kwargs )
            if data_array is not None:
                time_values = np.array([ cls.get_date_from_filename(os.path.basename(file)) ], dtype='datetime64[ns]')
                data_array = data_array.squeeze(drop=True)
                data_array = data_array.expand_dims( { 'time': time_values }, 0 )
                result = data_array if result is None else cls.concat([result, data_array])
        return result

    @classmethod
    def get_date_from_filename(cls, filename: str):
        from datetime import datetime
        basename = filename[:-4] if filename.endswith(".tif") else filename
        if "." in basename:
            toks = basename.split(".")[1]
            result = datetime.strptime(toks[1:],'%Y%j').date()
        else:
            toks = basename.split( "_")[1]
            result = datetime.strptime(toks, '%Y' ).date()
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
        result: xr.DataArray =  xr.DataArray( result_data, dims=array0.dims, coords=coords, attrs=array0.attrs )
        if hasattr( array0, 'spatial_ref' ): result['spatial_ref'] = array0.spatial_ref
#        print( f"Concat arrays along dim {array0.dims[0]}, input array dims = {array0.dims}, shape = {array0.shape}, Result array dims = {result.dims}, shape = {result.shape}")
        return result

    @classmethod
    def mergable(cls, arrays: List[xr.DataArray]) -> bool:
        for array in arrays:
            if array.shape != arrays[0].shape: return False
        return True
