import os, wget, sys, pprint, logging, glob
from multiprocessing import cpu_count, get_context, Pool
from functools import partial
import geopandas as gpd
from typing import List, Union, Tuple, Dict
from collections import OrderedDict
import time, numpy as np
from datetime import datetime, timedelta
from floodmap.util.configuration import opSpecs
from floodmap.util.xgeo import XGeo
from ..util.logs import getLogger, getLogFile
import xarray as xr
pp = pprint.PrettyPrinter(depth=4).pprint
from floodmap.util.configuration import ConfigurableObject

def getStreamLogger( level ):
    logger = logging.getLogger (__name__ )
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def get_tile_dir(data_dir, tile: str) -> str:
    logger = getLogger(False)
    logger.log( f" ---> get_tile_dir: data_dir={data_dir}, tile={tile}")
    loc_dir = os.path.join( data_dir, tile )
    if not os.path.exists(loc_dir): os.makedirs(loc_dir)
    return loc_dir

def download( target_url: str, result_dir: str, token: str ):
    logger = getLogger(False)
    (lname,log_file) = getLogFile( False )
    print( f"Downloading Tile: {target_url}", flush=True )
    cmd = f'wget -e robots=off -m -np -R .html,.tmp -nH --no-check-certificate -a {log_file} --cut-dirs=4 "{target_url}" --header "Authorization: Bearer {token}" -P "{result_dir}"'
    stream = os.popen(cmd)
    output = stream.read()
    logger.info(f"Downloading url {target_url} to dir {result_dir}: result = {output}")

def local_file_path( product, path_template, collection, data_dir, tile, year, day ):
    location_dir = get_tile_dir(data_dir, tile)
    path = path_template.format( collection=collection, product=product )
    (iD,iY) = (day,year) if (day > 0) else (365+day,year-1)
    timestr = f"{iY}{iD:03}"
    target_file = f"{product}.A{timestr}.{tile}.{collection:03}.tif"
    return os.path.join( location_dir, path, target_file )

def has_tile_data( product, path_template, collection, data_dir, tile ) -> Tuple[str,bool]:
    location_dir = get_tile_dir(data_dir, tile)
    path = path_template.format( collection=collection, product=product )
    target_file = f"{product}.A*.{tile}.{collection:03}.tif"
    files: List[str] = glob.glob( os.path.join( location_dir, path, target_file ) )
    return ( tile, ( len( files ) > 0 ) )

def access_sample_tile( product, path_template, collection, token, data_dir, data_source_url, tile) -> Tuple[str,bool]:
    logger = getLogger(False)
    tt = datetime.now().timetuple()
    day_of_year = tt.tm_yday
    this_year = tt.tm_year
    day = day_of_year - 3
    location_dir = get_tile_dir(data_dir, tile)
    path = path_template.format( collection=collection, product=product )
    (iD,iY) = (day,this_year) if (day > 0) else (365+day,this_year-1)
    timestr = f"{iY}{iD:03}"
    target_file = f"{product}.A{timestr}.{tile}.{collection:03}.tif"
    target_file_path = os.path.join( location_dir, path, target_file )
    dtime = np.datetime64( datetime.strptime( f"{timestr}", '%Y%j').date() )
    logger.info(f" Accessing MPW Tile[{day}] for {tile}:{dtime}")
    if os.path.exists(target_file_path):
        logger.info(f" Local NRT file exists: {target_file_path}")
        return (tile, True)
    else:
        target_url = data_source_url + f"/{path}/{target_file}"
        download( target_url, location_dir, token )
        return (tile, os.path.exists( target_file_path ))

class MWPDataManager(ConfigurableObject):
    _instance: "MWPDataManager" = None

    def __init__(self, data_dir: str, data_source_url: str, **kwargs ) :
        ConfigurableObject.__init__( self, **kwargs )
        self.data_dir = data_dir
        self.data_source_url = data_source_url
        self.logger = getLogger( False )
        self._valid_tiles: List[str] = None

    @classmethod
    def instance( cls, **kwargs ) -> "MWPDataManager":
        if cls._instance is None:
            results_dir = opSpecs.get('results_dir')
            source_spec = opSpecs.get('source')
            data_url = source_spec.get('url')
            day0, year0 = cls.today()
            cls._instance = MWPDataManager(results_dir, data_url)
            cls._instance.setDefaults()
            cls._instance.parms['product'] = source_spec.get('product')
            cls._instance.parms['token'] = source_spec.get('token')
            cls._instance.parms['day'] = source_spec.get('day',day0)
            cls._instance.parms['year'] = source_spec.get('year',year0)
            cls._instance.parms['path'] = source_spec.get('path')
            cls._instance.parms['collection'] = source_spec.get('collection')
            cls._instance.parms['max_history_length'] = source_spec.get( 'max_history_length', 300 )
            cls._instance.parms.update( kwargs )
        if cls._instance._valid_tiles is None:
            cls._instance._valid_tiles = kwargs.get('tiles')
        return cls._instance

    def set_day(self, day: int ):
        self.parms['day'] = day

    @classmethod
    def target_date(cls) -> List[int]:
        return [ cls.instance().parms[pid] for pid in ('year','day') ]

    def get_target_date(self) -> str:
        daystr = f"{self.parms['year']}-{self.parms['day']}"
        return datetime.strptime( daystr, "%Y-%j").strftime("%m-%d-%Y")

    def list_required_tiles(self, **kwargs) -> List[str]:
        from .tiles import TileLocator
        lake_mask = kwargs.get( 'lake_mask', None )
        roi_bounds = kwargs.get('roi', None)
        if lake_mask is not None:
            required_tiles = TileLocator.infer_tiles_xa( lake_mask, **kwargs )
        elif roi_bounds is not None:
            if isinstance( roi_bounds, gpd.GeoSeries ):    rbnds = roi_bounds.geometry.boundary.bounds.values[0]
            else:                                          rbnds = roi_bounds
            required_tiles = TileLocator.get_tiles( *rbnds, **kwargs )
            self.logger.info(f"Processing roi bounds (xmin, xmax, ymin, ymax): {rbnds}, tiles = {required_tiles}")
        else:
            raise Exception( "Must supply either source.tile, roi, or lake masks in order to locate region")
        for tile in required_tiles:
            if tile not in self._valid_tiles:
                raise Exception( f"Lake requires nonexistent tile: {tile}")
        return required_tiles

    def download_mpw_data( self, **kwargs ) -> List[str]:
        self.logger.info( "downloading mpw data")
        tiles = kwargs.get( 'tiles', self.get_valid_tiles() )
        for tile in tiles:
            self.get_tile( tile, **kwargs )
        return tiles

    def get_valid_tiles(self, **kwargs) -> List[str]:
        if self._valid_tiles is None:
            product = self.getParameter("product", **kwargs)
            path_template = self.getParameter("path", **kwargs)
            collection = self.getParameter("collection", **kwargs)
            all_tiles = [ has_tile_data( product, path_template, collection, self.data_dir, tile ) for tile in self.global_tile_list() ]
            if not True in [valid for (tile, valid) in all_tiles]:
                parallel = kwargs.get('parallel', True)
                token = self.getParameter("token", **kwargs)
                processor = partial( access_sample_tile, product, path_template, collection, token, self.data_dir, self.data_source_url )
                if parallel:
                    self.logger.log(f" ---> get_valid_tiles[PARALLEL]: all_tiles={all_tiles}")
                    with get_context("spawn").Pool( processes=cpu_count() ) as p:
                        tiles = [ tile for (tile, valid) in all_tiles]
                        all_tiles = p.map( processor, tiles )
                else:
                    self.logger.info(f"  ---> get_valid_tiles: all_tiles={all_tiles}")
                    all_tiles = [ processor(tile) for (tile, valid) in all_tiles ]
            self._valid_tiles = [ tile for (tile, valid) in all_tiles if valid ]
            print( f"Got {len(self._valid_tiles)} valid Tiles")
        return self._valid_tiles

    @classmethod
    def today(cls) -> Tuple[int,int]:
        today = datetime.now()
        day_of_year = today.timetuple().tm_yday
        return ( day_of_year, today.year )

    def get_dstr(self, **kargs ) -> str:
        return f"{self.parms['year']}{self.parms['day']:03}"



    def delete_if_empty( self, tile: str  ):
        ldir = get_tile_dir(self.data_dir, tile)
        try: os.rmdir( ldir )
        except OSError: pass

    def getTimeslice(self, array: xr.DataArray, index: int, dtype: np.dtype = np.float ) -> xr.DataArray:
        input_array =  array[index].astype(dtype)
        self.transferMetadata( array, input_array )
        return input_array

    def test_if_damaged( self, file_path ):
        import rioxarray
        try:
            result: xr.DataArray = rioxarray.open_rasterio(file_path)
            return False
        except Exception as err:
            return True

    def reload_damaged_files(self, tile: str = "120W050N", **kwargs) -> List[str]:
        start_day = self.getParameter( "start_day", **kwargs )
        end_day =   self.getParameter( "end_day",   **kwargs )
        years =     self.getParameter( "years",      **kwargs )
        year =      self.getParameter("year", **kwargs)
        product =   self.getParameter( "product",   **kwargs )
        location_dir = get_tile_dir(self.data_dir, tile)
        files = []
        if years is None: years = year
        iYs = years if isinstance(years, list) else [years]
        for iY in iYs:
            for iFile in range(start_day+1,end_day+1):
                target_file = f"MWP_{iY}{iFile:03}_{tile}_{product}.tif"
                target_file_path = os.path.join( location_dir, target_file )
                if self.test_if_damaged( target_file_path ):
                    target_url = self.data_source_url + f"/{tile}/{iY}/{target_file}"
                    try:
                        wget.download( target_url, target_file_path )
                        self.logger.info(f"Downloading url {target_url} to file {target_file_path}")
                        files.append( target_file_path )
                    except Exception as err:
                        self.logger.error( f"     ---> Can't access {target_url}: {err}")
                else:
                    self.logger.info(f" Array[{len(files)}] -> Time[{iFile}]: {target_file_path}")
                    files.append( target_file_path )
        self.logger.info(" Downloaded replacement files:")
        pp( files )
        return files

    def global_tile_list(self):
        locs = []
        for hi in range(0,36):
            for vi in range(0, 18):
                locs.append( f"h{hi:02d}v{vi:02d}")
        return locs

    def get_date_from_filepath(self, filename: str) -> datetime:
        fname = os.path.basename(filename)
        dateId = fname.split(".")[1]
        rv = datetime.strptime(dateId[1:], '%Y%j')
        return rv

    def delete_old_files(self, **kwargs ):
        max_history_length = self.getParameter("max_history_length", **kwargs)
        if max_history_length > 0:
            path_template =  self.getParameter( "path", **kwargs)
            product = self.getParameter("product", **kwargs)
            collection= self.getParameter( "collection", **kwargs )
            path = path_template.format(collection=collection, product=product)
            for tile in self.global_tile_list():
                location_dir = get_tile_dir(self.data_dir, tile)
                target_dir = os.path.join(location_dir, path )
                if os.path.exists( target_dir ):
                    files = glob.glob(f"{target_dir}/{product}.A*.{tile}.{collection:03}.tif")
                    for file in files:
                        dt: timedelta = datetime.now() - self.get_date_from_filepath(file)
                        if dt.days > max_history_length: os.remove( file )

    def get_tile(self, tile, **kwargs) -> OrderedDict:
        from floodmap.util.configuration import opSpecs
        day_of_year = datetime.now().timetuple().tm_yday
        water_maps_opspec = opSpecs.get('water_map', {})
        history_length = self.getParameter( 'history_length', 8 )
        bin_size = water_maps_opspec.get( 'bin_size', 8 )
        this_day = self.getParameter( "day", day_of_year )
        this_year = self.getParameter("year", **kwargs )
        product =   self.getParameter( "product",   **kwargs )
        path_template =  self.getParameter( "path", **kwargs)
        collection= self.getParameter( "collection", **kwargs )
        token=        self.getParameter( "token", **kwargs )
        tile_dir = get_tile_dir(self.data_dir, tile)
        files = OrderedDict()
        days = range( this_day-history_length, this_day )
        path = path_template.format( collection=collection, product=product )
        dstrs, tstrs = [], []
        for day in days:
            (iD,iY) = (day,this_year) if (day > 0) else (365+day,this_year-1)
            timestr = f"{iY}{iD:03}"
            target_file = f"{product}.A{timestr}.{tile}.{collection:03}.tif"
            target_file_path = os.path.join( tile_dir, path, target_file )
            dtime = np.datetime64( datetime.strptime( f"{timestr}", '%Y%j').date() )
            self.logger.info(f" Accessing MPW Tile[{day}] for {tile}:{dtime}")
            if not os.path.exists( target_file_path ):
                if ( this_day - day ) <= bin_size:
                    self.logger.info(f" Local NRT file does not exist: {target_file_path}")
                    target_url = self.data_source_url + f"/{path}/{target_file}"
                    download( target_url, tile_dir, token )
                    if os.path.exists(target_file_path):
                        self.logger.info(f" Downloaded NRT file: {target_file_path}")
                        files[dtime] = target_file_path
                        dstrs.append(timestr)
                    else:
                        self.logger.info( f" Can't access NRT file: {target_file_path}" )
            else:
                self.logger.info(f" Array[{len(files)}] -> Time[{iY}:{iD}]: {target_file_path}")
                files[dtime] = target_file_path
                tstrs.append(timestr)
        if len(dstrs): self.logger.info( f"Downloading MWP data for dates, day range = [{this_day-history_length, this_day}]: {dstrs}" )
        if len(tstrs): self.logger.info( f"Reading MWP data for dates: {tstrs}" )
        return files

    def get_array_data(self, files: List[str], merge=False ) ->  Union[xr.DataArray,List[xr.DataArray]]:
        arrays = XGeo.loadRasterFiles( files, region = self.getParameter("bbox") )
        return self.time_merge(arrays) if merge else arrays

    # def get_tile_data(self, tile: str, merge=False, **kwargs) -> Union[xr.DataArray,List[xr.DataArray]]:
    #     files = self.get_tile(tile, **kwargs)
    #     return self.get_array_data( files, merge )

    @staticmethod
    def extent( transform: Union[List, Tuple], shape: Union[List, Tuple], origin: str ):
        (sy,sx) = (shape[1],shape[2]) if len(shape) == 3 else (shape[0],shape[1])
        ext =  [transform[2], transform[2] + sx * transform[0] + sy * transform[1],
                transform[5], transform[5] + sx * transform[3] + sy * transform[4]]
        if origin == "upper": ( ext[2], ext[3] ) = ( ext[3], ext[2] )
        return ext

    def get_global_tile_list(self) -> List:
        global_locs = []
        for ix in range(10,181,10):
            for xhemi in [ "E", "W" ]:
                for iy in range(10,71,10):
                    for yhemi in ["N", "S"]:
                        global_locs.append( f"{ix:03d}{xhemi}{iy:03d}{yhemi}")
        for ix in range(10,181,10):
            for xhemi in [ "E", "W" ]:
                global_locs.append( f"{ix:03d}{xhemi}000S")
        for iy in range(10, 71, 10):
            for yhemi in ["N", "S"]:
                global_locs.append(f"000E{iy:03d}{yhemi}")
        return global_locs

    def _segment(self, strList: List[str], nSegments ):
        seg_length = int( round( len( strList )/nSegments ) )
        return [strList[x:x + seg_length] for x in range(0, len(strList), seg_length)]





