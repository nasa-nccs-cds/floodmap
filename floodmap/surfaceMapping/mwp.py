import os, wget, sys, pprint, logging, glob
import traceback
from multiprocessing import cpu_count, get_context, Pool
import rioxarray, subprocess
from functools import partial
import geopandas as gpd
from typing import List, Union, Tuple, Dict, Optional
from collections import OrderedDict
import time, numpy as np
from datetime import datetime, timedelta, date
from floodmap.util.configuration import opSpecs
from floodmap.util.xgeo import XGeo
from ..util.logs import getLogger, getLogFile, getLogFileObject
import xarray as xr
pp = pprint.PrettyPrinter(depth=4).pprint
from floodmap.util.configuration import ConfigurableObject

def s2b( sval: str ): return sval.lower().startswith('t')

def parse_collection( collection: Union[int,str]):
    try:
        return f"{int(collection):03d}"
    except Exception as err:
        return collection

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
    loc_dir = os.path.join( data_dir, tile )
    if not os.path.exists(loc_dir): os.makedirs(loc_dir)
    return loc_dir

def download( target_url: str, result_dir: str, token: str ):
    logger = getLogger(False)
    (lname,log_file) = getLogFile( False )
    print( f"Downloading Tile: {target_url} -> {result_dir}", flush=True )
    cmd = f'wget -e robots=off -m -np -R .html,.tmp -nH --no-check-certificate -a {log_file} --cut-dirs=4 "{target_url}" --header "Authorization: Bearer {token}" -P "{result_dir}"'
    logger.info(f"Using download command: '{cmd}'")
    proc = subprocess.Popen(cmd, shell=True, stdout=getLogFileObject(False), stderr=subprocess.STDOUT, bufsize=-1 )
    logger.info(f"Downloading url {target_url} to dir {result_dir}")
    proc.wait()
    logger.info(f"  **FINISHED Downloading url {target_url} **** ")

def has_tile_data( product, path_template, file_template, collection, data_dir, tile, year ) -> Tuple[ str, Optional[List[Tuple[float,float]]] ]:
    logger = getLogger(False)
    tt = datetime.now().timetuple()
    day_of_year = tt.tm_yday
    day = day_of_year - 3
    location_dir = get_tile_dir(data_dir, tile)
    path = path_template.format( collection=collection, product=product, year=year, tile=tile )
    target_file = file_template.format( collection=collection, product=product, year=year, tile=tile, day=day )
    glob_str = os.path.join( location_dir, path, target_file )
    files: List[str] = glob.glob( glob_str )
    nfiles = len(files)
    roi = None if (nfiles==0) else get_roi( files[0] )
    if nfiles > 0:
        logger.info( f" --> has_tile_data: glob_str='{glob_str}', #files={nfiles}")
    else:
        logger.info(f" --> NO data for tile {tile}: glob_str='{glob_str}'")
    return ( tile, roi )

def get_roi( target_file_path: str ) -> Optional[List[Tuple[float,float]]]:
    if os.path.exists( target_file_path ):
        raster: xr.DataArray = rioxarray.open_rasterio(target_file_path).squeeze().xgeo.gdal_reproject()
        if ('band' in raster.dims): raster = raster.isel(band=0, drop=True)
        [ yc, xc ] = [ raster.coords[c].values for c in raster.dims ]
        return [ (xc[0],yc[0]), (xc[0],yc[-1]), (xc[-1],yc[-1]), (xc[-1],yc[0]) ]

def access_sample_tile( product, path_template, file_template, collection, token, data_dir, data_source_url, day, year, tile ) -> Tuple[str,List[Tuple[float,float]]]:
    logger = getLogger(False)
    location_dir = get_tile_dir(data_dir, tile)
    path = path_template.format( collection=collection, product=product, year=year, tile=tile )
    (iD,iY) = (day,year) if (day > 0) else (365+day,year-1)
    timestr = f"{iY}{iD:03}"
    target_file = file_template.format( collection=collection, product=product, year=year, tile=tile, day=day )
    target_file_path = os.path.join( location_dir, path, target_file )
    dtime = np.datetime64( datetime.strptime( f"{timestr}", '%Y%j').date() )
    if os.path.exists(target_file_path):
        logger.info(f" Local NRT file exists: {target_file_path}")
    else:
        if data_source_url.startswith("file:/"):
            data_file_path = data_source_url[5:] + f"/{path}/{target_file}"
            files_exist = os.path.exists(data_file_path)
            if files_exist and (data_file_path != target_file_path):
                logger.info(f" Creating symlink: {target_file_path} -> {data_file_path} ")
                os.makedirs( os.path.dirname(target_file_path), exist_ok=True )
                os.symlink(data_file_path, target_file_path)
            logger.info(f"TILE {tile}: files_exist= {files_exist}, Source_file_path= {data_file_path}")
        else:
            print( f"Local file does not exist (downloading): {target_file_path}")
            target_url = data_source_url + f"/{path}/{target_file}"
            download( target_url, location_dir, token )
    roi = get_roi( target_file_path )
    return (tile, roi )

class MWPDataManager(ConfigurableObject):
    _instance: "MWPDataManager" = None
    default_file_template = "{product}.A{year}{day:03d}.{tile}.{collection}.tif"

    def __init__(self, data_dir: str, data_source_url: str, **kwargs ) :
        ConfigurableObject.__init__( self, **kwargs )
        self.data_dir = data_dir
        self.data_source_url = data_source_url
        self.logger = getLogger( False )
        self._valid_tiles: Dict[str, List[Tuple[float, float]]] = None

    @classmethod
    def instance( cls, **kwargs ) -> "MWPDataManager":
        if cls._instance is None:
            results_dir = opSpecs.get('results_dir')
            data_dir = opSpecs.get('data_dir',results_dir)
            op_range = opSpecs.get('op_range', [] )
            source_spec = opSpecs.get('source')
            data_url = source_spec.get('url')
            today, this_year = cls.today()
            history_length = source_spec.get('history_length', 30, **kwargs)
            download_length = source_spec.get('download_length', history_length, **kwargs)
            day0 = source_spec.get('day', today, **kwargs)
            year0 = source_spec.get('year', this_year, **kwargs)
            default_day_range = [day0 - history_length, day0] if (len(op_range)==0) else op_range[:2]
            cls._instance = MWPDataManager(data_dir, data_url)
            cls._instance.setDefaults()
            cls._instance.parms['product'] = source_spec.get('product')
            cls._instance.parms['token'] = source_spec.get('token')
            cls._instance.parms['parallel'] = s2b( source_spec.get( 'parallel', 'True' ) )
            cls._instance.parms['year'] = year0
            cls._instance.parms['day'] = day0
            cls._instance.parms['op_range'] = op_range
            cls._instance.parms['history_length'] = history_length
            cls._instance.parms['download_length'] = download_length
            cls._instance.parms['day_range'] = source_spec.get('day_range', default_day_range )
            cls._instance.logger.info( f"DR: Setting day range: {cls._instance.parms['day_range']}, default={default_day_range}, today={today}, kwargs={kwargs}")
            cls._instance.parms['path'] = source_spec.get('path')
            cls._instance.parms['file'] = source_spec.get( 'file', cls.default_file_template )
            cls._instance.parms['collection'] = source_spec.get('collection')
            cls._instance.parms['max_history_length'] = source_spec.get( 'max_history_length', 300 )
            cls._instance.parms.update( kwargs )
        return cls._instance

    def target_date(self) -> List[int]:
        day_range = self.parms['day_range']
        days = range(day_range[0], day_range[-1] + 1)
        year = self.parms['year']
        return [ year, days[-1] ]

    def local_file_path(self, product, path_template, collection, data_dir, tile, year, day):
        location_dir = get_tile_dir(data_dir, tile)
        path = path_template.format(collection=collection, product=product, year=year, tile=tile)
        (iD, iY) = (day, year) if (day > 0) else (365 + day, year - 1)
        timestr = f"{iY}{iD:03}"
        target_file = self.default_file_template
        return os.path.join(location_dir, path, target_file)

    def get_target_date(self) -> str:
        day_range = self.parms['day_range']
        days = range(day_range[0], day_range[-1] + 1)
        daystr = f"{self.parms['year']}-{days[-1]}"
        return datetime.strptime( daystr, "%Y-%j").strftime("%m-%d-%Y")

    def list_required_tiles(self, **kwargs) -> Optional[List[str]]:
        from .tiles import TileLocator
        lake_mask = kwargs.get( 'lake_mask', None )
        roi_bounds = kwargs.get('roi', None)
        valid_tiles = self.get_valid_tiles( **kwargs )
        if lake_mask is not None:
            required_tiles = TileLocator.infer_tiles_xa( lake_mask, tiles=valid_tiles, **kwargs )
        elif roi_bounds is not None:
            if isinstance( roi_bounds, gpd.GeoSeries ):    rbnds = roi_bounds.geometry.boundary.bounds.values[0]
            else:                                          rbnds = roi_bounds
            required_tiles = TileLocator.get_tiles( *rbnds, tiles=valid_tiles, **kwargs )
            self.logger.info(f"Processing roi bounds (xmin, xmax, ymin, ymax): {rbnds}, tiles = {required_tiles}")
        else:
            raise Exception( "Must supply either source.tile, roi, or lake masks in order to locate region")
        for tile in required_tiles:
            if tile not in valid_tiles:
                self.logger.info( f"Required tile {tile} is not valid, required_tiles = {required_tiles}, valid_tiles = {valid_tiles}")
                return None
        return required_tiles

    def download_mpw_data( self, **kwargs ) -> List[str]:
        self.logger.info( "downloading mpw data")
        current_tiles_only = opSpecs.get('current_tiles_only', False)
        if not current_tiles_only: kwargs['current_lakes'] = None
        print( "Downloading mpw data" )
        tiles = kwargs.get( 'tiles', self.get_valid_tiles(**kwargs).keys() )
        for tile in tiles:
            self.get_tile( tile, **kwargs )
        return tiles

    def get_day_range( self, **kwargs ):
        return self.parms['day_range']

    def filter_current_tiles(self, all_tiles: List[Tuple[ str, Optional[List[Tuple[float,float]]]]], current_lakess: Optional[Dict[int, Union[str, List[float]]]]):
        rois = [ ]
        current_tiles = []
        return current_tiles

    def get_valid_tiles(self, **kwargs) -> Dict[str,List[Tuple[float,float]]]:
        logger = getLogger(False)
        try:
            if self._valid_tiles is None:
                this_day = datetime.now().timetuple().tm_yday
                current_lakes: Optional[Dict[int, Union[str, List[float]]]] = kwargs.get('current_lakes')
                product = self.getParameter("product", **kwargs)
                path_template = self.getParameter( "path", **kwargs)
                file_template = self.getParameter( "file", **kwargs )
                parallel = opSpecs.get( 'parallel', False )
                collection = self.getParameter("collection", **kwargs)
                history_length = self.getParameter('history_length', 30, **kwargs)
                year = int( self.getParameter("year", datetime.now().timetuple().tm_year, **kwargs) )
                day_range = self.getParameter("day_range", [ this_day-history_length, this_day ], **kwargs)
                possible_tiles = self.global_tile_list()
                all_tiles = [ has_tile_data( product, path_template, file_template, collection, self.data_dir, tile, year ) for tile in possible_tiles ]
                # if current_lakes is not None:
                #     all_tiles = self.filter_current_tiles( all_tiles, current_lakes )
                logger.info(f" **get_valid_tiles(parallel={parallel}): all_tiles={all_tiles}")
                days = range( int(day_range[0]), int(day_range[-1]) + 1)
                if not True in [ (valid!=None) for (tile, valid) in all_tiles]:
                    token = self.getParameter("token", **kwargs)
                    processor = partial( access_sample_tile, product, path_template, file_template, collection, token, self.data_dir, self.data_source_url, days[0], year )
                    if parallel:
                        if type(parallel) == bool: parallel = "spawn"
                        with get_context(parallel).Pool( processes=cpu_count() ) as p:
                            tiles = [ tile for (tile, roi) in all_tiles]
                            logger.info(f" ---> process[PARALLEL]: tiles={tiles}")
                            all_tiles = p.map( processor, tiles )
                    else:
                        all_tiles = [ processor(tile) for (tile, roi) in all_tiles ]
                self._valid_tiles = { tile: roi for (tile, roi) in all_tiles if (roi is not None) }
                logger.info( f"Got {len(self._valid_tiles)} valid Tiles:")
                for tile,roi in self._valid_tiles.items():
                    logger.info(f" ** {tile}: {roi}")
        except Exception as err:
            logger.error( f"Unable to get valid tiles: {err}")
            logger.error( traceback.format_exc() )
            self._valid_tiles = {}
        return self._valid_tiles



    @classmethod
    def today(cls) -> Tuple[int,int]:
        today = datetime.now()
        day_of_year = today.timetuple().tm_yday
        return ( day_of_year, today.year )

    def get_dstr(self, **kargs ) -> str:
        day_range = self.parms['day_range']
        days = range(day_range[0], day_range[-1] + 1)
        return f"{self.parms['year']}{days[-1]:03}"

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
            for tile in self.global_tile_list():
                path = path_template.format(collection=collection, product=product, year="*", tile=tile)
                location_dir = get_tile_dir(self.data_dir, tile)
                target_dir = os.path.join(location_dir, path )
                if os.path.exists( target_dir ):
                    files = glob.glob(f"{target_dir}/{product}.A*.{tile}.*.tif")
                    for file in files:
                        dt: timedelta = datetime.now() - self.get_date_from_filepath(file)
                        if dt.days > max_history_length: os.remove( file )

    def get_tile(self, tile, **kwargs) -> Dict[date,str]:
        day_range = self.parms['day_range']
        iyear = self.parms['year']
        product =   self.getParameter( "product",   **kwargs )
        file_template = self.getParameter("file",  self.default_file_template, **kwargs)
        path_template =  self.getParameter( "path", **kwargs)
        collection= self.getParameter( "collection", **kwargs )
        download_length = self.getParameter("download_length", **kwargs)
        token=        self.getParameter( "token", **kwargs )
        tile_dir = get_tile_dir(self.data_dir, tile)
        files = OrderedDict()
        dstrs, tstrs = [], []
        days = range( day_range[0], day_range[-1]+1 )
        nday = len( days )
        for idx, iday in enumerate(days):
            (day,year) = (iday,iyear) if (iday > 0) else (365+iday,iyear-1)
            path = path_template.format( collection=collection, product=product, year=year, tile=tile )
            data_file = file_template.format( collection=collection, product=product, year=year, day=day, tile=tile )
            target_file = data_file # file_template.format( collection=collection, product=product, year=year, day=day, tile=tile )
            target_file_path = os.path.join( tile_dir, path, target_file )
            timestr = f"{year}{day:03}"
            dtime: date = datetime.strptime( timestr, '%Y%j').date()
            if not os.path.exists( target_file_path ):
                if self.data_source_url.startswith("file:/"):
                    data_file_path = self.data_source_url[5:] + f"/{path}/{data_file}"
                    if os.path.exists( data_file_path ):
                        self.logger.info(f" Creating symlink: {target_file_path} -> {data_file_path} ")
                        os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                        os.symlink( data_file_path, target_file_path )
                else:
                    if (nday-idx) <= download_length:
                        print(f"Local tile does not exist (downloading): {target_file_path}")
                        data_url = self.data_source_url + f"/{path}/{data_file}"
                        download( data_url, tile_dir, token )
                if os.path.exists(target_file_path):
                    self.logger.info(f" Downloaded NRT file: {target_file_path}")
                    files[dtime] = target_file_path
                    dstrs.append(timestr)
                else:
                    self.logger.info( f" Can't access NRT file: {target_file_path}" )
            else:
                self.logger.info(f" Array[{len(files)}] -> Time[{year}:{day}]: {target_file_path}")
                files[dtime] = target_file_path
                tstrs.append(timestr)
        if len(dstrs): self.logger.info( f"Downloading MWP data for dates, day range = [{day_range}]: {dstrs}" )
        if len(tstrs): self.logger.info( f"Reading MWP data for dates: {tstrs}, nfiles = {len(files)}" )
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





