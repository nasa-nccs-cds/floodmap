import time, os, sys, pprint, logging, pathlib
from typing import List, Union, Dict
import numpy as np
from multiprocessing import Pool
from floodmap.xext.xgeo import XGeo
from ..util.logs import getLogger
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

class MWPDataManager(ConfigurableObject):
    BASE_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()

    def __init__(self, data_dir: str, data_source_url: str, **kwargs ):
        ConfigurableObject.__init__( self, **kwargs )
        self.data_dir = data_dir
        self.data_source_url = data_source_url
        self.logger = getLogger( False )

    def get_location_dir( self, location: str ) -> str:
        loc_dir = os.path.join( self.data_dir, location )
        if not os.path.exists(loc_dir): os.makedirs(loc_dir)
        return loc_dir

    def delete_if_empty( self, location: str  ):
        ldir = self.get_location_dir( location )
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

    def reload_damaged_files(self, location: str = "120W050N", **kwargs) -> List[str]:
        import wget
        start_day = self.getParameter( "start_day", **kwargs )
        end_day =   self.getParameter( "end_day",   **kwargs )
        years =     self.getParameter( "years",      **kwargs )
        year =      self.getParameter("year", **kwargs)
        product =   self.getParameter( "product",   **kwargs )
        location_dir = self.get_location_dir( location )
        files = []
        if years is None: years = year
        iYs = years if isinstance(years, list) else [years]
        for iY in iYs:
            for iFile in range(start_day+1,end_day+1):
                target_file = f"MWP_{iY}{iFile:03}_{location}_{product}.tif"
                target_file_path = os.path.join( location_dir, target_file )
                if self.test_if_damaged( target_file_path ):
                    target_url = self.data_source_url + f"/{location}/{iY}/{target_file}"
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

    def get_tile(self, location: str = "h19v08", **kwargs) -> List[str]:
        download =  self.getParameter( "download",  **kwargs )
        start_day = self.getParameter( "start_day", **kwargs )
        end_day =   self.getParameter( "end_day",   **kwargs )
        years =     self.getParameter( "years",   [ self.getParameter("year", **kwargs) ] )
        product =   self.getParameter( "product",   **kwargs )
        key =       self.getParameter("key", **kwargs)
        collection = self.getParameter("collection", **kwargs)
        location_dir = self.get_location_dir( location )
        hdf2nc = os.path.join( self.BASE_DIR, "bin", "h4tonccf_nc4" )
        self.logger.info( f" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% hdf2nc: {hdf2nc}  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% " )
        files = []
        for iY in list(years):
            for iFile in range(start_day+1,end_day+1):
                target_dir = f"{collection:02}/{product}/{iY}/{iFile:03}"
                target_file = f"{product}.A{iY}{iFile:03}.{location}.{collection:03}"
                target_file_path = "/".join( [ location_dir, "allData", target_dir, target_file] )
                if not os.path.exists( target_file_path+".nc" ):
                    if download:
                        target_url = "/".join( [self.data_source_url, target_dir, target_file+".hdf"] )
                        try:
                            download_script = f'wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=4 "{target_url}" --header "Authorization: Bearer {key}" -P {location_dir}'
                            self.logger.info(f"Downloading url {target_url} to file '{target_file_path}.hdf'")
                            stream = os.popen( download_script )
                            self.logger.info( stream.read() )

                            self.logger.info(f"Transforming '{target_file_path}.hdf' ---> '{target_file_path}.nc'")
                            transform_script = f"{hdf2nc} '{target_file_path}.hdf' '{target_file_path}.nc'"
                            stream = os.popen( transform_script )
                            self.logger.info( stream.read() )

                            os.remove(f"{target_file_path}.hdf")
                            files.append( target_file_path+".nc"  )
                        except Exception as err:
                            self.logger.error( f"     ---> Can't access {target_url}: {err}")
                else:
                    self.logger.info(f" Array[{len(files)}] -> Time[{iFile}]: {target_file_path}.nc")
                    files.append( target_file_path+".nc" )
        return files

    def get_array_data(self, files: List[str], merge=False ) ->  Union[xr.DataArray,List[xr.DataArray]]:
        arrays = XGeo.loadRasterFiles( files, region = self.getParameter("bbox") )
        return self.time_merge(arrays) if merge else arrays

    def get_tile_data(self, location: str = "120W050N", merge=False, **kwargs) -> Union[xr.DataArray,List[xr.DataArray]]:
        files = self.get_tile(location, **kwargs)
        return self.get_array_data( files, merge )

    def get_global_locations( self ) -> List:
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

    def remove_empty_directories(self, nProcesses: int = 8):
        locations = dataMgr.get_global_locations()
        with Pool(nProcesses) as p:
            p.map(dataMgr.delete_if_empty, locations, nProcesses)

    def _segment(self, strList: List[str], nSegments ):
        seg_length = int( round( len( strList )/nSegments ) )
        return [strList[x:x + seg_length] for x in range(0, len(strList), seg_length)]

    def download_tiles(self, nProcesses: int = 8 ):
        location = self.parms.get( 'location' )
        locations = dataMgr.get_global_locations( ) if location is None else [ location ]
        with Pool(nProcesses) as p:
            p.map(dataMgr.get_tile, locations, nProcesses)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print( "Usage: >> python -m floodmap.surfaceMapping.mwp <dataDirectory>\n       Downloads all MWP tiles to the data directory")
    else:
        dataMgr = MWPDataManager( sys.argv[1], "https://floodmap.modaps.eosdis.nasa.gov/Products" )
        dataMgr.setDefaults( product = "1D1OS", download = True, year = 2018, start_day = 1, end_day = 365, location='120W050N' )
        dataMgr.download_tiles( 10 )
        dataMgr.remove_empty_directories(10)



