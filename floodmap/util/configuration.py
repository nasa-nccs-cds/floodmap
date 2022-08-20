import xarray
from osgeo import osr
from shapely.geometry import Point
import xarray as xr
import string, random, json
import pandas as pd
import yaml, sys, os
from typing import List, Dict

def argfilter( args: Dict, **kwargs ) -> Dict:
    return { key: args.get(key,value) for key,value in kwargs.items() }

def sanitize_ds( dset: xr.Dataset, squeeze=False ):
    variables = { vid:sanitize(v,squeeze) for (vid,v) in dset.data_vars.items() }
    return xr.Dataset( variables, dset.coords, dset.attrs )

def sanitize( array: xr.DataArray, squeeze=False ):
    for key, value in array.attrs.items():
        if key == "cmap" and not isinstance(value,str):
            array.attrs[key] = json.dumps(value)
        elif isinstance(value, osr.SpatialReference ):
            array.attrs[key] = str( value.ExportToPrettyWkt() )
        else:
            array.attrs[key] = value
    # if hasattr( array, 'spatial_ref' ):
    #     if isinstance( array.spatial_ref, osr.SpatialReference ):
    #         array['spatial_ref'] = array.spatial_ref.ExportToPrettyWkt()
    #     elif isinstance( array.spatial_ref, xarray.DataArray ):
    #         array['spatial_ref'] = array.spatial_ref.attrs.get('crs_wkt','')
    return array.squeeze( drop=True ) if squeeze else array

class Region:

    def __init__(self, origin: List[int], size: int ):
        self.origin: List[int] = origin
        self.size: int = size
        self.bounds: List[int] = [ origin[0] + size, origin[1] + size ]

class ConfigurableObject:

    def __init__(self, **kwargs):
        self.parms = { **kwargs }

    def getParameter(self, name: str, default=None, **kwargs ):
        return kwargs.get( name, self.parms.get(name,default) )

    def setDefaults( self, **kwargs ):
        self.parms.update( kwargs )

    @classmethod
    def randomId( cls, length: int ) -> str:
        sample = string.ascii_lowercase+string.digits+string.ascii_uppercase
        return ''.join(random.choice(sample) for i in range(length))

    @classmethod
    def transferMetadata( cls, ref_array: xr.DataArray, new_array: xr.DataArray ):
        new_attrs = { key: value for key, value in ref_array.attrs.items() if  key not in new_array.attrs }
        return new_array.assign_attrs( new_attrs )

    @classmethod
    def parseLocation( cls, location: str ) -> Point:
        lonVal, latStr, latVal = None, None, None
        try:
            if "E" in location:
                coords = location.split("E")
                lonVal = int(coords[0])
                latStr = coords[1]
            elif "W" in location:
                coords = location.split("W")
                lonVal = -int(coords[0])
                latStr = coords[1]
            if "N" in latStr:
                latVal = int(latStr[:-1])
            elif "S" in latStr:
                latVal = -int(latStr[:-1])
            assert lonVal and latVal, "Missing NSEW"
        except Exception as err:
            raise Exception( f"Format error parsing location {location}: {err}")

        return Point( lonVal, latVal )

    def frames_merge( self, data_arrays: List[xr.DataArray] ) -> xr.DataArray:
        frame_names = [ da.name for da in data_arrays ]
        merge_coord = pd.Index( frame_names, name="frames" )
        return xr.concat( objs=data_arrays, dim=merge_coord )

    @classmethod
    def time_merge( cls, data_arrays: List[xr.DataArray], **kwargs ) -> xr.DataArray:
        time_axis = kwargs.get('time',None)
        frame_indices = range( len(data_arrays) )
        merge_coord = pd.Index( frame_indices, name=kwargs.get("dim","time") ) if time_axis is None else time_axis
        result: xr.DataArray =  xr.concat( data_arrays, dim=merge_coord )
        return result # .assign_coords( {'frames': frame_names } )

class OpSpecs:

    def __init__(self):
        if len(sys.argv) < 2:
            print( "Must pass config file path as first argument.")
            sys.exit(1)
        opspec_file = sys.argv[1]
        with open(opspec_file) as f:
            self._specs = yaml.load(f, Loader=yaml.FullLoader)
            self._defaults = self._specs.get( "defaults", {} )

    def get( self, key: str , default = None ):
        result = self._defaults.get( key, default )
        assert result is not None, f"Required parameter {key} missing from specs file"
        return result

    def set( self, key: str , value: str ):
        self._defaults[key] = value

    def setmod( self, module: str, key: str , value: str ):
        self._defaults[module][key] = value

opSpecs = OpSpecs()






