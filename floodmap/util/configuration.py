from shapely.geometry import Point
import xarray as xr
import string, random, json
import pandas as pd
import os
from typing import List, Dict

def argfilter( args: Dict, **kwargs ) -> Dict:
    return { key: args.get(key,value) for key,value in kwargs.items() }

def sanitize( array: xr.DataArray ):
    for key, value in array.attrs.items():
        if key == "cmap" and not isinstance(value,str):
            array.attrs[key] = json.dumps(value)
        else:
            array.attrs[key] = value
    return array

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

class Region:

    def __init__(self, origin: List[int], size: int ):
        self.origin: List[int] = origin
        self.size: int = size
        self.bounds: List[int] = [ origin[0] + size, origin[1] + size ]






