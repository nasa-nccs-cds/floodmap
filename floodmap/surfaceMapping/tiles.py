import xarray as xa
import geopandas as gpd
from math import floor, ceil
from ..util.logs import getLogger
from typing import List, Union, Tuple, Optional

class TileLocator:

    @classmethod
    def floor10(cls, fval: float) -> int:
        return abs( int(floor(fval / 10.0)) * 10 )

    @classmethod
    def ceil10(cls, fval: float) -> int:
        return abs( int(ceil(fval / 10.0)) * 10 )

    @classmethod
    def unwrap(cls, coord: float) -> float:
        return coord if coord < 180 else coord - 360

    @classmethod
    def hc( cls, coord: float ) -> str:
        nc = coord if coord < 180 else coord - 360
        rv = int( ( nc + 180 ) // 10 )
        return f"h{rv:02d}"

    @classmethod
    def vc( cls, coord: float ) -> str:
        rv = int( (90 - coord) // 10 )
        return f"v{rv:02d}"

    @classmethod
    def lon_label(cls, lon: float ) -> str:
        ulon = cls.unwrap( lon )
        if ulon < 0: return f"{cls.floor10(ulon):03d}W"
        else:        return f"{cls.floor10(ulon):03d}E"

    @classmethod
    def lat_label(cls, lat: float ) -> str:
        if lat > 0: return f"{cls.ceil10(lat):03d}N"
        else:       return f"{cls.ceil10(lat):03d}S"

    @classmethod
    def infer_tiles_xa( cls, array: xa.DataArray, **kwargs ) -> List[str]:
        return cls.get_tiles( *array.xgeo.extent(), **kwargs )

    @classmethod
    def infer_tiles_gpd( cls, series: gpd.GeoSeries ) -> List[str]:
        [xmin, ymin, xmax, ymax] = series.geometry.boundary.bounds.values[0]
        return cls.get_tiles( xmin, xmax, ymin, ymax )


    @classmethod
    def get_tiles_legacy( cls, xmin, xmax, ymin, ymax ) -> List[str]:
        xvals = { cls.lon_label( xmin ), cls.lon_label( xmax ) }
        yvals = { cls.lat_label( ymin ), cls.lat_label( ymax ) }
        results = [ f"{xval}{yval}" for xval in xvals for yval in yvals ]
        return results

    @classmethod
    def get_tiles_nrt( cls, xmin, xmax, ymin, ymax ) -> List[str]:
        xvals = { cls.hc( xmin ), cls.hc( xmax ) }
        yvals = { cls.vc( ymin ), cls.vc( ymax ) }
        results = [ f"{xval}{yval}" for xval in xvals for yval in yvals ]
        return results

    @classmethod
    def get_tiles( cls, xmin, xmax, ymin, ymax, **kwargs ) -> List[str]:
        legacy = kwargs.get('legacy', False)
        if legacy:  return TileLocator.get_tiles_legacy( xmin, xmax, ymin, ymax )
        else:       return TileLocator.get_tiles( xmin, xmax, ymin, ymax )

    @classmethod
    def get_bounds(cls, array: xa.DataArray ) -> List:
        x_coord = array.coords[array.dims[-1]].values
        y_coord = array.coords[array.dims[-2]].values
        return [ x_coord[0], x_coord[-1], y_coord[0], y_coord[-1] ]