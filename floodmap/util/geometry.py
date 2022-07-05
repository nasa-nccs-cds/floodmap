from shapely.geometry import Polygon
from floodmap.util.logs import getLogger
from typing import Dict, List, Tuple, Optional

def intersects( p0: List[Tuple[float,float]], p1: List[Tuple[float,float]] ):
    poly1, poly2 = Polygon(p0), Polygon(p1)
    rv =  poly1.intersects( poly2 )
    return rv