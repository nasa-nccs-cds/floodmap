from shapely.geometry import Polygon
from typing import Dict, List, Tuple, Optional

def intersects( p0: List[Tuple[float,float]], p1: List[Tuple[float,float]] ):
    poly1, poly2 = Polygon(p0), Polygon(p1)
    return poly1.intersects( poly2 )