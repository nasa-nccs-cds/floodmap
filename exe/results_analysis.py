from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import xarray as xa
import numpy as np
from floodmap.util.configuration import opSpecs
from floodmap.util.analysis import FloodmapProcessor

results_dir = opSpecs.get( 'results_dir' )
outliers = None # [ 26, 314, 333, 334, 336, 337  ]
fmp = FloodmapProcessor( results_dir )
means = fmp.get_means( outliers )