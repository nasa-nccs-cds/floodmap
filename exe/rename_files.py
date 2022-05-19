import os, glob
from pathlib import Path


fdir = "/adapt/nobackup/people/zwwillia/MODIS_water/model_outputs/BirketCompare/simple/2021/h09v05_v2"
files = glob.glob( f"{fdir}/*-Simple.tif")
for file in files:
    toks = Path(file).stem.split("-")
    basename = "-".join( toks[:4] )
    new_file = f"{fdir}/{basename}-simple.tif"
    print( file )
    print( "   ---> " + new_file )
    os.rename( file, new_file )