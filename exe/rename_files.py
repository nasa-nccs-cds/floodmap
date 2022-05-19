import os, glob
from pathlib import Path


fdir = "/adapt/nobackup/people/zwwillia/MODIS_water/model_outputs/BirketCompare/simple/2021/h09v05_v2"
files = glob.glob( f"{fdir}/*-Simple.tif")
for file in files:
    toks = Path(file).stem.split("-")
    basename = "-".join( toks[:4] )
    print( f"{fdir}/{basename}-simple.tif")
#    os.rename('guru99.txt','career.guru99.txt')