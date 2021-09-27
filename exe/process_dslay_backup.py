import glob, os
from functools import partial
from multiprocessing import cpu_count, get_context, Pool
nproc = cpu_count()

def process_file( archive_dir: str, collection: str, dstr: str, hdfFilepath: str ) -> int:
    fName = os.path.basename(hdfFilepath)
    fdir = os.path.dirname(hdfFilepath)
    tile = fName.split('.')[2]
    outpath = f"{tile}/allData/{collection}/MCDWD_L3_F2_NRT/Recent"
    result_file = f"MCDWD_L3_F2_NRT.A{dstr}.{tile}.{collection:03}.tif"
    os.makedirs(f'{archive_dir}/{outpath}', exist_ok=True)
    product = f"HDF4_EOS:EOS_GRID:MCDWD_L3_NRT.A{dstr}.{tile}.{collection:03}.hdf:Grid_Water_Composite:'Flood 2-Day 250m'"
    result_path = f"{archive_dir}/{outpath}/{result_file}"
    command = f"cd {fdir}; gdal_translate {product} {result_path} -ot Byte -co 'COMPRESS=JPEG'"
    rv = os.system(command)
    print(f" [{rv}]-> {outpath}:  {result_file}")
    return rv

collection = 61
year = 2021
day_range = [252,253]
archive_dir = "/att/nobackup/tpmaxwel/data/MCDWD_NRT"

for day in day_range:
    dstr = f"{year}{day:03}"
    source_dir = f"/att/nobackup/dslaybac/MCDWD_NRT/MCDWD_L3_NRT_{dstr}"
    gfstr = f"{source_dir}/MCDWD_L3_NRT.A{dstr}.h*v*.{collection:03}.hdf"
    infiles = glob.glob( gfstr )
    print( f"Processing HDF files for [{dstr}]: '{gfstr}'" )
    print( f"Converting {len(infiles)} hdf files, saving to location: {archive_dir}:")
    processor = partial( process_file, archive_dir, collection, dstr )

    with get_context("spawn").Pool(processes=nproc) as p:
        results = p.map( processor, infiles )
        p.join()






