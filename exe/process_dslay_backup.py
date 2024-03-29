import glob, os, time
from functools import partial
from multiprocessing import cpu_count, get_context, Pool, freeze_support
nproc = cpu_count()

collection = 61
archive_dir = "/explore/nobackup/projects/ilab/projects/Birkett/MOD44W/data"

def process_file( archive_dir: str, collection: str, hdfFilepath: str ) -> int:
    fName = os.path.basename(hdfFilepath)
    fdir = os.path.dirname(hdfFilepath)
    ftoks = fName.split('.')
    tile, dstr = ftoks[2], ftoks[1][1:]
    outpath =   f"{tile}/allData/{collection}/MCDWD_L3_F2_NRT/Recent"
    result_file = f"MCDWD_L3_F2_NRT.A{dstr}.{tile}.{collection:03d}.tif"
    os.makedirs(f'{archive_dir}/{outpath}', exist_ok=True)
    product = f"HDF4_EOS:EOS_GRID:MCDWD_L3_NRT.A{dstr}.{tile}.{collection:03d}"
    if ftoks[4] != "hdf": product = f"{product}.{ftoks[4]}"
    result_path = f"{archive_dir}/{outpath}/{result_file}"
    if os.path.isfile( result_path ):
        rv = 0
        print( f" *** SKIPPING EXISTING {outpath}: {result_file}" )
    else:
        command = f"cd {fdir}; gdal_translate {product}.hdf:Grid_Water_Composite:'Flood 2-Day 250m' {result_path} -q -co 'COMPRESS=DEFLATE'"
        rv = os.system(command)
        print( f" *** [{rv}]->      {outpath}:  {result_file}" )
    return rv

if __name__ == '__main__':
    freeze_support()
    t0 = time.time()
    dstr = f"2021*"
    tile = "h*v*" # "h*v*" "h20v09"
    source_dir = f"/explore/nobackup/people/dslaybac/MCDWD_NRT/MCDWD_L3_NRT_{dstr}"
    gfstr = f"{source_dir}/MCDWD_L3_NRT.A{dstr}.{tile}.{collection:03d}.hdf"
    infiles = glob.glob( gfstr )
    print( f"Processing HDF files for [{dstr}]: '{gfstr}'" )
    print( f"Converting {len(infiles)} hdf files, saving to location: {archive_dir}:")
    processor = partial( process_file, archive_dir, collection )

    with get_context("spawn").Pool(processes=nproc) as p:
        results = p.map( processor, infiles )
    p.join()

    print( f"Completed processing files {dstr}:{tile} in {(time.time()-t0)/60.0:.2f} min.")






