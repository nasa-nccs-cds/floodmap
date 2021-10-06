import glob, os, time
from functools import partial
from multiprocessing import cpu_count, get_context, Pool, freeze_support
nproc = cpu_count()

collection = 61
archive_dir = "/att/pubrepo/ILAB/projects/Birkett/MOD44W/results"

def process_file( archive_dir: str, collection: str, hdfFilepath: str ) -> int:
    fName = os.path.basename(hdfFilepath)
    fdir = os.path.dirname(hdfFilepath)
    ftoks = fName.split('.')
    tile, dstr = ftoks[2], ftoks[1][1:]
    outpath = f"{tile}/allData/{collection}/MCDWD_L3_F2_NRT/Recent"
    result_file = f"MCDWD_L3_F2_NRT.A{dstr}.{tile}.{collection:03}.tif"
    os.makedirs(f'{archive_dir}/{outpath}', exist_ok=True)
    product = f"HDF4_EOS:EOS_GRID:MCDWD_L3_NRT.A{dstr}.{tile}.{collection:03}.hdf:Grid_Water_Composite:'Flood 2-Day 250m'"
    result_path = f"{archive_dir}/{outpath}/{result_file}"
    command = f"cd {fdir}; gdal_translate {product} {result_path} -q -co 'COMPRESS=DEFLATE'"
    rv = os.system(command)
    print(f" *** [{rv}]->      {outpath}:  {result_file}")
    return rv

if __name__ == '__main__':
    freeze_support()
    t0 = time.time()
    dstr = f"2021*"
    tile = "h*v*" # "h*v*" "h20v09"
    source_dir = f"/att/nobackup/dslaybac/MCDWD_NRT/MCDWD_L3_NRT_{dstr}"
    gfstr = f"{source_dir}/MCDWD_L3_NRT.A{dstr}.{tile}.{collection:03}.hdf"
    infiles = glob.glob( gfstr )
    print( f"Processing HDF files for [{dstr}]: '{gfstr}'" )
    print( f"Converting {len(infiles)} hdf files, saving to location: {archive_dir}:")
    processor = partial( process_file, archive_dir, collection )

    with get_context("spawn").Pool(processes=nproc) as p:
        results = p.map( processor, infiles )
    p.join()

    print( f"Completed processing files {dstr}:{tile} in {(time.time()-t0)/60.0:.2f} min.")






