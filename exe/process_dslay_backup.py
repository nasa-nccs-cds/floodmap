import glob, os

collection = 61
year = 2021
day_range = [10,11]

for day in day_range:
    dstr = f"{year}{day:03}"
    source_dir = f"/att/nobackup/dslaybac/MCDWD_NRT/MCDWD_L3_NRT_{dstr}"
    archive_dir = "/att/nobackup/tpmaxwel/data/MCDWD_NRT"
    gfstr = f"{source_dir}/MCDWD_L3_NRT.A{dstr}.h*v*.{collection:03}.hdf"
    infiles = glob.glob( gfstr )
    print( f"Processing HDF files: '{gfstr}'" )
    print( f"Converting {len(infiles)} hdf files, saving to location: {archive_dir}:")

    for filepath in infiles:
        fName = os.path.basename(filepath)
        fdir = os.path.dirname(filepath)
        tile = fName.split('.')[2]
        outpath = f"{tile}/allData/{collection}/MCDWD_L3_F2_NRT/Recent"
        result_file = f"MCDWD_L3_F2_NRT.A{dstr}.{tile}.{collection:03}.tif"
        os.makedirs( f'{archive_dir}/{outpath}', exist_ok = True )
        product = f"HDF4_EOS:EOS_GRID:MCDWD_L3_NRT.A{dstr}.{tile}.{collection:03}.hdf:Grid_Water_Composite:'Flood 2-Day 250m'"
        result_path = f"{archive_dir}/{outpath}/{result_file}"
        command = f"cd {fdir}; gdal_translate {product} {result_path} -ot Byte -co 'COMPRESS=JPEG'"
        rv = os.system(command)
        print( f" [{rv}]-> {outpath}:  {result_file}" )




