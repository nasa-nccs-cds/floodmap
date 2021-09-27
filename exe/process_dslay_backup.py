import glob, os

collection = 61
year = 2021
month = 10
day_range = [0,5]

for day in day_range:
    dstr = f"{year}{month:02}{day:02}"
    source_dir = f"/att/nobackup/dslaybac/MCDWD_NRT/MCDWD_L3_NRT_{dstr}"
    result_dir = "/att/nobackup/tpmaxwel/data/MCDWD_NRT"
    gfstr = f"{source_dir}/MCDWD_L3_NRT.A{dstr}.h*v*.{collection:03}.hdf"

    for filepath in glob.glob( gfstr ):
        fName = os.path.basename(filepath)
        fdir = os.path.dirname(filepath)
        tile = fName.split()[2]
        result_dir = f"{result_dir}/{tile}/allData/{collection}/MCDWD_L3_F2_NRT/Recent"
        result_file = f"{result_dir}/MCDWD_L3_F2_NRT.A{dstr}.{tile}.{collection:03}.tif"
        layer = f"HDF4_EOS:EOS_GRID:MCDWD_L3_NRT.A{dstr}.{tile}.{collection:03}.hdf:Grid_Water_Composite:'Flood 2-Day 250m'"
        command = f"cd {fdir}; gdal_translate {layer} {result_file} -ot Byte -co 'COMPRESS=JPEG'"
        os.system(command)




