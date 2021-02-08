import rasterio

source_file_path = "/Users/tpmaxwel/Development/Data/WaterMapping/LakeMasks/2019/5_2019.tif"

with rasterio.open( source_file_path ) as src:
    print( f"PROFILE: {src.profile}" )
    src_crs = ''.join(src.crs.wkt.split())
    print(f" ---> CRS: {src_crs}")