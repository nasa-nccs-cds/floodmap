from floodmap.surfaceMapping.lakeExtentMapping import WaterMapGenerator

if __name__ == '__main__':
    waterMapGenerator = WaterMapGenerator()
    rbnds = [-75,30,-120,70]
    waterMapGenerator.get_pct_nodata( rbnds, day=275 )

