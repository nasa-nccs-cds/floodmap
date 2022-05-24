def hc(coord: float) -> str:
    nc = coord if coord < 180 else coord - 360
    rv = int((nc + 180) // 10)
    return f"h{rv:02d}"


for ic in range( -60, -120, -1 ):
    print( f" {ic+0.5}: {hc(ic)}")