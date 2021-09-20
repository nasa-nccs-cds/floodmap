import xarray as xr
from ..util.logs import getLogger
from typing import List, Union, Tuple, Dict, Optional

result_colors = [   ( 0, 'nodata', (0, 0, 0)),
                    ( 1, 'land',   (0, 1, 0)),
                    ( 2, 'water',  (0, 0, 1)),
                    ( 3, 'interp land',   (0, 0.5, 0)),
                    ( 4, 'interp water',  (0, 0, 0.5)),
                    ( 5, 'mask', (0.25, 0.25, 0.25) ),
                    ( 6, 'mask', (0.25, 0.25, 0.25) ),
                    ( 7, 'mask', (0.25, 0.25, 0.25) ) ]

floodmap_colors = [ ( 0, 'land',         (0, 1, 0)),
                    ( 1, 'perm water',   (0, 0, 1)),
                    ( 2, 'flood water',  (0, 0, 0.5)),
                    ( 3, 'flood water',  (0, 0, 0.7)),
                    ( 255, 'nodata',     (0, 0, 0)) ]

def get_color_bounds( color_values: List[float] ) -> List[float]:
    color_bounds = []
    for iC, cval in enumerate( color_values ):
        if iC == 0: color_bounds.append( cval - 0.5 )
        else: color_bounds.append( (cval + color_values[iC-1])/2.0 )
    color_bounds.append( color_values[-1] + 0.5 )
    return color_bounds

def create_cmap( colors ):
    from matplotlib.colors import ListedColormap
    import matplotlib as mpl
    rgbs = [cval[2] for cval in colors]
    cmap: ListedColormap = ListedColormap(rgbs)
    tick_labels = [cval[1] for cval in colors]
    color_values = [float(cval[0]) for cval in colors]
    color_bounds = get_color_bounds(color_values)
    norm = mpl.colors.BoundaryNorm(color_bounds, len(colors))
    cbar_args = dict(cmap=cmap, norm=norm, boundaries=color_bounds, ticks=color_values, spacing='proportional', orientation='horizontal')
    return tick_labels, dict( cmap=cmap, norm=norm, cbar_kwargs=cbar_args )

def plot_array( title: str, array: xr.DataArray):
    try:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 1)
        figure.suptitle(title, fontsize=12)
        tick_labels, cmap_specs = create_cmap( result_colors )
        array.plot.imshow( ax=axes, **cmap_specs )
        plt.show()
    except Exception as err:
        logger = getLogger( True )
        logger.warning( f"Can't plot array due to error: {err}" )

def plot_floodmap_arrays( title: str, array: xr.DataArray ):
    from matplotlib.widgets import Slider
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 1)
        fig.suptitle(title, fontsize=12)
        tick_labels, cmap_specs = create_cmap( floodmap_colors )
        image = array[2].plot.imshow( ax=axes, **cmap_specs )

        # def update(val):
        #     ind = int(slider.val)
        #     im = array[ind].squeeze()
        #     image.set_data(im)
        #     fig.canvas.draw()
        #
        # slider = Slider( axes, 'Time index', 0, array.shape[0] - 1, valinit=0, valfmt='%i')
        # slider.on_changed(update)
        plt.show()

    except Exception as err:
        logger = getLogger( True )
        logger.warning( f"Can't plot array due to error: {err}" )
