import xarray
import traceback, xarray as xr
import numpy as np
from ..util.logs import getLogger
from typing import List, Union, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, MultiCursor

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
                    ( 4, 'nodata',       (0, 0, 0)) ]

def get_color_bounds( color_values: List[float] ) -> List[float]:
    color_bounds = []
    for iC, cval in enumerate( color_values ):
        if iC == 0: color_bounds.append( cval - 0.5 )
        else: color_bounds.append( (cval + color_values[iC-1])/2.0 )
    color_bounds.append( color_values[-1] + 0.5 )
    return color_bounds

def create_cmap( colors: List[Tuple] ):
    from matplotlib.colors import ListedColormap
    import matplotlib as mpl
    rgbs = [ cval[2] for cval in colors ]
    cmap: ListedColormap = ListedColormap(rgbs)
    tick_labels = [cval[1] for cval in colors]
    color_values = [float(cval[0]) for cval in colors]
    color_bounds = get_color_bounds(color_values)
    norm = mpl.colors.BoundaryNorm(color_bounds, len(colors))
    cbar_args = dict( boundaries=color_bounds, ticks=color_values, spacing='proportional', orientation='vertical' )
    return tick_labels, dict( cmap=cmap, norm=norm, cbar_kwargs=cbar_args )

def plot_array( ax, array: xr.DataArray, **kwargs ):
    try:
        import matplotlib.pyplot as plt
        colors = kwargs.get( 'colors', result_colors )
        plot_coords = kwargs.get('plot_coords', {})
        tick_labels, cmap_specs = create_cmap( colors )
        image = array.plot.imshow( ax=ax, **plot_coords, **cmap_specs )
        ax.title.set_text(kwargs.get('title',""))
    except Exception as err:
        logger = getLogger( True )
        logger.warning( f"Can't plot array due to error: {err}" )
        logger.error( traceback.format_exc() )

def update_cursor( cursor: MultiCursor ):
    if cursor is not None:
        for line in cursor.vlines:
            line.set_visible(cursor.visible)
        for line in cursor.hlines:
            line.set_visible(cursor.visible)
        cursor._update()

def plot_arrays( ax, arrays: Dict[int,xr.DataArray], **kwargs ):
    try:
        colors = kwargs.get( 'colors', result_colors )
        cursor = kwargs.get( 'cursor', None )
        plot_coords = kwargs.get('plot_coords',{})
        tvals = list(arrays.keys())
        t0, t1, a0 = tvals[0], tvals[-1], arrays[tvals[0]]
        tick_labels, cmap_specs = create_cmap( colors )
        image = a0.plot.imshow( ax=ax, **plot_coords, **cmap_specs )
        sax = plt.axes([0.2, 0.01, 0.6, 0.03])   # [left, bottom, width, height]
        slider = Slider( sax, 'Time index', t0, t1, t0, valfmt='%i', valstep=1 )
        title = kwargs.get('title', "")
        probe_arrays: Dict[str,xr.DataArray] = kwargs.get( 'probe_arrays', {})
        ax.title.set_text( title )

        def on_button(event):
            xc, yc = a0.x.values, a0.y.values
            if (event.xdata is not None) and (event.ydata is not None):
                ix =  np.abs( xc - event.xdata ).argmin()
                iy =  np.abs( yc - event.ydata ).argmin()
                print( f"ix, iy = [ {ix}, {iy} ]" )
                for (label,array) in probe_arrays.items():
                    print( f"{label}: {array.values[iy,ix]}" )

        def on_key(event):
            ind, new_ind = int(slider.val), -1
            if event.key in ['f', 'right', 'up' ]:
                if (event.key == 'up') and (ind == slider.valmax): return
                new_ind = ind + slider.valstep
                if new_ind > slider.valmax: new_ind = slider.valmin
            if event.key in ['b', 'left', 'down' ]:
                if (event.key == 'down') and (ind == slider.valmin): return
                new_ind = ind - slider.valstep
                if new_ind < slider.valmin: new_ind = slider.valmax
            if new_ind >= 0: slider.set_val( new_ind )

        def on_draw(event):
            update_cursor(cursor)

        ax.figure.canvas.callbacks.connect('button_press_event', on_button )
        ax.figure.canvas.callbacks.connect('key_press_event', on_key )
        ax.figure.canvas.callbacks.connect('draw_event', on_draw )

        def update(val):
            ind = int(slider.val)
            if ind in arrays:
                im = arrays[ind].squeeze()
                image.set_data(im)
                ax.figure.canvas.draw()

        slider.on_changed(update)
    except Exception as err:
        logger = getLogger( True )
        logger.warning( f"Can't plot array due to error: {err}" )
        logger.error(traceback.format_exc())

def plot_floodmap_arrays( title: str, array: xr.DataArray ):
    from matplotlib.widgets import Slider
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 1)
        fig.suptitle(title, fontsize=12)
        tick_labels, cmap_specs = create_cmap( floodmap_colors )
        image = array[2].plot.imshow( ax=axes, **cmap_specs )
        slider = Slider( axes, 'Time index', 0, array.shape[0] - 1, valinit=0, valfmt='%i')

        def update(val):
            ind = int(slider.val)
            im = array[ind].squeeze()
            image.set_data(im)
            fig.canvas.draw()

        slider.on_changed(update)
        plt.show()

    except Exception as err:
        logger = getLogger( True )
        logger.warning( f"Can't plot array due to error: {err}" )
        logger.error(traceback.format_exc())
