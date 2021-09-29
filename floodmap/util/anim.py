import matplotlib.animation as animation
from matplotlib.backend_bases import TimerBase
from matplotlib.colors import LinearSegmentedColormap, Normalize
from .configuration import ConfigurableObject, Region
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
from typing import List, Tuple, Union
import os, time
import xarray as xr

def rgb( r: int, g: int, b: int ):
    return ( r/255.0, g/255.0, b/255.0 )

class TilePlotter(ConfigurableObject):

    def __init__(self, **kwargs ):
        ConfigurableObject.__init__( self, **kwargs )
        colors = self.getParameter( "colors", [(0, 0, 0), (0.5, 1, 0.25), (1, 1, 0), (0, 0, 1)] )
        self.setColormap( colors )

    def setColormap(self, colors: List[Tuple[float,float,float]] ):
        self.norm = Normalize( 0,len(colors) )
        self.cm = LinearSegmentedColormap.from_list( "geoProc-TilePlotter", colors, N=len(colors) )

    def plot(self, axes, data_arrays: Union[xr.DataArray,List[xr.DataArray]], timeIndex = -1 ):
        print("Plotting tile")
        axes.set_yticklabels([]); axes.set_xticklabels([])
        if not isinstance(data_arrays, list): data_arrays = [data_arrays]
        if timeIndex >= 0:
            axes.imshow( data_arrays[timeIndex].values, cmap=self.cm, norm=self.norm )
        else:
            if len( data_arrays ) == 1:
                axes.imshow( data_arrays[0].values, cmap=self.cm, norm=self.norm )
            else:
                da: xr.DataArray = self.time_merge( data_arrays )
                result = da[0].copy()
                result = result.where( result == 0, 0 )
                land = ( da == 1 ).sum( axis=0 )
                perm_water = ( da == 2 ).sum( axis=0 )
                print( "Computed masks" )
                result = result.where( land == 0, 1 )
                result = result.where( perm_water == 0, 2 )
                axes.imshow( result.values, cmap=self.cm, norm=self.norm )

class ArrayAnimation(ConfigurableObject):

    def __init__( self, **kwargs ):
        ConfigurableObject.__init__( self, **kwargs )
        self.animator: animation.ArtistAnimation = None
        self._running = False
        self.fps = kwargs.get('fps',1)

    def create_file_animation( self,  files: List[str], savePath: str = None, **kwargs ) -> animation.TimedAnimation:
        from floodmap.util.xgeo import XGeo
        bbox: Region = self.getParameter("bbox")
        data_arrays: List[xr.DataArray] = XGeo.loadRasterFiles(files, region=bbox)
        return self.create_animation( data_arrays, savePath, **kwargs )

    @classmethod
    def loadRasterFile( cls, filePath: str, **args ) -> xr.DataArray:
        from .grid import GDALGrid
        dirName = os.path.basename(os.path.dirname(filePath))
        name: str = args.get( "name", os.path.basename( filePath ) )
        band: int = args.get("band",-1)
        grid = GDALGrid( filePath )
        if name is None: name = os.path.basename(filePath)
        return grid.xarray( name, band )

    @classmethod
    def loadRasterFiles( cls, filePaths: List[str], **args ) -> List[xr.DataArray]:
        bbox: Region = args.get("region")
        if bbox is None:
            data_arrays: List[xr.DataArray] = [ cls.loadRasterFile( file, **args ) for file in filePaths]
        else:
            data_arrays: List[xr.DataArray] = [ cls.loadRasterFile( file, **args )[ bbox.origin[1]:bbox.bounds[1], bbox.origin[0]:bbox.bounds[0] ] for file in filePaths]
        return data_arrays

    def create_array_animation(self,  data_array: xr.DataArray, savePath: str = None, **kwargs ) -> animation.TimedAnimation:
        data_arrays: List[xr.DataArray] = [  data_array[iT] for iT in range(data_array.shape[0]) ]
        return self.create_animation( data_arrays, savePath, **kwargs )

    def toggle_animation(self, arg ):
        print( f'toggle_animation: {arg}' )
        source: TimerBase = self.animator.event_source
        if self._running: source.stop()
        else: source.start( interval=1000.0/self.fps )
        self._running = not self._running

    def create_animation( self, data_arrays: List[xr.DataArray], savePath: str = None, **kwargs ) -> animation.TimedAnimation:
        images = []
        overwrite = kwargs.get('overwrite', False )
        display = kwargs.get('display', True)
        t0 = time.time()
        color_map = kwargs.get( 'color_map', None )
        if color_map is not None:
            nc = len(color_map)
            colors = [ color_map[ic] for ic in range(len(color_map)) ]
            norm = Normalize( 0, len(colors)-1 )
            color_map = LinearSegmentedColormap.from_list( "lake-map", colors, N=nc )
        else:
            color_map = kwargs.get('cmap',"jet")
            drange = kwargs.get( 'range' )
            norm = Normalize(*drange) if drange else None
        self.fps = self.getParameter( "fps", self.fps )
        roi = self.getParameter("roi")
        print(f"\n Executing create_array_animation, input shape = {data_arrays[0].shape}, dims = {data_arrays[0].dims} ")
        figure, axes = plt.subplots()
        overlays = kwargs.get('overlays', {})
        if (len(data_arrays) == 1) and data_arrays[0].ndim == 3:
            data_arrays = [ data_arrays[0][iS] for iS in range( data_arrays[0].shape[0] ) ]

        if roi is  None:
            axes.set_yticklabels([]); axes.set_xticklabels([])
            for iF, da in enumerate(data_arrays):
                im: Image = axes.imshow( da.values, animated=True, cmap=color_map, norm=norm )
                for color, overlay in overlays.items():
                    overlay.plot(ax=axes, color=color, linewidth=2)
                ts = str(da.time.values).split('T')[0]
                t = axes.annotate( f"{da.name}[{iF}/{len(data_arrays)}]: {ts}", (0,0) )
                images.append([im,t])
        else:
            for iF, da in enumerate(data_arrays):
                im0: Image = axes.imshow( da.values, animated=True, cmap=color_map, norm=norm  )
                for color, overlay in overlays.items():
                    overlay.plot(ax=axes, color=color, linewidth=2)
                ts = str(da.time.values).split('T')[0]
                t = axes.annotate( f"Lake[{iF}/{len(data_arrays)}]: {ts}", ( 0,0) )
                images.append( [im0,t] )

        self.animator = animation.ArtistAnimation( figure, images, interval=1000.0/self.fps )
        axstop = plt.axes([0.8, 0.01, 0.15, 0.05])
        bstop = Button(axstop, 'Start/Stop')
        bstop.on_clicked( self.toggle_animation )
        self._running = True

        if savePath is not None:
            if ( overwrite or not os.path.exists( savePath )):
                self.animator.save( savePath, fps=fps )
                print( f" Animation saved to {savePath}" )
            else:
                print( f" Animation file already exists at '{savePath}'', set 'overwrite = True'' if you wish to overwrite it." )
        print(f" Completed create_array_animation in {time.time()-t0:.3f} seconds" )
        if display: plt.show()
        return self.animator

    def getDataSubset( self, data_arrays: List[xr.DataArray], frameIndex: int, bin_size: 8, roi: Region ):
        results = []
        for iFrame in range(frameIndex,frameIndex+bin_size):
            da = data_arrays[ min( iFrame, len(data_arrays)-1 ) ]
            results.append( da[ roi.origin[0]:roi.bounds[0], roi.origin[1]:roi.bounds[1] ] )
        return results

    def create_watermap_diag_animation(self, title: str, data_arrays: List[xr.DataArray], savePath: str = None, overwrite=False) -> animation.TimedAnimation:
        images = []
        t0 = time.time()
        colors = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)]
        norm = Normalize(0, 4)
        cm = LinearSegmentedColormap.from_list("lake-map", colors, N=4)
        fps = self.getParameter("fps", 0.5)
        roi: Region = self.getParameter("roi")
        print("\n Executing create_array_animation ")
        figure, axes = plt.subplots(2, 2)
        water_maps = [{}, {}, {}]
        figure.suptitle(title, fontsize=16)

        anim_running = True

        def onClick(event):
            nonlocal anim_running
            if anim_running:
                anim.event_source.stop()
                anim_running = False
            else:
                anim.event_source.start()
                anim_running = True

        for frameIndex in range(len(data_arrays)):
            waterMaskIndex = frameIndex // 8
            da0 = data_arrays[frameIndex]
            waterMask11 = water_maps[0].setdefault(waterMaskIndex, waterMapGenerator.get_water_mask(self.getDataSubset(data_arrays, frameIndex, 8, roi), 0.5, 1))
            waterMask12 = water_maps[1].setdefault(waterMaskIndex, waterMapGenerator.get_water_mask(self.getDataSubset(data_arrays, frameIndex, 8, roi), 0.5, 2))
            waterMask13 = water_maps[2].setdefault(waterMaskIndex, waterMapGenerator.get_water_mask(self.getDataSubset(data_arrays, frameIndex, 8, roi), 0.5, 3))
            #            im0: Image = axes[0].imshow(da.values, animated=True, cmap=cm, norm=norm  )
            axes[0, 0].title.set_text('raw data');
            axes[0, 0].set_yticklabels([]);
            axes[0, 0].set_xticklabels([])
            im0: Image = axes[0, 0].imshow(da0[roi.origin[0]:roi.bounds[0], roi.origin[1]:roi.bounds[1]], animated=True, cmap=cm, norm=norm)
            axes[0, 1].title.set_text('minw: 1');
            axes[0, 1].set_yticklabels([]);
            axes[0, 1].set_xticklabels([])
            im1: Image = axes[0, 1].imshow(waterMask11, animated=True, cmap=cm, norm=norm)
            axes[1, 0].title.set_text('minw: 2');
            axes[1, 0].set_yticklabels([]);
            axes[1, 0].set_xticklabels([])
            im2: Image = axes[1, 0].imshow(waterMask12, animated=True, cmap=cm, norm=norm)
            axes[1, 1].title.set_text('minw: 3');
            axes[1, 1].set_yticklabels([]);
            axes[1, 1].set_xticklabels([])
            im3: Image = axes[1, 1].imshow(waterMask13, animated=True, cmap=cm, norm=norm)
            images.append([im0, im1, im2, im3])

        #        rect = patches.Rectangle( roi.origin, roi.size, roi.size, linewidth=1, edgecolor='r', facecolor='none')
        #        axes[0].add_patch(rect)
        figure.canvas.mpl_connect('button_press_event', onClick)
        anim = animation.ArtistAnimation(figure, images, interval=1000.0 / fps, repeat_delay=1000)

        if savePath is not None:
            if (overwrite or not os.path.exists(savePath)):
                anim.save(savePath, fps=fps)
                print(f" Animation saved to {savePath}")
            else:
                print(f" Animation file already exists at '{savePath}'', set 'overwrite = True'' if you wish to overwrite it.")
        print(f" Completed create_array_animation in {time.time() - t0:.3f} seconds")
        plt.tight_layout()
        plt.show()
        return anim


    def create_multi_array_animation(self, title: str, data_arrays: List[xr.DataArray], savePath: str = None, overwrite=False, colors = None, count_values = None ) -> animation.TimedAnimation:
        anim_frames = []
        t0 = time.time()
        cm_colors = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)] if colors is None else colors
        norm = Normalize(0, len(cm_colors))
        cm = LinearSegmentedColormap.from_list( "tmp-colormap", cm_colors, N=len(cm_colors))
        fps = self.getParameter("fps", 1.0)
        roi: Region = self.getParameter("roi")
        print("\n Executing create_array_animation ")
        figure, axes = plt.subplots(1, len(data_arrays) if count_values is None else len(data_arrays) + 1 )
        figure.suptitle(title, fontsize=16)

        anim_running = True

        def onClick(event):
            nonlocal anim_running
            if anim_running:
                anim.event_source.stop()
                anim_running = False
            else:
                anim.event_source.start()
                anim_running = True


        counts = None if count_values is None else data_arrays[-1].xgeo.countInstances(count_values)

        for frameIndex in range(data_arrays[0].shape[0]):
            images = []
            for iImage, data_array in enumerate(data_arrays):
                axis = axes[ iImage ]
                axis.title.set_text( data_array.name )
                axis.set_yticklabels([]); axis.set_xticklabels([])
                image_data = data_array[frameIndex] if roi is None else data_array[frameIndex, roi.origin[0]:roi.bounds[0], roi.origin[1]:roi.bounds[1]]
                image: Image = axis.imshow( image_data, animated=True, cmap=cm, norm=norm )
                images.append( image )

            if count_values is not None:
                bar_heights = counts[frameIndex]
                axis: plt.Axes =  axes[ len(data_arrays) ]
                axis.set_yticklabels([]); axis.set_xticklabels([])
                plot = axis.barh( [0,1], bar_heights.values, animated=True )
                images.append( plot )

            anim_frames.append( images )

        figure.canvas.mpl_connect('button_press_event', onClick)
        anim = animation.ArtistAnimation(figure, anim_frames, interval=1000.0 / fps, repeat_delay=1000)

        if savePath is not None:
            if (overwrite or not os.path.exists(savePath)):
                anim.save(savePath, fps=fps)
                print(f" Animation saved to {savePath}")
            else:
                print(f" Animation file already exists at '{savePath}'', set 'overwrite = True'' if you wish to overwrite it.")
        print(f" Completed create_array_animation in {time.time() - t0:.3f} seconds")
#        plt.tight_layout()
        plt.show()
        return anim

    def animateGifs(self, gifList: List[str] ):
        images = [ Image.open(gifFile).convert('RGB') for gifFile in gifList ]
        nImages = len( images )
        nRows = nImages // 3
        nCols = nImages // nRows
        figure, axes = plt.subplots( nRows, nCols )



#
# if __name__ == '__main__':
#     from geoproc.data.mwp import MWPDataManager
#
#     t0 = time.time()
#     locations = [ "090W050N" ]
#     products = [  "2D2OT" ]
#     DATA_DIR = "/Users/tpmaxwel/Dropbox/Tom/Data/Birkitt"
#     location: str = locations[0]
#     product = products[0]
#     year = 2016
#     download = False
#     roi = None
#     bbox = Region( [3000,3500], 750 )
#     savePath = DATA_DIR + "/watermap_diagnostic_animation.gif"
#     fps = 1.0
#     time_index_range = [ 0, 30 ]
#
#     dataMgr = MWPDataManager(DATA_DIR, "https://floodmap.modaps.eosdis.nasa.gov/Products")
#     dataMgr.setDefaults( product=product, download=download, year=year, start_day=time_index_range[0], end_day=time_index_range[1], bbox=bbox )
#     data_arrays = dataMgr.get_tile_data(location)
#
#     animator = ArrayAnimation( roi=roi, fps=fps )
#     anim = animator.create_animation( data_arrays )
# #    anim = animator.create_watermap_diag_animation( f"{product} @ {location}", data_arrays, savePath, True )
