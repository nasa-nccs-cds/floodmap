import geopandas as gpd
import pandas as pd
from datetime import date
from osgeo import osr
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
import xarray, getpass
from floodmap.util.configuration import opSpecs
import xarray as xr
from glob import glob
import functools, traceback
from floodmap.util.xrio import XRio
from ..util.configuration import sanitize, ConfigurableObject, sanitize_ds
from .tiles import TileLocator
from ..util.logs import getLogger
import numpy as np
from datetime import datetime
import os, time, collections
from floodmap.util.xgeo import XGeo

def class_sum( class_cnts: List[int], class_vals: List[int] ) -> int:
    csum = 0
    for cval in class_vals: csum = csum + class_cnts[cval]
    return csum

class WaterMapGenerator(ConfigurableObject):

    def __init__( self, **kwargs ):
        ConfigurableObject.__init__( self, **kwargs )
        self.water_map: xr.DataArray = None
        self.water_probability: xr.DataArray = None
        self.persistent_classes: xr.DataArray = None
        self.lake_mask: xr.DataArray = None
        self.roi_bounds: gpd.GeoSeries = None
        self.mask_value = kwargs.get( 'mask_value', 5 )
        self.logger = getLogger( False )

    def get_water_map_colors(self) -> List[Tuple]:
        return [(0, 'nodata', (0, 0, 0)),
               (1, 'land', (0, 1, 0)),
               (2, 'water', (0, 0, 1)),
               (3, 'int-land', (0, 0.6, 0)),
               (4, 'int-water', (0, 0, 0.6)),
               (self.mask_value, 'mask', (1, 1, 0.7)) ]

    def get_input_colors( self, mask_value=5 ) -> List[Tuple]:
        return [(0, 'undetermined', (1, 1, 0)),
               (1, 'land', (0, 1, 0)),
               (2, 'water', (0, 0, 1)),
               (mask_value, 'mask', (0.25, 0.25, 0.25))]

    @classmethod
    def get_date_from_year(cls, year: int):
        return np.datetime64( datetime(year, 7, 1) )

    def get_viable_file(self, fpaths: List[str] ) -> str:
        for fpath in fpaths:
            if os.path.isfile(fpath):
                return fpath
        raise Exception( f"None of the tested file paths are viable: {fpaths}")

    # def get_yearly_lake_area_masks( self, opspec: Dict, **kwargs) -> Optional[xr.DataArray]:
    #     t0 = time.time()
    #     images = {}
    #     data_dir = opspec.get('data_dir')
    #     wmask_opspec = opspec.get('water_masks')
    #     if wmask_opspec is None: return None
    #     lake_masks_dir: str = wmask_opspec.get('location', "" ).replace("{data_dir}",data_dir)
    #     lake_index = opspec.get('index')
    #     lake_id = opspec.get('id' )
    #     yearly_lake_masks_file = os.path.join(data_dir, f"Lake{lake_id}_fill_masks.nc")
    #     cache = kwargs.get('cache',"update")
    #     lake_mask_nodata = int( wmask_opspec.get('nodata', 256) )
    #     if cache==True and os.path.isfile( yearly_lake_masks_file ):
    #         yearly_lake_masks_dataset: xr.Dataset = xr.open_dataset(yearly_lake_masks_file)
    #         yearly_lake_masks: xr.DataArray = yearly_lake_masks_dataset.yearly_lake_masks
    #     else:
    #          for sdir in glob( f"{lake_masks_dir}/*" ):
    #             year = os.path.basename(sdir)
    #             filepaths = [ f"{lake_masks_dir}/{year}/{prefix}{lake_index}_{year}.tif" for prefix in ["lake", ""] ]
    #             images[int(year)] = self.get_viable_file( filepaths )
    #          sorted_file_paths = OrderedDict(sorted(images.items()))
    #          time_values = np.array([self.get_date_from_year(year) for year in sorted_file_paths.keys()], dtype='datetime64[ns]')
    #          yearly_lake_masks: xr.DataArray = XRio.load(list(sorted_file_paths.values()), band=0, mask_value=lake_mask_nodata, index=time_values)
    #          yearly_lake_masks = yearly_lake_masks.where( yearly_lake_masks != lake_mask_nodata, self.mask_value )
    #
    #     if cache in [ True, "update" ]:
    #         result = xr.Dataset(dict(yearly_lake_masks=sanitize(yearly_lake_masks)))
    #         result.to_netcdf(yearly_lake_masks_file)
    #         self.logger.info(f"Saved cropped_data to {yearly_lake_masks_file}")
    #
    #     yearly_lake_masks = yearly_lake_masks.persist()
    #     self.logger.info(f"Done yearly_lake_masks in time {time.time() - t0} secs")
    #     return yearly_lake_masks

    def get_persistent_classes(self, opspec: Dict, **kwargs) -> xr.DataArray:
        # Computes perm water and perm land using occurrence thresholds over available history
        water_probability: xr.DataArray = self.get_water_probability(opspec, **kwargs)
        self.logger.info(f"Executing get_persistent_classes")
        results_dir = opspec.get('results_dir')
        lake_index = opspec.get('lake_index')
        t0 = time.time()
        cache = kwargs.get('cache', "update")
        thresholds = opspec.get('water_class_thresholds', [ 0.05, 0.95 ] )
        perm_water_mask: xr.DataArray = water_probability > thresholds[1]
        boundaries_mask: xr.DataArray = water_probability > 1.0
        perm_land_mask: xr.DataArray = water_probability < thresholds[0]
        roi_mask: xr.DataArray = (self.water_map >= self.mask_value) | boundaries_mask
        result = xr.where(roi_mask, self.water_map, xr.where(perm_water_mask, np.uint8(2), xr.where(perm_land_mask, np.uint8(1), np.uint8(0))))
        result = result.persist()
        result.name = "Persistent_Classes"
        self.logger.info(f"Done get_persistent_classes in time {time.time() - t0}")
        persistent_class_map = result.assign_attrs( cmap = dict( colors=self.get_water_map_colors() ) )
        if cache in [ True, "update" ]:
            persistent_class_map_file = os.path.join(results_dir, f"lake_{lake_index}_persistent_class_map.nc")
            result = xr.Dataset(dict(persistent_class_map=sanitize(persistent_class_map)))
            sanitize_ds(result).to_netcdf(persistent_class_map_file)
            msg = f"Saved persistent_class_map to {persistent_class_map_file}"
            self.logger.info(msg); print( msg )
        return persistent_class_map

    def get_water_probability( self, opspec: Dict, **kwargs ) -> xr.DataArray:
        self.logger.info(f"Executing get_water_probability" )
        t0 = time.time()
        cache = kwargs.get( "cache", "update" )
        data_dir = opspec.get('results_dir')
        lake_index = opspec['lake_index']
        water_probability_file = os.path.join(data_dir, f"lake_{lake_index}_water_probability.nc")

        if cache==True and os.path.isfile( water_probability_file ):
            water_probability_dataset: xr.Dataset = xr.open_dataset(water_probability_file)
            water_probability: xr.DataArray = water_probability_dataset.water_probability
        else:
            water = ( self.floodmap_data == 2 )
            land = ( self.floodmap_data == 1 )
            unmasked = ( self.floodmap_data < 5 )
            water_cnts = water.sum(axis=0)
            land_cnts = land.sum(axis=0)
            visible_cnts = (water_cnts + land_cnts)
            water_probability: xr.DataArray = water_cnts / visible_cnts
            water_probability = water_probability.where( unmasked, 1.01 )
            water_probability.name = "water_probability"
            if cache in [True,"update"]:
                result = xr.Dataset(dict(water_probability=sanitize(water_probability)))
                sanitize_ds(result).to_netcdf(water_probability_file)
                msg = f"Saved water_probability to {water_probability_file}"
                self.logger.info( msg ); print( msg )
        water_probability = water_probability.persist()
        self.logger.info(f"Done get_water_probability in time {time.time() - t0}")
        return water_probability

    def count_class_occurrences(self, da: xr.DataArray, class_values: List[int], axis=0 ) -> xr.DataArray:
        ma: Optional[xr.DataArray] = None
        for cval in class_values:
            ccount = (da == cval).sum(axis=axis)
            ma = ccount if ma is None else ma + ccount
        return ma

    def total_class_occurrences(self, da: xr.DataArray, cval: int ) -> int:
        return np.count_nonzero( da.values == cval )

    def compute_raw_water_map(self)-> xr.Dataset:
        self.logger.info(f"compute_raw_water_map")
        water_maps_opspec = opSpecs.get('water_map', {})
        bin_size = water_maps_opspec.get( 'bin_size', 8 )
        threshold = water_maps_opspec.get('threshold', 0.5 )
        land_values = water_maps_opspec.get('land_values', [0] )
        water_values = water_maps_opspec.get('water_values', [1,2,3] )
        da: xr.DataArray = self.floodmap_data[-bin_size:]
        binSize = da.shape[0]
        masks = [ self.mask_value, self.mask_value+1, self.mask_value+2  ]
        nodata_value = da.attrs['_FillValue']
        da0 = da[0].drop_vars( self.floodmap_data.dims[0] )
        masked = da0.isin( masks )
        class_cnts = [ self.total_class_occurrences(da,cval) for cval in range(4) ]
        for cval in range(4):
            self.logger.info( f" --> class count({cval}): {class_cnts[cval]}" )
        land = self.count_class_occurrences( da, land_values )
        water =  self.count_class_occurrences( da, water_values )
        nodata = self.count_class_occurrences( da, [nodata_value] )
        visible = ( water + land )
        prob_h20 = water / visible
        water_inference = prob_h20 >= threshold
        land_mask = land > 0
        water_mask = water > 0
        interp_mask = nodata == binSize
        nndata = np.count_nonzero(interp_mask)
        nlandwater = np.count_nonzero(visible.values)
        mixed_class_data = land_mask & water_mask
        nmixed = np.count_nonzero(mixed_class_data.values)
        self.logger.info( f"compute_raw_water_map: land_values={land_values}, water_values={water_values}, #land={np.count_nonzero(land.values)}, #water={np.count_nonzero(water.values)}, total={water.values.size}")
        class_data = xr.where( masked, np.uint8(6), xr.where( water_inference, np.uint8(2), xr.where( land_mask, np.uint8(1), np.uint8(0) ) ) )
        result = da0.copy( data=class_data.values )
        result.attrs['nodata'] = 0
        result.attrs['masks']  = masks
        result.attrs['nndata'] = nndata
        result.attrs['nviz'] =  nlandwater + nndata
        result.attrs['nmixed'] = nmixed
        result.attrs['interp_mask'] = interp_mask
        result.attrs['_FillValue'] = 0
        mixed_classes = da0.copy( data=mixed_class_data )
        self.logger.info( f" NODATA count: {nndata}, mixed_classes count: {nmixed}" )
        return xr.Dataset( { "water_map": result,  "mixed_classes": mixed_classes, "interp_mask": interp_mask } )

    def get_raw_water_map(self, dstr: str, **kwargs):
        # data_array = timeseries of LANCE floodmap data over all years & days configures in specs, cropped to lake bounds
        # this method computes land & water pixels over bins of {bin_size} days using thresholds
        self.logger.info(" -------> Executing get_water_map ")
        t0 = time.time()
        data_dir = opSpecs.get('results_dir')
        lake_index = kwargs.get( 'index', 0 )
        water_map_file = os.path.join(data_dir, f"lake_{lake_index}_water_map_{dstr}.nc")
        water_data_file = os.path.join(data_dir, f"lake_{lake_index}_floodmap_data.nc")
        try:
            self.floodmap_data.to_netcdf( water_data_file, compat='override' )
            print( f"Saving floodmap data for lake {lake_index} to {water_data_file}")
        except Exception as err:
            self.logger.info( f"Unable to cache water_data to {water_data_file}: {err}" )

        water_map_dset:  xr.Dataset = self.compute_raw_water_map()

        try:
            sanitize_ds(water_map_dset).to_netcdf(water_map_file)
            self.logger.info(f"Cached water_map to {water_map_file}")
        except Exception as err:
            self.logger.info( f"Unable to cache water_map to {water_map_file}: {err}" )

        self.logger.info( f" ***** Completed get_water_map in {time.time()-t0:.3f} seconds" )
        water_map_array: xr.DataArray = water_map_dset.water_map
        # class_counts = self.get_class_counts( water_maps_array.values[0] )
        # for tI in range(water_maps_array.shape[0]):
        #     plot_array( f"get_water_map-{tI}", water_maps_array[tI] )
        water_map_array.name = "Water_Map"
        self.water_map: xr.DataArray =  water_map_array.assign_attrs(cmap = dict(colors=self.get_water_map_colors()))

    def update_metrics( self, data_array: xr.DataArray, **kwargs ):
        metrics = data_array.attrs.get('metrics', {} )
        metrics.update( **kwargs )
        data_array.attrs['metrics'] = metrics

    def interpolate( self, opspec: Dict, **kwargs ) -> xr.DataArray:
        highlight = kwargs.get( "highlight", True )
        ffill =  kwargs.get( "ffill", True )
#        remove_anomalies = kwargs.get( "remove_anomalies", False )
#         init_water_map: xr.DataArray = self.spatial_interpolate( opspec, **kwargs ) if remove_anomalies else self.water_map
        nndata = self.water_map.attrs['nndata']
        nviz = float(self.water_map.attrs['nviz'])
        nmixed = self.water_map.attrs['nmixed']
        interp_mask = self.water_map.attrs['interp_mask']
        pwmap: xr.DataArray = self.temporal_ffill(self.water_map) if ffill else self.water_map
        patched_result: xr.DataArray = pwmap.where( ~interp_mask, pwmap + 2 ) if highlight else pwmap
        pct_interp = (nndata / nviz) * 100.0 if (nviz > 0) else 0.0
        pct_mixed  = (nmixed / nviz) * 100.0 if (nviz > 0) else 0.0
        print( f" ---> interpolate---> interpolate: highlight={highlight}, %interp = {pct_interp}, %mixed = {pct_mixed}")
        patched_result.attrs['pct_interp'] = pct_interp
        patched_result.attrs['pct_mixed'] = pct_mixed
        return patched_result

    # def interpolate1(self, opspec: Dict, **kwargs) -> xr.DataArray:
    #     highlight = kwargs.get("highlight", True)
    #     ffill = kwargs.get("ffill", True)
    #     #        remove_anomalies = kwargs.get( "remove_anomalies", False )
    #     #         init_water_map: xr.DataArray = self.spatial_interpolate( opspec, **kwargs ) if remove_anomalies else self.water_map
    #     pwmap: xr.DataArray = self.temporal_ffill(self.water_map) if ffill else self.water_map
    #     valid_mask = (pwmap == self.water_map)
    #     patched_result: xr.DataArray = pwmap.where(valid_mask, pwmap + 2) if highlight else pwmap
    #     pct_interp = (valid_mask.size - np.count_nonzero(valid_mask)) * 100.0 / valid_mask.size
    #     print(f" ---> interpolate: highlight={highlight}, %interp = {pct_interp}")
    #     return patched_result

    def spatial_interpolate( self, opspec: Dict, **kwargs  ) -> xr.DataArray:
        t0 = time.time()
        persistent_classes: xr.DataArray = self.get_persistent_classes(opspec, **kwargs)
        override_current = kwargs.get( 'override_current', False )
        dynamics_class = kwargs.get( "dynamics_class", 0 )
        dynamics_mask: xr.DataArray = ( persistent_classes.isin( [dynamics_class] )  )
        if not override_current: dynamics_mask = dynamics_mask & ( self.water_map > 0 )
        result =  self.water_map.where( dynamics_mask, persistent_classes  )
        self.logger.info(f"Done spatial interpolate in time {time.time() - t0}")
        return result

    def temporal_ffill(self, water_map: xr.DataArray, **kwargs ) -> xr.DataArray:
        t0 = time.time()
        nodata = self.floodmap_data.attrs['_FillValue']
        water_history_data: xr.DataArray = self.floodmap_data.where( self.floodmap_data != nodata, np.nan )
        interp_water_history: xr.DataArray = water_history_data.ffill( water_history_data.dims[0] ).bfill( water_history_data.dims[0] ).astype( np.uint8 )
        interp_water_map: xr.DataArray = interp_water_history[-1,:,:].squeeze( drop = True )
        self.logger.info( f"Done temporal interpolate in time {time.time() - t0}, history data shape = {water_history_data.shape}" )
        result =  water_map.where( (water_map > 0), interp_water_map )
        return result

    def time_merge( cls, data_arrays: List[xr.DataArray], **kwargs ) -> xr.DataArray:
        time_axis = kwargs.get('time',None)
        frame_indices = range( len(data_arrays) )
        merge_coord = pd.Index( frame_indices, name=kwargs.get("dim","time") ) if time_axis is None else time_axis
        result: xr.DataArray =  xr.concat( data_arrays, dim=merge_coord )
        return result

    def get_mwp_data(self, **kwargs) -> Tuple[Optional[xr.DataArray], Optional[ List[date] ]]:
        from .mwp import MWPDataManager
        from floodmap.util.configuration import opSpecs
        lakeMaskSpecs: Dict = opSpecs.get("lake_masks", None)
        source_specs: Dict = opSpecs.get( 'source' )
        lake_id = kwargs.get('index')
        dataMgr = MWPDataManager.instance(**kwargs)
        tiles = dataMgr.list_required_tiles( roi=self.roi_bounds, lake_mask = self.lake_mask, id=lake_id )
        if len(tiles) == 0:
            msg = f"Processing lake {lake_id}: days={dataMgr.parms['day_range']}, ROI={self.roi_bounds}:\n\n ----> NO TILES!\n"
            self.logger.error( msg ); print( msg )
        else:
            msg = f"Processing lake {lake_id}: days={dataMgr.parms['day_range']}, ROI={self.roi_bounds}, using tiles: {tiles}"
            self.logger.info(msg); print(msg)
            dataMgr.download_mpw_data( tiles=tiles, **source_specs )
            cropped_tiles: Dict[str,xr.DataArray] = {}
            time_values = None
            file_paths = None
            for tile in tiles:
                try:
                    lake_mask_value =  lakeMaskSpecs.get('mask',0)
                    tile_filespec: Dict = dataMgr.get_tile(tile)
                    file_paths = list(tile_filespec.values())
                    time_values: List[date] = list(tile_filespec.keys())
                    tile_raster: Optional[xr.DataArray] =  XRio.load( file_paths, mask=self.roi_bounds, band=0,
                                            mask_value=self.mask_value, index=[np.datetime64(t) for t in time_values] )
                    self.logger.info( f"Reading Tile[{tile}], file_paths={file_paths}" )
                    self.logger.info(f" --> roi = {self.roi_bounds}, mask_value={self.mask_value}, time_values={time_values}")
                    if (tile_raster is not None) and tile_raster.size > 0:
                        if self.lake_mask is None:
                            cropped_tiles[tile] = tile_raster
                        else:
                            lake_mask_interp: xr.DataArray = sanitize( self.lake_mask, True )
                            lake_mask_interp = lake_mask_interp.interp_like( sanitize( tile_raster[0,:,:] ) )
                            lake_mask_interp = lake_mask_interp.fillna( lake_mask_value )
                            tile_mask: xr.DataArray = ( lake_mask_interp == lake_mask_value )
                            tile_mask_data: np.ndarray = np.broadcast_to( tile_mask.values, tile_raster.shape ).flatten()
                            tile_raster_data: np.ndarray = tile_raster.values.flatten()
                            tile_raster_data[ tile_mask_data ] = self.mask_value + 1
                            cropped_tiles[tile] = tile_raster.copy( data=tile_raster_data.reshape(tile_raster.shape) )
                except Exception as err:
                    try:
                        for file in file_paths:
                            if not os.path.isfile( file ): self.logger.warning( f"   --> File {file} does not exist!")
                        exc = traceback.format_exc()
                        msg = f"Error reading mpw data for tile {tile} \n  Error: {err}: \n{exc}"
                        self.logger.error( msg )
                        XRio.print_array_dims( file_paths )
                    except OSError: pass
            nTiles = len( cropped_tiles.keys() )
            if nTiles > 0:
                self.logger.info( f"Merging {nTiles} Tiles ")
                cropped_data = self.merge_tiles(cropped_tiles)
                cropped_data.attrs.update( roi = self.roi_bounds )
                cropped_data = cropped_data.persist()
                return cropped_data, time_values
            else:
                self.logger.error(f"NO Tiles avaialble!")
        return None, None

    @classmethod
    def get_class_count_layers(cls, class_layers: Dict[int,xr.DataArray] ) -> Tuple[xr.DataArray,xr.DataArray]:
        time = xr.DataArray( list(class_layers.keys()), name = "time" )
        class_data = xr.concat( list(class_layers.values()), dim=time )
        land = ( class_data == 1 ).sum( axis=0 )
        water =  ( class_data == 2 ).sum( axis=0 )
        return (water,land)

    @classmethod
    def update_classes(cls, mpw_data: xr.DataArray ) -> xr.DataArray:
        water = mpw_data.isin([1, 2, 3])
        land = ( mpw_data == 0 )
        nodata = np.logical_or( (mpw_data > 200), (mpw_data < 0) )
        result =  xr.where( nodata, np.uint8(0), xr.where( water, np.uint8(2), xr.where(land, np.uint8(1), mpw_data ) ) )
        result['spatial_ref'] = mpw_data.spatial_ref
        return result

    def merge_tiles(self, cropped_tiles: Dict[str,xr.DataArray], name="mpw" ) -> xr.DataArray:
        lat_bins = {}
        for (key, cropped_tile) in cropped_tiles.items():
            lat_bins.setdefault( key[0:4], [] ).append( cropped_tile )
        concat_tiles = [ self.merge_along_axis( sub_arrays, -2 ) for sub_arrays in lat_bins.values() ]
        result =  self.merge_along_axis( concat_tiles, -1 )
        result.name = name
        return result

    def merge_along_axis( self, sub_arrays: List[xr.DataArray], axis: int ) -> xr.DataArray:
        if len( sub_arrays ) == 1:  return sub_arrays[0]
        concat_dim = sub_arrays[0].dims[axis]
        ccoord: np.ndarray = sub_arrays[0].coords[concat_dim].values
        reverse = (ccoord[0] > ccoord[-1])
        sub_arrays.sort( reverse =reverse , key = lambda x: x.coords[concat_dim].values[0] )
        result: xr.DataArray = xr.concat( sub_arrays, dim = concat_dim )
        print(f"Merging tiles with shapes: {[ta.shape for ta in sub_arrays]} along axis {axis}, result shape = {result.shape}")
        return result

    def merge_opspec_values( self, value0, value1 ):
        if isinstance( value0, collections.Mapping ):
            return { **value0, **value1 }
        else: return value1

    # def get_roi_bounds(self ):
    #     data_dir = opSpecs.get('data_dir')
    #     roi = opSpecs.get('roi', None)
    #     if roi is not None:
    #         if isinstance(roi, list):
    #             self.roi_bounds = [ float(x) for x in roi ]
    #         elif isinstance(roi, str) and "," in roi:
    #             self.roi_bounds = [ float(x) for x in roi.split(",") ]
    #         elif isinstance(roi, str):
    #             self.roi_bounds: gpd.GeoSeries = gpd.read_file( roi.replace("{data_dir}", data_dir) )
    #         else:
    #             raise Exception( f" Unrecognized roi: {roi}")
    #     else:
    #         assert self.lake_mask is not None, "Must specify roi to locate lake"
    #         self.roi_bounds =  TileLocator.get_bounds(self.lake_mask[0])

    # def get_patched_water_map(self, name: str, **kwargs) -> xr.DataArray:
    #     t0 = time.time()
    #     opspec = self.get_opspec( name.lower() )
    #     data_dir = opspec.get('data_dir')
    #     lake_index = opspec['lake_index']
    #     lake_id = f"{name}.{lake_index}"
    #     patched_water_map_file = f"{data_dir}/{lake_id}_patched_water_map.nc"
    #     cache = kwargs.get("cache", False )
    #     patch = kwargs.get("patch", True)
    #
    #     if cache==True and os.path.isfile(patched_water_map_file):
    #         patched_water_map: xr.DataArray = xr.open_dataset(patched_water_map_file).Water_Maps
    #         patched_water_map.attrs['cmap'] = dict(colors=self.get_water_map_colors())
    #     else:
    #         self.lake_mask: xr.DataArray = self.get_yearly_lake_area_masks(opspec, **kwargs)
    #         self.get_roi_bounds( opspec )
    #         water_mapping_data = self.get_mpw_data( **opspec, cache="update" )
    #         self.get_raw_water_map(water_mapping_data, opspec)
    #         patched_water_map = self.patch_water_map( opspec, **kwargs ) if patch else self.water_map
    #
    #     if ((cache == True) and not os.path.isfile(patched_water_map_file)) or ( cache == "update" ):
    #         sanitize(patched_water_map,True).to_netcdf( patched_water_map_file )
    #         self.logger.info( f"Saving patched_water_map to {patched_water_map_file}")
    #
    #     self.logger.info(f"Completed get_patched_water_map in time {(time.time() - t0)/60.0} minutes")
    #     patched_water_map.name = lake_id
    #     return patched_water_map.assign_attrs( roi = self.roi_bounds )

    def write_result_report( self, lake_index, report: str ):
        results_dir = opSpecs.get('results_dir')
        file_path = f"{results_dir}/lake_{lake_index}_task_report.txt"
        with open( file_path, "w" ) as file:
            file.write( report )

    def compute_pct_nodata(self, **kwargs) -> Optional[xr.DataArray]:
        from floodmap.surfaceMapping.mwp import MWPDataManager
        dstr = MWPDataManager.instance().get_dstr(**kwargs)
        lake_index = kwargs.get('index', 0)
        self.lake_mask: Optional[xr.DataArray] = kwargs.get('mask', None)
        self.roi_bounds = kwargs.get('roi', None)
        skip_existing = opSpecs.get('skip_existing', True)
        format = opSpecs.get('format', 'tif')
        results_dir = opSpecs.get('results_dir')
        patched_water_map_file = f"{results_dir}/lake_{lake_index}_patched_water_map_{dstr}"
        result_water_map_file = patched_water_map_file + ".tif" if format == 'tif' else patched_water_map_file + ".nc"
        if skip_existing and os.path.isfile(result_water_map_file):
            msg = f" Lake[{lake_index}]: Skipping already processed file: {result_water_map_file}"
            self.logger.info(msg), print(msg)
            return None
        else:
            (self.floodmap_data, time_values) = self.get_mwp_data(**kwargs)
            if self.floodmap_data is None:
                return None
            self.logger.info(f" --------------------->> Generating result file: {result_water_map_file}")
            self.logger.info(f"process_yearly_lake_masks: water_mapping_data shape = {self.floodmap_data.shape}")
            self.logger.info(f"yearly_lake_masks roi_bounds = {self.roi_bounds}")
            stats_file = f"{results_dir}/lake_{lake_index}_stats.txt"
            self.write_nodata_pct_results( stats_file )
            msg = f"Saving results for lake {lake_index} to {stats_file} and {result_water_map_file}"
            self.logger.info(msg)
            print(msg)

    def class_counts( self, label: str, raster: xarray.DataArray ):
        print(f" --------------------- {label} class counts: shape={raster.shape} size={raster.size} --------------------- ")
        for iclass in range(0, 8):
            print(f" ** [{iclass}]: {np.count_nonzero(raster.isin([iclass]))}")

    def generate_lake_water_map(self, **kwargs) -> Optional[xr.DataArray]:
        from floodmap.surfaceMapping.mwp import MWPDataManager
        from floodmap.util.crs import CRS
        dataMgr = MWPDataManager.instance()
        lake_index = kwargs.get('index',0)
        msg = f" ** generate_lake_water_map[{lake_index}]: parms={dataMgr.parms}"; self.logger.info(msg); print(msg)
        self.lake_mask: Optional[xr.DataArray] = kwargs.get('mask',None)
        self.roi_bounds = kwargs.get('roi', None)
        format = opSpecs.get('format','tif')
        results_dir = os.path.join( opSpecs.get('results_dir'), getpass.getuser() )
        os.makedirs( results_dir, exist_ok=True )
        results_file = opSpecs.get('results_file', f'lake_{lake_index}_stats.csv').format( lake_index=lake_index )
        (self.floodmap_data, time_values) = self.get_mwp_data(**kwargs)
        if time_values is None:
            self.logger.warn(f" No time values! ")
        else:
            dtime: date = time_values[-1]
            dstr = f"{dtime.month:02d}{dtime.day:02d}{dtime.year}"
            patched_water_map_file = f"{results_dir}/lake_{lake_index}_patched_water_map_{dstr}"
            result_water_map_file = patched_water_map_file + ".tif" if format ==  'tif' else patched_water_map_file + ".nc"
            result_geog_water_map_file = patched_water_map_file + "-geog.nc"
            if self.floodmap_data is None:
                self.logger.warn(f" No floodamp data! ")
                return None
            else:
                try:
                    times = [ np.datetime64(timestr) for timestr in time_values ]  # datetime.strptime(f"{timestr}", '%Y%j').date()
                    nrt_input_data = sanitize( self.floodmap_data.assign_coords( time = np.array( times, dtype='datetime64') ) )
                    water_data_file = os.path.join( results_dir, f"lake_{lake_index}_nrt_input_data_{dstr}.nc")
                    nrt_input_data.to_netcdf( water_data_file )
                except Exception as err:
                    self.logger.warn( f" Can't save nrt_input_data: {err} " )
            for iT in range( self.floodmap_data.shape[0] ):
                raster = self.floodmap_data[ iT, :, : ].squeeze( drop=True )
                self.class_counts( f"floodmap_data[{iT}]", raster )
            self.logger.info(f" --------------------->> Generating result file: {result_water_map_file}")
            self.logger.info( f"process_yearly_lake_masks: water_mapping_data shape = {self.floodmap_data.shape}")
            self.logger.info(f"yearly_lake_masks roi_bounds = {self.roi_bounds}")
            self.get_raw_water_map( dstr, **kwargs )
            patched_water_map = self.patch_water_map( **kwargs )
            patched_water_map.attrs['crs'] = CRS.get_geographic_proj4()
            patched_water_map.name = f"Lake-{lake_index}"
            print( f"LAKE[{lake_index}]: Generated patched_water_map{patched_water_map.dims}, shape = {patched_water_map.shape}", flush=True )
            self.class_counts( 'patched_water_map', patched_water_map )
            [y,x] = [ patched_water_map.coords[c].values for c in patched_water_map.dims ]
            sref = CRS.get_utm_proj4( x[x.size//2], y[y.size//2] )
            utm_result = sanitize( patched_water_map.rio.reproject( sref, 250.0 ) )
            self.class_counts('utm_result', utm_result )
            latlon_result = sanitize( patched_water_map ).rename( dict( x="lon", y="lat" ) )
            stats_file = f"{results_dir}/{results_file}"
            self.write_water_area_results( dtime, utm_result, stats_file, **patched_water_map.attrs )
            try:
                if format ==  'tif':    utm_result.xgeo.to_tif( result_water_map_file )
                else:                   utm_result.astype(np.int).to_netcdf( result_water_map_file )
                latlon_result.astype(np.int).to_netcdf(result_geog_water_map_file)
            except Exception as err:
                self.logger.info( f"Error writing UTM result: {err}")

            msg = f"Saving results for lake {lake_index} to {stats_file} and {result_water_map_file} ({result_geog_water_map_file})"
            self.logger.info( msg ); print( msg )
            return patched_water_map.assign_attrs( roi = self.roi_bounds )

    def today(self) -> str:
        today = datetime.now()
        day_of_year = today.timetuple().tm_yday
        return f"{day_of_year}:{today.year}"

    def write_nodata_pct_results(self, outfile_path: str,  **kwargs ):
        from floodmap.surfaceMapping.mwp import MWPDataManager
        da: xr.DataArray = self.floodmap_data
        roi = self.roi_bounds
        sdate = MWPDataManager.instance().get_target_date()
        interp_water_class = kwargs.get( 'interp_water_class', 4 )
        water_classes = kwargs.get('water_classes', [2,4] )
        water_counts, class_proportion = self.get_class_proportion( da, interp_water_class, water_classes )
        file_exists = os.path.isfile(outfile_path)
        with open( outfile_path, "a" ) as outfile:
            if not file_exists:
                outfile.write( "date water_area_km2 percent_interpolated\n")
            percent_interp = class_proportion.values
            num_water_pixels = water_counts.values
            outfile.write( f"{sdate} {num_water_pixels/16.0:.2f} {percent_interp:.1f}\n" )
            self.logger.info( f"Wrote results to file {outfile_path}")

    def write_water_area_results(self, rdate: date, patched_water_map: xr.DataArray, outfile_path: str,  **kwargs ):
        interp_water_class = kwargs.get( 'interp_water_class', 4 )
        water_classes = kwargs.get('water_classes', [2,4] )
        num_water_pixels, class_proportion = self.get_class_proportion(patched_water_map, interp_water_class, water_classes)
        file_exists = os.path.isfile(outfile_path)
        with open( outfile_path, "a" ) as outfile:
            if not file_exists:
                outfile.write( "date water_area_km2 percent_interpolated percent_mixed\n")
            percent_interp = kwargs.get('pct_interp', class_proportion )
            percent_mixed = kwargs.get('pct_mixed', 0.0 )
            outfile.write( f"{rdate.month:02d}-{rdate.day:02d}-{rdate.year} {num_water_pixels/16.0:.2f} {percent_interp:.2f} {percent_mixed:.2f}\n" )
            self.logger.info( f"Wrote results to file {outfile_path}")

    def get_class_counts( self, array: np.ndarray )-> Dict:
        return { ic: np.count_nonzero( array == ic ) for ic in range(10) }

    # def repatch_water_map(self, lakeId: str, **kwargs) -> xr.DataArray:
    #     t0 = time.time()
    #     opspec = self.get_opspec( lakeId.lower() )
    #     data_dir = opspec.get('data_dir')
    #     lake_id = opspec['id']
    #     patched_water_map_file = f"{data_dir}/{lake_id}_patched_water_map.nc"
    #     cache = kwargs.get("cache", False )
    #
    #     self.lake_mask: xr.DataArray = self.get_yearly_lake_area_masks(opspec, **kwargs)
    #     self.get_roi_bounds( opspec )
    #     self.water_map: xr.DataArray =  self.get_raw_water_map(None, opspec, cache=True)
    #     patched_water_map = self.patch_water_map( opspec, **kwargs )
    #
    #     if ((cache == True) and not os.path.isfile(patched_water_map_file)) or ( cache == "update" ):
    #         sanitize(patched_water_map).to_netcdf( patched_water_map_file )
    #         self.logger.info( f"Saving patched_water_map to {patched_water_map_file}")
    #
    #     self.logger.info(f"Completed get_patched_water_map in time {(time.time() - t0)/60.0} minutes")
    #     patched_water_map.name = lake_id
    #     return patched_water_map.assign_attrs( roi = self.roi_bounds )

    def patch_water_map( self, **kwargs ) -> xr.DataArray:
        patched_water_map: xr.DataArray = self.interpolate( opSpecs, **kwargs ).assign_attrs(**self.water_map.attrs)
        patched_water_map.attrs['cmap'] = dict( colors=self.get_water_map_colors() )
        rv = patched_water_map.fillna( self.mask_value )
        return rv

    def get_class_proportion(self, class_map: xr.DataArray, target_class: int, relevant_classes: List[int] ) -> Tuple[float,float]:
        sdims = [ class_map.dims[-1], class_map.dims[-2] ]
        total_relevant_population = class_map.isin( relevant_classes ).sum( dim=sdims ).values
        class_population = (class_map == target_class).sum( dim=sdims ).values
        return ( total_relevant_population,  ( class_population / total_relevant_population ) * 100 )
