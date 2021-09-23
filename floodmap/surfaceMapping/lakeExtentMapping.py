import geopandas as gpd
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
import xarray
import xarray as xr
from glob import glob
import functools, traceback
from floodmap.util.xrio import XRio
from ..util.configuration import sanitize, ConfigurableObject
from .tiles import TileLocator
from ..util.logs import getLogger
import numpy as np
from datetime import datetime
import os, time, collections
from floodmap.util.xgeo import XGeo

class WaterMapGenerator(ConfigurableObject):

    def __init__( self, opspecs: Dict, **kwargs ):
        self._opspecs = { key.lower(): value for key,value in opspecs.items() }
        ConfigurableObject.__init__( self, **kwargs )
        self.water_map: xr.DataArray = None
        self.water_probability: xr.DataArray = None
        self.persistent_classes: xr.DataArray = None
        self.lake_mask: xr.DataArray = None
        self.roi_bounds: gpd.GeoSeries = None
        self.mask_value = 5
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
        data_dir = opspec.get('results_dir')
        lake_index = opspec.get('lake_index')
        t0 = time.time()
        cache = kwargs.get('cache', "update")
        thresholds = opspec.get('water_class_thresholds', [ 0.05, 0.95 ] )
        perm_water_mask: xr.DataArray = water_probability > thresholds[1]
        boundaries_mask: xr.DataArray = water_probability > 1.0
        perm_land_mask: xr.DataArray = water_probability < thresholds[0]
        roi_mask: xr.DataArray = (self.water_map >= self.mask_value) | boundaries_mask
        result = xr.where(roi_mask, self.water_map, xr.where(perm_water_mask, 2, xr.where(perm_land_mask, 1, 0)))
        result = result.persist()
        result.name = "Persistent_Classes"
        self.logger.info(f"Done get_persistent_classes in time {time.time() - t0}")
        persistent_class_map = result.assign_attrs( cmap = dict( colors=self.get_water_map_colors() ) )
        if cache in [ True, "update" ]:
            persistent_class_map_file = os.path.join(data_dir, f"lake_{lake_index}_persistent_class_map.nc")
            result = xr.Dataset(dict(persistent_class_map=sanitize(persistent_class_map)))
            result.to_netcdf(persistent_class_map_file)
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
            water = self.floodmap_data.isin([1, 2, 3])
            land = self.floodmap_data.isin([0])
            unmasked = ( self.floodmap_data[0] != self.mask_value )
            water_cnts = water.sum(axis=0)
            land_cnts = land.sum(axis=0)
            visible_cnts = (water_cnts + land_cnts)
            water_probability: xr.DataArray = water_cnts / visible_cnts
            water_probability = water_probability.where( unmasked, 1.01 )
            water_probability.name = "water_probability"
            if cache in [True,"update"]:
                result = xr.Dataset(dict(water_probability=sanitize(water_probability)))
                result.to_netcdf(water_probability_file)
                msg = f"Saved water_probability to {water_probability_file}"
                self.logger.info( msg ); print( msg )
        water_probability = water_probability.persist()
        self.logger.info(f"Done get_water_probability in time {time.time() - t0}")
        return water_probability

    def compute_raw_water_map(self)-> xr.Dataset:
        from floodmap.util.configuration import opSpecs
        water_maps_opspec = opSpecs.get('water_map', {})
        bin_size = water_maps_opspec.get( 'bin_size', 8 )
        threshold = water_maps_opspec.get('threshold', 0.5 )
        da: xr.DataArray = self.floodmap_data[-bin_size:]
        binSize = da.shape[0]
        da0 = da[0].drop_vars( self.floodmap_data.dims[0] )
        masked = da0.isin( [ self.mask_value, self.mask_value+1, self.mask_value+2  ] )
        land = da.isin( [0] ).sum( axis=0 )
        water =  da.isin( [1,2,3] ).sum( axis=0 )
        visible = ( water + land )
        reliability = visible / float(binSize)
        prob_h20 = water / visible
        water_mask = prob_h20 >= threshold
        result =  xr.where( masked, da0, xr.where( water_mask, 2, xr.where( land, 1, 0 ) ) )
        return xr.Dataset( { "water_map": result,  "reliability": reliability } )

    def get_raw_water_map(self, **kwargs):
        # data_array = timeseries of LANCE floodmap data over all years & days configures in specs, cropped to lake bounds
        # this method computes land & water pixels over bins of {bin_size} days using thresholds
        self.logger.info("\n Executing get_water_map ")
        t0 = time.time()
        data_dir = self._opspecs.get('results_dir')
        lake_index = self._opspecs['lake_index']
        water_map_file = os.path.join(data_dir, f"lake_{lake_index}_water_map.nc")
        water_data_file = os.path.join(data_dir, f"lake_{lake_index}_floodmap_data.nc")
        self.floodmap_data.to_netcdf(water_data_file)
        print( f"Saving floodmap data for lake {lake_index} to {water_data_file}")
        cache = kwargs.get( "cache", "update" )
        if cache==True and os.path.isfile( water_map_file ):
            water_map_dset: xr.Dataset = xr.open_dataset(water_map_file)
        else:
            water_map_dset:  xr.Dataset = self.compute_raw_water_map()
            if cache in [True,"update"]:
                water_map_dset.to_netcdf(water_map_file)
                self.logger.info(f"Cached water_map to {water_map_file}")
        self.logger.info( f" Completed get_water_map in {time.time()-t0:.3f} seconds" )
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
        remove_anomalies = kwargs.get( "remove_anomalies", False )
        init_water_map: xr.DataArray = self.spatial_interpolate( opspec, **kwargs ) if remove_anomalies else self.water_map
        pwmap: xr.DataArray = self.temporal_ffill(init_water_map, **kwargs) if ffill else init_water_map
        patched_result: xr.DataArray = pwmap.where( pwmap == self.water_map, pwmap + 2 ) if highlight else pwmap
        return patched_result

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

    def temporal_ffill(self, water_map: xr.DataArray, **kwargs) -> xr.DataArray:
        t0 = time.time()
        nodata_val = self.floodmap_data.attrs['_FillValue']
        water_history_data = self.floodmap_data.where( self.floodmap_data != nodata_val, np.nan )
        interp_water_history: xr.DataArray = water_history_data.ffill( water_history_data.dims[0] )
        interp_water_map: xr.DataArray = interp_water_history[-1,:,:].squeeze( drop = True )
        self.logger.info( f"Done temporal interpolate in time {time.time() - t0}" )
        result =  water_map.where( (water_map > 0), interp_water_map )
        return result.fillna( water_map )

    def time_merge( cls, data_arrays: List[xr.DataArray], **kwargs ) -> xr.DataArray:
        time_axis = kwargs.get('time',None)
        frame_indices = range( len(data_arrays) )
        merge_coord = pd.Index( frame_indices, name=kwargs.get("dim","time") ) if time_axis is None else time_axis
        result: xr.DataArray =  xr.concat( data_arrays, dim=merge_coord )
        return result

    def get_date_from_filename(self, filename: str):
        toks = filename.split("_")
        dateId = toks[3].split(".")[1]
        result = datetime.strptime(dateId[1:], '%Y%j').date()
        return np.datetime64(result)

    def infer_tile_locations(self) -> List[str]:
        if self.lake_mask is not None:
            return TileLocator.infer_tiles_xa(self.lake_mask)
        if self.roi_bounds is not None:
            if isinstance( self.roi_bounds, gpd.GeoSeries ):    rbnds = self.roi_bounds.geometry.boundary.bounds.values[0]
            else:                                               rbnds = self.roi_bounds
            tiles =  TileLocator.get_tiles( *rbnds )
            self.logger.info(f"Processing roi bounds (xmin, xmax, ymin, ymax): {rbnds}, tiles = {tiles}")
            return tiles
        raise Exception( "Must supply either source.location, roi, or lake masks in order to locate region")

    def get_mpw_data(self, **kwargs ) -> Tuple[Optional[xr.DataArray],Optional[ np.array]]:
        self.logger.info( "reading mpw data")
        if self.lake_mask is not None:  self.roi_bounds = self.lake_mask.xgeo.extent()
        else:                           assert self.roi_bounds is not None, "Error, Must specify either lake mask file or roi"
        lake_id = kwargs.get('lake_index')
        print( f"ROI for lake {lake_id}: {self.roi_bounds}" )
        t0 = time.time()
        results_dir = kwargs.get('results_dir')
        download = kwargs.get( 'download', True )
        lake_mask_spec = kwargs.get('lake_masks')
        from .mwp import MWPDataManager
        source_spec = kwargs.get('source')
        data_url = source_spec.get('url')
        path = source_spec.get('path')
        product = source_spec.get('product')
        token = source_spec.get('token')
        collection = source_spec.get('collection')
        locations = source_spec.get( 'location', self.infer_tile_locations() )
        if not locations:
            self.logger.error( "NO LOCATION DATA.  ABORTING")
            return None, None

        dataMgr = MWPDataManager(results_dir, data_url )
        cropped_tiles: Dict[str,xr.DataArray] = {}
        time_values = None
        file_paths = None
        cropped_data = None
        for location in locations:
            try:
                lake_mask_value =  lake_mask_spec.get('mask',0)
                self.logger.info( f"Reading Location {location}" )
                dataMgr.setDefaults(product=product, token=token, path=path, collection=collection, download=download )
                tile_filespec: OrderedDict = dataMgr.get_tile(location)
                file_paths = list(tile_filespec.values())
                time_values = list(tile_filespec.keys())
                tile_raster: xr.DataArray =  XRio.load( file_paths, mask=self.roi_bounds, band=0, mask_value=self.mask_value, index=time_values )
                if self.lake_mask is None:
                    cropped_tiles[location] = tile_raster
                else:
                    lake_mask_interp: xr.DataArray = self.lake_mask.squeeze(drop=True).interp_like( tile_raster[0,:,:] ).fillna( lake_mask_value )
                    tile_mask: xr.DataArray = ( lake_mask_interp == lake_mask_value )
                    tile_mask_data: np.ndarray = np.broadcast_to( tile_mask.values, tile_raster.shape ).flatten()
                    tile_raster_data: np.ndarray = tile_raster.values.flatten()
                    tile_raster_data[ tile_mask_data ] = self.mask_value + 1
                    cropped_tiles[location] = tile_raster.copy( data=tile_raster_data.reshape(tile_raster.shape) )
            except Exception as err:
                for file in file_paths:
                    if not os.path.isfile( file ): self.logger.warning( f"   --> File {file} does not exist!")
                exc = traceback.format_exc()
                msg = f"Error reading mpw data for location {location} \n  Error: {err}: \n{exc}"
                self.logger.error( msg ); print( msg )
                XRio.print_array_dims( file_paths )
        nTiles = len( cropped_tiles.keys() )
        if nTiles > 0:
            self.logger.info( f"Merging {nTiles} Tiles ")
            cropped_data = self.merge_tiles( cropped_tiles)
            cropped_data.attrs.update( roi = self.roi_bounds )
            cropped_data = cropped_data.persist()
        self.logger.info(f"Done reading mpw data for lake {lake_id} in time {time.time()-t0}, nTiles = {nTiles}")
        return cropped_data, time_values

    def merge_tiles(self, cropped_tiles: Dict[str,xr.DataArray] ) -> xr.DataArray:
        lat_bins = {}
        for (key, cropped_tile) in cropped_tiles.items():
            lat_bins.setdefault( key[0:4], [] ).append( cropped_tile )
        concat_tiles = [ self.merge_along_axis( sub_arrays, -2 ) for sub_arrays in lat_bins.values() ]
        result =  self.merge_along_axis( concat_tiles, -1 )
        return result

    def merge_along_axis( self, sub_arrays: List[xr.DataArray], axis: int ) -> xr.DataArray:
        if len( sub_arrays ) == 1:  return sub_arrays[0]
        concat_dim = sub_arrays[0].dims[axis]
        ccoord: xr.DataArray = sub_arrays[0].coords[concat_dim].values
        sub_arrays.sort( reverse = (ccoord[0] > ccoord[-1]), key = lambda x: x.coords[concat_dim].values[0] )
        result: xr.DataArray = xr.concat( sub_arrays, dim = concat_dim )
        return result

    def get_opspec(self, lakeId: str ) -> Dict:
        opspec = self._opspecs.get( lakeId.lower() )
        if opspec is None:
            self.logger.error( "Can't find {lakeId.lower()} in opspecs, available opspecs: {self._opspecs.keys()}" )
            return {}
        merged_opspec: Dict = dict( **self._opspecs.get("defaults",{}) )
        for key,value in opspec.items():
            if key in merged_opspec:    merged_opspec[key] = self.merge_opspec_values( merged_opspec[key], value )
            else:                       merged_opspec[key] = value
        merged_opspec['id'] = lakeId
        return merged_opspec

    def merge_opspec_values( self, value0, value1 ):
        if isinstance( value0, collections.Mapping ):
            return { **value0, **value1 }
        else: return value1

    def get_roi_bounds(self, opspec: Dict ):
        data_dir = opspec.get('data_dir')
        roi = opspec.get('roi', None)
        if roi is not None:
            if isinstance(roi, list):
                self.roi_bounds = [ float(x) for x in roi ]
            elif isinstance(roi, str) and "," in roi:
                self.roi_bounds = [ float(x) for x in roi.split(",") ]
            elif isinstance(roi, str):
                self.roi_bounds: gpd.GeoSeries = gpd.read_file( roi.replace("{data_dir}", data_dir) )
            else:
                raise Exception( f" Unrecognized roi: {roi}")
        else:
            assert self.lake_mask is not None, "Must specify roi to locate lake"
            self.roi_bounds =  TileLocator.get_bounds(self.lake_mask[0])

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
        results_dir = self._opspecs.get('results_dir')
        file_path = f"{results_dir}/lake_{lake_index}_task_report.txt"
        with open( file_path, "w" ) as file:
            file.write( report )

    def generate_lake_water_map(self, lake_index: int, lake_mask: Optional[xr.DataArray], **kwargs) -> Optional[xr.DataArray]:
        skip_existing = kwargs.get('skip_existing', True)
        save_diagnostics = kwargs.get('save_diagnostics', True)
        format = kwargs.get('format','tif')
        results_dir = self._opspecs.get('results_dir')
        patched_water_map_file = f"{results_dir}/lake_{lake_index}_patched_water_map"
        result_file = patched_water_map_file + ".tif" if format ==  'tif' else patched_water_map_file + ".nc"
        if skip_existing and os.path.isfile(result_file):
            msg = f" Lake[{lake_index}]: Skipping already processed file: {result_file}"
            self.logger.info( msg ), print( msg )
            return None
        else:
            self.roi_bounds = kwargs.get( 'roi', None )
            self.lake_mask: xr.DataArray = lake_mask
            self.logger.info(f" --------------------->> Generating result file: {result_file}")
            (self.floodmap_data, time_values) = self.get_mpw_data( **self._opspecs )
            if self.floodmap_data is None:
                msg = f"No water mapping data! ABORTING Lake[{lake_index}]: {self._opspecs}"
                self.logger.warning( msg ); print( msg )
                return None
            self.logger.info( f"process_yearly_lake_masks: water_mapping_data shape = {self.floodmap_data.shape}")
            self.logger.info(f"yearly_lake_masks roi_bounds = {self.roi_bounds}")
            self.get_raw_water_map( time=time_values )
            patched_water_map = self.patch_water_map( self._opspecs, **kwargs )
            patched_water_map.name = f"Lake-{lake_index}"
            result = sanitize( patched_water_map.xgeo.to_utm( [250.0, 250.0] ) )
            self.write_water_area_results( result, patched_water_map_file + ".txt" )
            if format ==  'tif':    result.xgeo.to_tif( result_file )
            else:                   result.to_netcdf( result_file )
            msg = f"Saving results for lake {lake_index} to {result_file}"
            self.logger.info( msg ); print( msg )
            return patched_water_map.assign_attrs( roi = self.roi_bounds )

    def today(self) -> str:
        today = datetime.now()
        day_of_year = today.timetuple().tm_yday
        return f"{day_of_year}:{today.year}"

    def write_water_area_results(self, patched_water_map: xr.DataArray, outfile_path: str,  **kwargs ):
        interp_water_class = kwargs.get( 'interp_water_class', 4 )
        water_classes = kwargs.get('water_classes', [2,4] )
        water_counts, class_proportion = self.get_class_proportion(patched_water_map, interp_water_class, water_classes)
        file_exists = os.path.isfile(outfile_path)
        with open( outfile_path, "a" ) as outfile:
            if not file_exists:
                outfile.write( "date water_area_km2 percent_interploated\n")
            percent_interp = class_proportion.values
            num_water_pixels = water_counts.values
            outfile.write( f"{self.today()} {num_water_pixels/16.0:.2f} {percent_interp:.1f}\n" )
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

    def patch_water_map( self, opspec: Dict, **kwargs ) -> xr.DataArray:
        patched_water_map: xr.DataArray = self.interpolate( opspec, **kwargs ).assign_attrs(**self.water_map.attrs)
        patched_water_map.attrs['cmap'] = dict( colors=self.get_water_map_colors() )
        rv = patched_water_map.fillna( self.mask_value )
#        class_counts = self.get_class_counts( rv.values[0] )
        return rv

    def get_cached_water_map( self, lakeId: str ):
        opspec = self.get_opspec(lakeId.lower())
        self.water_map: xr.DataArray = self.get_raw_water_map(None, opspec, cache=True)
        return self.water_map

    def get_class_proportion(self, class_map: xr.DataArray, target_class: int, relevant_classes: List[int] ) -> Tuple[xr.DataArray,xr.DataArray]:
        sdims = [ class_map.dims[-1], class_map.dims[-2] ]
        total_relevant_population = class_map.isin( relevant_classes ).sum( dim=sdims )
        class_population = (class_map == target_class).sum( dim=sdims )
        return ( total_relevant_population,  ( class_population / total_relevant_population ) * 100 )
