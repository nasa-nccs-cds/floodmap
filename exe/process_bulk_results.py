from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import xarray as xa
import numpy as np
from floodmap.util.configuration import opSpecs
from datetime import datetime

class FloodmapProcessor:
    calendar = 'standard'
    units = 'days since 1970-01-01 00:00'
    results_dir = opSpecs.get('results_dir')

    def __init__(self):
        self._datasets = None

    @classmethod
    def results_file( cls, fmversion: str):
        result_name = f"floodmap_comparison_{fmversion}"
        return f"{cls.results_dir}/{result_name}.nc"

    @classmethod
    def pct_diff( cls,  x0: float, x1: float ) -> float:
        return (abs( x1-x0 ) * 100) / min(x0,x1)

    @classmethod
    def get_timestamp( cls, tstr: str, fmversion: str ) -> datetime:
        if fmversion == "nrt": (m, d, y) = tstr.split("-")
        elif fmversion == "legacy": (y, m, d) = tstr.split("-")
        else: raise Exception( f"Unrecognized fmversion: {fmversion}")
        return datetime(int(y), int(m), int(d))

    def get_datasets(self)-> Dict[str,xa.Dataset]:
        if self._datasets is None:
            self._datasets = { fmversion: xa.open_dataset( self.results_file(fmversion) ) for fmversion in ["legacy", 'nrt'] }
        return self._datasets

    def get_vars(self, name: str )-> Dict[str,xa.DataArray]:
        dsets: Dict[str, xa.Dataset] = self.get_datasets()
        return  { fmversion: dsets[fmversion].data_vars[ name ] for fmversion in [ "legacy", 'nrt' ] }

    # def get_means(self):
    #     water_area_means = {}
    #     interp_area_means = {}
    #     pct_interp_means = {}
    #     dsets = self.get_datasets()
    #     for fmversion in [ "legacy", 'nrt' ]:
    #         water_area: xa.DataArray = dsets[fmversion].data_vars['water_area']
    #         water_area_mean = water_area.mean(skipna=True).values.tolist()
    #         water_area_means[fmversion] = water_area_mean
    #         pct_interp_array: xa.DataArray = dsets[fmversion].data_vars['pct_interp']
    #         interp_area_means[fmversion] = ( pct_interp_array * water_area ).mean(skipna=True).values.tolist()
    #         pct_interp_means[fmversion] = pct_interp_array.mean(skipna=True).values.tolist()
    #     print(f"\nMeans: {water_area_means}")
    #     print(f"Pct DIFF: {self.pct_diff(*list(water_area_means.values())):.2f} %")
    #     print(f"\nPct Interp: {pct_interp_means}")
    #     print(f"Pct DIFF: {self.pct_diff(*list(pct_interp_means.values())):.2f} %")
    #     print(f"\nInterp Area: {interp_area_means}")
    #     print(f"Pct DIFF: {self.pct_diff(*list(interp_area_means.values())):.2f} %")
    #     return dict( water_area=water_area_means, interp_area=interp_area_means, pct_interp=pct_interp_means )

    def get_interp_diff(self):
        water_vars: Dict[str, xa.DataArray] = self.get_vars('water_area')
        interp_vars: Dict[str, xa.DataArray] = self.get_vars('pct_interp')
        lake_interp_means = {}
        for fmversion in ["legacy", 'nrt']:
            water_var: xa.DataArray = water_vars[fmversion]
            interp_var: xa.DataArray = interp_vars[fmversion]
            interp_area: xa.DataArray = (interp_var * water_var) / 1600
            lake_interp_means[fmversion] = interp_area.mean(axis=0, skipna=True)
        interp_diff =  lake_interp_means['nrt'] - lake_interp_means["legacy"]
        return interp_diff.dropna(dim=interp_diff.dims[0])

if __name__ == '__main__':
    fmp = FloodmapProcessor()
    interp_diff = fmp.get_interp_diff()
    print(interp_diff.data)




    # for ilake in range(4):
    #     lake_index = dset.lake.values[ilake]
    #     print(f"> lake {lake_index}, water_area shape = {water_area.shape}:")
    #     strdata = " ".join( [ f"{v:.1f}" for v in water_area.data[:,ilake].tolist() ] )
    #     print( "   " + strdata )

