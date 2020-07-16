# floodmap
This project computes surface water extents from lake shapefile boundary maps.  It uses the NRT Global Flood Mapping products produced from the LANCE-MODIS data processing system at NASA Goddard to compute the probability of water in each spatial cell within each shapefile boundary map.   It then thresholds this probability to produce water mask files.

## Installation

#### Build Conda Env
```
>> conda create --name floodmap
>> conda activate floodmap
(geoproc)>> conda install -c conda-forge xarray, numpy, rioxarray, shapely, regionmask, pandas, geopandas, wget 
(geoproc)>> pip install wget
```
#### Install floodmap
```
(geoproc)>> git clone https://github.com/nasa-nccs-cds/floodmap.git
(geoproc)>> cd floodmap
(geoproc)>> python setup.py install
```
#### Configure execution
```
(geoproc)>> cp floodmap/specs/sample-specs.yml ~/specs.yml
(geoproc)>> emacs ~/specs.yml
```
#### Run floodmap
```
(geoproc)>> python floodmap/exe/production.py ~/specs.yml
```