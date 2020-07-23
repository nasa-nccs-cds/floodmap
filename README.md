# floodmap
This project computes surface water extents from lake shapefile boundary maps.  It uses the NRT Global Flood Mapping products produced from the LANCE-MODIS data processing system at NASA Goddard to compute the probability of water in each spatial cell within each shapefile boundary map.   It then thresholds this probability to produce water mask files.

## Installation

#### Build Conda Env

If Anaconda is not yet installed in your environment, see: https://www.anaconda.com/products/individual for installation instructions.
   
```
>> conda create --name floodmap python=3.6
>> conda activate floodmap
(floodmap)>> conda install -c conda-forge xarray numpy zlib rioxarray shapely regionmask pandas bottleneck geopandas utm
(floodmap)>> pip install wget
```
#### Install floodmap
```
(floodmap)>> git clone https://github.com/nasa-nccs-cds/floodmap.git
(floodmap)>> cd floodmap
(floodmap)>> python setup.py install
```
#### Configure execution
```
(floodmap)>> cp ./specs/sample-specs.yml ~/specs.yml
(floodmap)>> emacs -nw ~/specs.yml
```
#### Run floodmap
```
(floodmap)>> python ./exe/production.py ~/specs.yml
```
#### Documentation
The floodmap user guide is found at `./docs/floodmap.pdf`