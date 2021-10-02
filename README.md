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

#### Download archived data
Download script for archived files for tile h20v09:
```
>> cd {results_dir}/h20v09/allData/61/MCDWD_L3_F2_NRT/Recent
>> scp "adaptlogin.nccs.nasa.gov:/att/nobackup/tpmaxwel/data/MCDWD_NRT/h20v09/allData/61/MCDWD_L3_F2_NRT/Recent/MCDWD_L3_F2_NRT.A*.h20v09.061.tif" .
>> scp "adaptlogin.nccs.nasa.gov:/att/nobackup/dslaybac/MCDWD_NRT/MCDWD_L3_NRT_2021250/MCDWD_L3_NRT.A202124*.h20v09.061.hdf" . 
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
#### Evaluate results
```
(floodmap)>> python ./exe/visually_evaluate_workflow.py ~/specs.yml
```
   This script starts up an application to evaluate the results of the workflow.  In the script
are two parameters, *lake_index*, and *day*, which set the lake and date thate are being examined.
In the gui there are four windows:
* MWP Data:  This window shows the *nbin* floodmap data layers that are used to compute the 
              water mapping product for the selected day, covering the range [day-nbin, day]
* Water Map:   The water mapping product for the selected day.
* Water Counts:  The count of water observations in each cell over the floodmap data layers.
* Land Counts:   The count of land observations in each cell over the floodmap data layers.

     There are several interactive tools available:
* Synchronized pan/zoom: Use the toolbar to activate these modes.
* Synchronised Cursor: Note that the cursor will not appear in pan or zoom modes.  Use the 
                toolbar to toggle out of these modes in order to show the cursor.
* Key maps: 'f'/'b': Step forward/backward in mpw data series.
* Mouse click callback:  Prints land/water counts for selected cell in shell.

#### Documentation
The floodmap user guide is found at `./docs/floodmap.pdf`