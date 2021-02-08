# floodmap
This project computes surface water extents from lake shapefile boundary maps.  It uses the NRT Global Flood Mapping products produced from the LANCE-MODIS data processing system at NASA Goddard to compute the probability of water in each spatial cell within each shapefile boundary map.   It then thresholds this probability to produce water mask files.


### Install floodmap

Retrieve the container from NASA NOMAD Large File Transfer Service (forward invite) and download into an INSTALLDIR on your local server.  As long as Singularity is available, no other configuration is required.

### Configure execution

1. Copy the [sample-specs](https://github.com/nasa-nccs-cds/floodmap/blob/v0/specs/sample_specs-v0.yml) file to a local location (e.g. ~/specs.yml)
2. Edit the file to reflect your local confioguration.   Explanations of the various parameters can be found in the user guide (see below).

### Run floodmap
```
>> cd INSTALLDIR
>> singularity run ilab-floodmap-1.0.0.simg python /usr/local/floodmap/exe/production.py ~/specs.yml
```
### Documentation
The floodmap user guide is found at [nasa-nccs-cds/floodmap](https://github.com/nasa-nccs-cds/floodmap/blob/master/docs/floodmap.pdf)

### Notes

- The output is saved to the **results_dir** directory specified in the specs.yml file.
- Floodmap produces two result files for each lake:
  1. Text file with the lake extent output
  2. Geotiff file showing classifications for diagnostic purposes.  The labels for these clases are show below.
- Floodmap will not reprocess lakes for which results already exist.  If you wish to reprocess certain lakes then
  you must delete the result files for those lakes from the results directory.

#### Class Labels for Diagnostic Files
    0. nodata (cloud obscured)
    1. land
    2. water
    3. interploated land
    4. interploated water
    5. mask