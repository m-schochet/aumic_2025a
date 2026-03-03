# The overall analysis and processing of the light curves requires a few things:

### All files in the `data` directory, including:
#### a) `cheops_aumic` which had 4 light light curve `.fits` files
#### b) `lco_aumic` which holds 3 folders. 2 of them are Sinistro `lcs_posttwirl` & `lcs_pretwirl` which contain `.xls` and `.tbl` files for the light curves respectively. Another `muscat` folder holds `muscat*.xls` files 

#####   note: these files must be processed by the `1_dataprep` script if running from the beginning off the LCO archive
#### c) `tess` which most significantly holds a `.fits` 120s lightcurve

### Ensuring these files on your machine is necessary for the scripts to run

# You can run the `process_photometry.py` script which will run all the jobs for you.
