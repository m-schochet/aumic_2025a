"""Code to find and save the image files from LCO that were 
ignored in creating the Sinistro light curves with AIJ"""

import os
from lco_functions import file_len, file_load, find_lost

COMMONPATH = '../../data/lco_aumic/lcs_posttwirl/'

savepaths = sorted([os.path.join(COMMONPATH, specific) for \
                specific in os.listdir(COMMONPATH) if specific.endswith('.xls')])
# output order is B U, V, gp, rp, ip

datas = []
for file in savepaths:
    datum = file_load(file, full=True)
    datas.append(datum)
    
BDATA, UDATA, VDATA, GDATA, IDATA, RDATA = datas

""" The `find_lost` function returns bad_files/*.txt files 
for each filter to note which Sinistro images were ignored.
These images were not either removed from the AIJ light curve 
or caused AIJ to throw an error. We do not speculate as to why 
a specific file is considered `bad` 

Note that this code requires a drive that holds all the original 
.fits files inside to access"""

find_lost(IDATA, "ip", 'sinistro')
find_lost(GDATA, "gp", 'sinistro')
find_lost(RDATA, "rp", 'sinistro')
find_lost(UDATA, "U", 'sinistro')
find_lost(BDATA, "B", 'sinistro')
