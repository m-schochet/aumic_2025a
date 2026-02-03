""" Code to assess the completeness of Sinistro light curves 
generated with AIJ from total number of LCO images"""

from glob import glob
import os
from lco_functions import file_load, file_len

COMMONPATH = '../../data/lco_aumic/lcs_posttwirl/'
paths = sorted([os.path.join(COMMONPATH, specific) for \
                specific in os.listdir(COMMONPATH) if specific.endswith('.xls')])
# output order is B, U, V, gp, ip, rp
lens = [file_len(path) for path in [paths[1], paths[4]]]
datas = []
for file, cleans in zip(paths, [[None, None], [None, None], [2, lens[1]-13],[2, None], [28, lens[0]-1], [1, None]]):
    datum = file_load(file, cleanrange=cleans, full=True)
    datas.append(datum)

BDATA, UDATA, VDATA, GDATA, IDATA, RDATA = datas

def complete(og_num, current):
    return (current*100/og_num)

og_gp, og_ip, og_rp, og_U, og_B, og_V = 494, 500, 483, 414, 125, 141
list_ognums = [og_U, og_B, og_V, og_gp, og_rp, og_ip]
lengths = [len(UDATA), len(BDATA), len(VDATA), len(GDATA), len(RDATA), len(IDATA)]

completenesses = [complete(og_num, length) for og_num,length in zip(list_ognums, lengths)]

for complete, filter in zip(completenesses, ['U', 'B', 'V','g\'', 'i\'', 'r\'']):
    print('The completeness of our processed Sinistro light curve in the', filter, 'filter is:', f"{complete:.2f}%")