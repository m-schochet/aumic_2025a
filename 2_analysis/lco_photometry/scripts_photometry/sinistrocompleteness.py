""" Code to assess the completeness of Sinistro light curves 
generated with AIJ from total number of LCO images"""

import os
from lco_functions import file_load, create_lc

COMMONPATH = '../../data/lco_aumic/lcs_posttwirl/'
paths = sorted([os.path.join(COMMONPATH, specific) for \
                specific in os.listdir(COMMONPATH) if specific.endswith('.xls')])
# output order is B, U, V, gp, ip, rp

datas = []
for file in paths:
    datum = file_load(file)
    datas.append(datum)
BDATA, UDATA, VDATA, GDATA, IDATA, RDATA = datas
B_LC, BSNR = create_lc(*BDATA)
U_LC, USNR = create_lc(*UDATA)
V_LC, VSNR = create_lc(*VDATA)
G_LC, GSNR = create_lc(*GDATA)
I_LC, ISNR = create_lc(*IDATA)
R_LC, RSNR = create_lc(*RDATA)

U_LC = U_LC[U_LC['flux'] < 16] # We caught a flare in on of our images. lets trim that to not mess us up.

def complete(og_num, current):
    """Calculate completeness as a percentage"""
    return current*100/og_num

og_gp, og_ip, og_rp, og_U, og_B, og_V = 494, 500, 483, 414, 125, 141
list_ognums = [og_U, og_B, og_V, og_gp, og_rp, og_ip]
lengths = [len(U_LC), len(B_LC), len(V_LC), len(G_LC), len(R_LC), len(I_LC)]

completenesses = [complete(og_num, length) for og_num,length in zip(list_ognums, lengths)]

for comp, filters, length, listog in zip(completenesses, ['U', 'B', 'V','g\'', 'i\'', 'r\''], lengths, list_ognums):
    print('The completeness of our processed Sinistro light curve in the',\
           filters, 'filter is:', f"{comp:.2f}. \nWe used {length} out of {listog} data points.")
