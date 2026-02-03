"""This notebook takes our output AIJ Sinistro light curves and plots them with minimal data cleaning"""
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from lco_functions import file_len, file_load, create_lc

os.chdir(pathlib.Path.cwd())
def set_rcparams():
    """Enforce matplotlib rcparams from a text file."""
    tab = Table.read('../../rcparams.txt', format='csv')
    for i in range(len(tab)):
        try:
            plt.rcParams[tab['key'][i]] = float(tab['val'][i])
        except ValueError:
            plt.rcParams[tab['key'][i]] = str(tab['val'][i])
set_rcparams()

COMMONPATH = '../../data/lco_aumic/lcs_posttwirl/'
paths = sorted([os.path.join(COMMONPATH, specific) for \
                specific in os.listdir(COMMONPATH) if specific.endswith('.xls')])
# output order is B, U, V, gp, ip, rp
lens = [file_len(path) for path in [paths[1], paths[4]]]
datas = []
for file, cleans in zip(paths, [[None, None], [2, lens[0]-13], [None, None], [2, None], [28, lens[1]-1], [1, None]]):
    datum = file_load(file, cleanrange=cleans)
    datas.append(datum)

BDATA, UDATA, VDATA, GDATA, IDATA, RDATA = datas

B_LC, BSNR = create_lc(*BDATA)
U_LC, USNR = create_lc(*UDATA)
V_LC, VSNR = create_lc(*VDATA)
G_LC,  GSNR = create_lc(*GDATA)
I_LC, ISNR = create_lc(*IDATA)
R_LC, RSNR = create_lc(*RDATA)

SNRLIST = [GSNR, RSNR, ISNR, USNR, BSNR, VSNR]
labels = ['g\'', 'r\'', 'i\'', 'U', 'B', 'V']
for vals in zip(SNRLIST, labels):
    snr, label = vals
    print(f"Median SNR for {label}: {np.median(snr):.2f}")

path = pathlib.Path('figures/sinistro/')
if not os.path.exists(path):
    path.mkdir(parents=True, exist_ok=True) 

fig, ax = plt.subplots(figsize=(24, 16))
colors = ['#00A800', '#C40000', '#270C0C', '#491B4F', '#000C74', '#0074A6']
lcs = [G_LC, R_LC, I_LC, U_LC, B_LC, V_LC]
labels = ['g\'', 'r\'', 'i\'', 'U', 'B', 'V']
for lc, color, lab in zip(lcs, colors, labels):
    lc.scatter(ax=ax, c=color, label=lab, s=100, rasterized=True)

ax.legend(fontsize=24, bbox_to_anchor=(0.8, 0.8), markerscale=3, loc='upper left')
ax.set_xlabel('JD-2400000')
ax.set_ylabel('Relative AIJ Flux (scaled to comparison star)')
fig.savefig(path / 'og_sinistro_lcs.png', dpi=300)
plt.clf()
plt.close()
