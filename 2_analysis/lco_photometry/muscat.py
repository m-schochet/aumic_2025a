"""No preprocessing LCO MuSCAT light curve creation and plotting."""
from glob import glob
import matplotlib.pyplot as plt
import pathlib
import os
from astropy.table import Table
from astropy.io import ascii
from lco_functions import muscat_lks


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

COMMONPATH = '../../data/lco_aumic/'
G, R, I, Z = [sorted(glob(os.path.join(COMMONPATH, FIL))) for FIL in ['muscat_gp*', 'muscat_rp*', 'muscat_ip*', 'muscat_zs_*']]
removals = [[0, 43, 1, 1, 0, 0],  [0, 1, 1, 0, 0, 0], [0, 0, 0, 700, 0, 0],  [0, 0, 12, 0, 0, 0]]
lightcurves = []
for vars in zip(['G', 'R', 'I', 'Z'], [G, R, I, Z], removals):
    filt, filelist, removelist = vars
    filt_objs = [ascii.read(f) for f in filelist]
    [filt_objs[i] == filt_objs[i].sort(keys='rel_flux_T1') for i in range(len(filt_objs))]
    for index, j in enumerate(removelist):
        filt_objs[index] = filt_objs[index][j:]
    lightcurves.append(muscat_lks(filt_objs, normalize=True))
GPLC, RPLC, IPLC, ZSLC = lightcurves

path = pathlib.Path('figures/muscat/')
if not os.path.exists(path):
    path.mkdir(parents=True, exist_ok=True)


fig, ax = plt.subplots(2, 2, figsize=(20, 12), layout='constrained')
axs = ax.flatten()
for i, (lc, color, title) in enumerate(zip([GPLC, RPLC, IPLC, ZSLC], ['green', 'orange', 'red', 'maroon'], ['g\'', 'r\'', 'i\'', 'z\''])):
    lc.scatter(ax=axs[i], color=color, rasterized=True)# marker='o', markersize=4, label=title)
    axs[i].set_title(f'{title}', fontweight='bold', fontsize=40)
    axs[i].set_xlabel('', fontsize=20)
    axs[i].set_ylabel('', fontsize=20)
    axs[i].label_outer()
fig.supxlabel('Time (JD - 2457000)', fontsize=30)
fig.supylabel('Normalized Flux', fontsize=30)
fig.savefig('figures/muscat/muscat_lcs.png', dpi=300)
plt.clf()
plt.close(fig)
