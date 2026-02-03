"""This program takes our base CHEOPS light curves and plots them with minimal data cleaning"""

import os
import pathlib
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.io import fits
import lightkurve as lk
from matplotlib import gridspec

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

cheops_im = glob('../../data/cheops_aumic/*_im.fits')
cheops_lcs = []
for index, image in enumerate(cheops_im):
    cheops_dat = Table(fits.getdata(image))
    cheops_lcs.append(cheops_dat)

    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

ax1 = fig.add_subplot(spec[0, :])

may_31 = cheops_lcs[:2]
june_8 = cheops_lcs[2:4]

def combine_cheops(samedate_datalist):
    """Combine multiple CHEOPS light curves from the same date into one light curve."""
    combined_time = np.array([])
    combined_flux = np.array([])
    combined_fluxerr = np.array([])
    for data in samedate_datalist:
        combined_time = np.concatenate((combined_time, data['BJD_TIME']-2457000))
        combined_flux =  np.concatenate((combined_flux, data['FLUX']))
        combined_fluxerr = np.concatenate((combined_fluxerr, data['FLUXERR']))
    time = combined_time
    flux = combined_flux
    error = combined_fluxerr
    return lk.LightCurve(time=time, flux=flux, flux_err=error)

path = pathlib.Path('figures/cheops/')
if not os.path.exists(path):
    path.mkdir(parents=True, exist_ok=True) 

combined_cheops_531 = combine_cheops(may_31).normalize()
combined_cheops_68 = combine_cheops(june_8).normalize()

combined_cheops_531.scatter(ax=ax1, label='CHEOPS May 31', color='black', alpha=0.7)
combined_cheops_68.scatter(ax=ax1, label='CHEOPS June 8', color='gray', alpha=0.7)
ax1.set_xlabel('')
ax1.set_ylabel('')
fig.supylabel('Normalized Flux')
fig.supxlabel('JD-2457000')
ax1.legend(loc='best', markerscale=15, fontsize=14)
fig.savefig('figures/cheops/combined_cheops.png', dpi=300)