"""Make Figure 1 from Schochet & Feinstein (in prep.)"""

from glob import glob
import pathlib
from math import pi
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import lightkurve as lk
from astropy.table import Table
from astropy.io import fits, ascii as asc
from scipy.optimize import curve_fit
from lco_functions import file_len, file_load, create_lc, muscat_lks

os.chdir(pathlib.Path.cwd())
def set_rcparams():
    """Set Matplotlib.rcparam data"""
    tab = Table.read('../../rcparams.txt', format='csv')
    for i in range(len(tab)):
        try:
            plt.rcParams[tab['key'][i]] = float(tab['val'][i])
        except ValueError:
            plt.rcParams[tab['key'][i]] = str(tab['val'][i])
set_rcparams()

HSTDAYS = [(2460809.7770374413, 2460810.10588031), (2460818.233912621, 2460818.5666751266), (2460826.4921415113, 2460826.8228429067),
(2460835.2038892913, 2460835.5353429066)]

CHEOPSDAYS = [(2460827.4291018597, 2460827.738845379), (2460826.6806421145, 2460827.381607217),
(2460834.964947009, 2460835.133426169), (2460835.1618256704, 2460835.817190319)]

JWSTDAYS = [(2460818.3330352814, 2460818.7497526538), (2460835.259565584, 2460835.676293359),
(2460809.8923711404, 2460810.3091083206), (2460826.7961735446, 2460827.21288887)]   

TESS95 = [(2460881, 2460907)]

# Sinistro Data processing

# Unbinned Sinistro (for R band)
COMMONPATH = '../../data/lco_aumic/lcs_posttwirl/'
paths = sorted([os.path.join(COMMONPATH, specific) for \
                specific in os.listdir(COMMONPATH) if specific.endswith('.xls')])

lens = [file_len(path) for path in [paths[1], paths[4]]]
datas = []
for file, cleans in zip(paths, [[None, None], [2, lens[0]-13], \
                                [None, None], [2, None], \
                                [28, lens[1]-1], [1, None]]):
    datum = file_load(file, cleanrange=cleans)
    datas.append(datum)

BDATA, UDATA, VDATA, GDATA, IDATA, RDATA = datas

RPUNBINNED, RSNR = create_lc(*RDATA)
RPUNBINNED = RPUNBINNED.normalize()

## Binned Sinistro
RPUNFOLDED = pd.read_csv('../../data/lco_aumic/binned_sinistro/rp.csv')
RPUNFOLDEDLC = lk.LightCurve(time=RPUNFOLDED['time']-57000, \
                             flux=RPUNFOLDED['flux'], \
                             flux_err=RPUNFOLDED['flux_err']).normalize()

## Folded Sinistro
FOLDEDPATH = '../../data/lco_aumic/folded_sinistro/'
paths = sorted([os.path.join(FOLDEDPATH, specific) for \
                specific in os.listdir(FOLDEDPATH) if specific.endswith('.csv')])
foldedlcs = [pd.read_csv(path) for path in paths]

B, U, V, GP, IP, RP = foldedlcs
sinistrolcs = []

for band in foldedlcs:
    sinistrolcs.append(lk.LightCurve(time=band['time'], \
                                     flux=band['flux'], \
                                     flux_err=band['flux_err']).normalize())
BLC, ULC, VLC, GPLC, IPLC, RPLC = sinistrolcs

"""MuSCAT data processing"""

MUSCATPATH = '../../data/lco_aumic/'
G, R, I, Z = [sorted(glob(os.path.join(MUSCATPATH, FIL))) for \
              FIL in ['muscat_gp*', 'muscat_rp*', 'muscat_ip*', 'muscat_zs_*']]
removals = [[0, 43, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0], \
            [0, 0, 0, 700, 0, 0], [0, 0, 12, 0, 0, 0]]
lightcurves = []
for vari in zip(['G', 'R', 'I', 'Z'], [G, R, I, Z], removals):
    filt, filelist, removelist = vari
    filt_objs = [asc.read(f) for f in filelist]
    [filt_objs[i] == filt_objs[i].sort(keys='rel_flux_T1') for i in range(len(filt_objs))]
    for index, j in enumerate(removelist):
        filt_objs[index] = filt_objs[index][j:]
    lightcurves.append(muscat_lks(filt_objs, normalize=True))
GPLC_MUSC, RPLC_MUSC, IPLC_MUSC, ZSLC_MUSC = lightcurves


"""This code utilizes the output TESS 120s light curve 
after assigning a stella flare probabilities to each data point. 
This requires running the 2_analysis/tess/stella_probability.ipynb
notebook first to generate the probabilities."""

stella_data = pd.read_csv('../tess/stella/stella_lc.csv')
stella_lc_og = lk.LightCurve(flux=stella_data['flux'], \
                             time=stella_data['time'], \
                             flux_err=stella_data['flux_err'])

with open('../tess/stella/stella_probabilities.npy', 'rb') as f:
    avg_pred = np.load(f)

stella_lc_og.add_column(avg_pred, name='stella_prob')
stella_lc_folded = stella_lc_og.copy().fold(4.86, epoch_time=2400000, epoch_phase=3)

og_params = [np.ascontiguousarray(param, dtype=np.float64) for param in \
             [stella_lc_og.time.value, stella_lc_og.flux, \
              stella_lc_og.flux_err * stella_lc_og.stella_prob, \
              stella_lc_og.flux_err]]
folded_params = [np.ascontiguousarray(param, dtype=np.float64) for param in \
                 [stella_lc_folded.time.value, stella_lc_folded.flux, \
                  stella_lc_folded.flux_err * stella_lc_folded.stella_prob, \
                  stella_lc_folded.flux_err]]

x_og, y_og, yerr_og_scaled, yerr_og_not = og_params
x, y, yerr_scaled, yerr = folded_params

"""Now we'll make the 2-sine wave model to our stella TESS light curve"""

parameters_2term = [np.pi/70, np.max(y) - np.min(y), 3, (np.max(y) - np.min(y))/2, 3, np.max(y)]

def model_sine2(xes, omega, amp, phase, amp2, phase2, offset):
    """A 2-sine wave function with parameters for variable phase, total offset, and frequency.
    The second sine wave has a harmonic frequency (frequency/2).

    Args:
        xes (list): List of x-values to generate the 2-sine wave model over.
        omega (float): Angular frequency of the first sine wave.
        amp (float): Amplitude of the first sine wave.
        phase (float): Phase shift of the first sine wave.
        amp2 (float): Amplitude of the second sine wave (harmonic).
        phase2 (float): Phase shift of the second sine wave.
        offset (float): Constant offset added to the entire model.

    Returns:
        list: f(x), the value of the 2-sine wave model at each x-value in X.
    """
    
    return amp * np.sin(2*pi* omega * xes + phase) + amp2 * np.sin(pi * omega * xes + phase2) + offset
popt, pcov = curve_fit(model_sine2, x.tolist(), y,
                  p0=parameters_2term,
                  maxfev=50000)

fig = plt.figure(figsize=(16, 8))
ax = plt.subplot(1, 1, 1)
ax.plot(x, y, label='TESS', color='k')
ax.plot(x, model_sine2(x, *popt), 'g--', label='2 sine Fit')

print(f'Parameters for 2-sine fit are: \nω={popt[0]:.8} \
       \n(sin of f) A1={popt[1]:.8f}, φ1={popt[2]:.8f} \
       \n(sin of f/2) A2={popt[3]:.8f}, φ2={popt[4]:.8f}. \
       \nC={popt[5]:.8f}')
print('\nFunctional form of 2-sine fit is:\n Flux = A1*sin(ω*t + φ1) + A2*sin(ω/2*t + φ2) + C')
ax.set_ylabel('TESS Normalized Flux')
ax.set_xlabel('Phase')
ax.legend(markerscale=10)


path = pathlib.Path('figures')
if not os.path.exists(path):
    path.mkdir(parents=True, exist_ok=True) 

path2 = pathlib.Path('figures/tess')
if not os.path.exists(path2):
    path2.mkdir(parents=True, exist_ok=True) 

fig.savefig('figures/tess/2sine_fit_tess.png', dpi=300)
plt.clf()
plt.close()

fig = plt.figure(constrained_layout=True, figsize=(30, 16), facecolor='w')
spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

ax1 = fig.add_subplot(spec[0, :])
ax2 = fig.add_subplot(spec[1, 0], sharey=ax1)
ax5 = fig.add_subplot(spec[2, 0], sharey=ax1)

ax3 = fig.add_subplot(spec[1, 1], sharey=ax1)
ax4 = fig.add_subplot(spec[1, 2], sharey=ax1)
ax6 = fig.add_subplot(spec[2, 1], sharey=ax1)
ax7 = fig.add_subplot(spec[2, 2], sharey=ax1)

# Plot the bands for HST, CHEOPS, and JWST
for num, info in enumerate(zip([HSTDAYS, CHEOPSDAYS, JWSTDAYS],
                               ["#002FFF", "#C6C634", "#FF0000"],
                               ['HST', 'CHEOPS', 'JWST'])):
    day_range, color, label = info
    for num, (start, end) in enumerate(day_range):
        if num==0:
            ax1.axvspan(start-2457000, end-2457000, color=color, alpha=0.3, label=label)
        else:
            ax1.axvspan(start-2457000, end-2457000, color=color, alpha=0.3)

# Now the TESS Model and data
model_times = np.linspace(np.min(RPUNFOLDEDLC.time.value), \
                          np.min(stella_lc_og.time.value), 20000)
model = ax1.plot(model_times, model_sine2(model_times, *popt), \
                 c='b', label='TESS 2-sine Fit', rasterized=True)
stella_lc_og.scatter(ax=ax1, s=100, c="#353434", \
                     label='TESS S95', alpha=0.25, rasterized=True)

# LCO data
RPUNFOLDEDLC.scatter(ax=ax1, s=100,  label='Binned r\' (LCO Sinistro)', c="#D0B658", rasterized=True)
RPUNBINNED.scatter(ax=ax1, s=100, label='Unbinned r\'', alpha=0.5, c="#D0B658", rasterized=True)
ZSLC_MUSC.scatter(ax=ax1, s=100, label='z\' (LCO MuSCAT)', c="#BA0016", rasterized=True)

# Now the lower subplots with the LCO Sinistro Data
lowerplots = [curve.errorbar(ax=axs, markersize=5, fmt='o', elinewidth=2, capsize=5,
                label=label, c=color, rasterized=True) for curve, axs, label, color
                in zip([ULC, BLC, GPLC, VLC, RPLC, IPLC], [ax2, ax3, ax4, ax5, ax6, ax7],\
                       ['U (LCO Sinistro)', 'B (LCO Sinistro)', 'g\' (LCO Sinistro)',\
                        'V (LCO Sinistro)', 'r\' (LCO Sinistro)', 'i\' (LCO Sinistro)'],\
                        ["#004BA8", "#6CCFF6", '#0B6E4F', "#000000", "#D0B658", '#764248'])]

# Matplotlib formatting
legend = ax1.legend(fontsize=18, facecolor='w', labelcolor='k', ncols=9,
                    bbox_to_anchor=(0.5, 1.25), loc='upper center', markerscale=5)
ax1.set_xlim(np.min(RPUNFOLDEDLC.time.value)-0.5, np.max(stella_lc_og.time.value)+0.5)
[ax.legend(fontsize=25, facecolor='w', labelcolor='k') for ax in [ax2, ax3, ax4, ax5, ax6, ax7]]

ylabs = [ax.set_ylabel('') for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]]
xlabs = [ax.set_xlabel('') for ax in [ax2, ax3, ax4, ax5, ax6, ax7]]
ax1.set_xlabel('Time (BJD-2457000)')

fig.supylabel('Normalized Flux')
fig.supxlabel('Phase')

for num, ax in enumerate(plt.gcf().axes):
    if num==0:
        continue
    ax.label_outer()
fig.savefig('figures/fig1_lco_aumic_photometry.png', dpi=300)
plt.clf()
plt.close()
