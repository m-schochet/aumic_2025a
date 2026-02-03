"""Code to get flare probabilities for the TESS S95 light curve of AU Mic 

Note: the conda environment used to run this job requires a working installation of stella.
We recommened using a new environment with new installs of required packages to avoid conflicts."""

import os
import sys
import pathlib
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import astropy.units as u
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import stella
from tqdm import tqdm_notebook

#20s cadence
tess_lc20 = lk.read('../../data/tess/20s/tess2025206162959-s0095-0000000441420236-0292-a_fast-lc.fits')

#120s cadence
tess_lc120 = lk.read('../../data/tess/120s/tess2025206162959-s0095-0000000441420236-0292-s_lc.fits')

ds = stella.DownloadSets()
ds.download_models()
OUT_DIR = '/Users/mschochet/Desktop/stella_results/' #CHANGE THIS to your desired output directory

cnn = stella.ConvNN(output_dir=OUT_DIR)
cnn.predict(modelname=str(ds.models[0]),
            times=tess_lc120.time.value,
            fluxes=tess_lc120.flux,
            errs=tess_lc120.flux_err)
single_pred = cnn.predictions[0]

preds = np.zeros((len(ds.models),len(cnn.predictions[0])))

for i, m in enumerate(ds.models):
    cnn.predict(modelname=m,
                times=tess_lc120.time.value,
                fluxes=tess_lc120.flux,
                errs=tess_lc120.flux_err)
    preds[i] = cnn.predictions[0]

avg_pred = np.nanmedian(preds, axis=0)

stella_lc = lk.LightCurve(flux=cnn.predict_flux[0], time = cnn.predict_time[0], flux_err = cnn.predict_err[0])
stella_lc.to_csv('stella/stella_lc.csv', overwrite=True) 
with open('stella/stella_probabilities.npy', 'wb') as f:
    np.save(f, avg_pred)

with open('stella/stella_probabilities.npy', 'rb') as f:
    avg_pred = np.load(f)
stella_lc.add_column(avg_pred, name='probs')


path = pathlib.Path('figures/')
if not os.path.exists(path):
    path.mkdir(parents=True, exist_ok=True) 

fig, (ax1, ax2) = plt.subplots(figsize=(14,8), nrows=2,
                               sharex=True, sharey=True)
im = ax1.scatter(cnn.predict_time[0], cnn.predict_flux[0],
            c=avg_pred, vmin=0, vmax=1, rasterized=True)
ax2.scatter(cnn.predict_time[0], cnn.predict_flux[0],
            c=single_pred, vmin=0, vmax=1, rasterized=True)
ax2.set_xlabel('Time [BJD-2457000]')
ax2.set_ylabel('Normalized Flux', y=1.2)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.81, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Probability')

ax1.set_title('Averaged Predictions')
ax2.set_title('Single Model Predictions')

plt.subplots_adjust(hspace=0.4)
fig.savefig('figures/tess_stella_probabilities.png', dpi=300)
