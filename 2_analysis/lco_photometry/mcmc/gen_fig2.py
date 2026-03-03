from astroML.datasets import fetch_sdss_filter
from specutils.spectra import Spectrum
from IPython.display import display, Math
from specutils.fitting import fit_generic_continuum
from chromatic import *
import warnings
from corner import corner
from glob import glob
import os
import pickle
import emcee
import matplotlib.pyplot as plt
import lightkurve as lk
import numpy as np
from speclite import filters
from astropy.constants import h, c, k_B
import astropy.units as u
import numpy as np
import pathlib
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', default = 'n', type = str, help = 'Do you want to fit print values as they are calculated (y/n) ? (Default is n)')
args = parser.parse_args()
verbosity = args.verbose

labels = ['T_{amb}', 'T_{spot}', 'f_{spot}', '\Delta f_{spot}']

os.environ["OMP_NUM_THREADS"] = "8"

if not os.path.exists("joint_ind_muscat_sinistro"):
    os.makedirs("joint_ind_muscat_sinistro/", exist_ok=True)
    if not os.path.exists("joint_ind_muscat_sinistro/muscat"):
        os.makedirs("joint_ind_muscat_sinistro/muscat", exist_ok=True)
    if not os.path.exists("joint_ind_muscat_sinistro/sinistro"):
        os.makedirs("joint_ind_muscat_sinistro/sinistro", exist_ok=True)

path = "../../../data/tessmcmc"
pathlist = sorted(glob(os.path.join(path, "*")))

os.chdir(pathlib.Path.cwd())
def set_rcparams():
    """Set Matplotlib.rcparam data"""
    tab = Table.read('../../../rcparams.txt', format='csv')
    for i in range(len(tab)):
        try:
            plt.rcParams[tab['key'][i]] = float(tab['val'][i])
        except ValueError:
            plt.rcParams[tab['key'][i]] = str(tab['val'][i])
set_rcparams()

with open(pathlist[1], "rb") as f:
    state = pickle.load(f)
    percentiles = np.percentile(state, [50], axis=0)
    ndim = state.shape[1]
    lister = []
    for i in range(ndim):
        p50 = percentiles[:, i]
        lister.append(p50)
    freq, a1given, a2given, phi_1, phi_2 = [val[0] for val in lister]

with open(f'joint_contrasts/muscat/mcmc_state_job_joint.pkl', 'rb') as f:
    samples_muscat = pickle.load(f)

with open(f'joint_contrasts/sinistro/mcmc_state_job_joint.pkl', 'rb') as f:
    samples_sinistro = pickle.load(f)

stella_path = os.path.join('../../tess/stella/stella_lc.csv')
stella_data = pd.read_csv(stella_path)

stella_lc_og = lk.LightCurve(flux=stella_data['flux'], time=stella_data['time'], flux_err=stella_data['flux_err'])
parameterlist = [stella_lc_og.time.value, stella_lc_og.flux, stella_lc_og.flux_err]

with open('../../tess/stella/stella_probabilities.npy', 'rb') as f:
    stella_probs = np.load(f)

stella_lc_og['probs'] = stella_probs
stella_lc_og['flux_err'] /= np.median(stella_lc_og['flux_err'])
stella_lc_og['flux_err'] *= stella_lc_og['probs']

bandpass=np.linspace(3800.,10000.,400)

sdss_responses = filters.load_filters('sdss2010-*')
response_g = sdss_responses[1].interpolator(bandpass)
response_r = sdss_responses[2].interpolator(bandpass)
response_i = sdss_responses[3].interpolator(bandpass)
response_z = sdss_responses[4].interpolator(bandpass)


c_gband_sin = 0
c_rband_sin = 0
c_iband_sin = 0

e_g_sini = 0
e_r_sini = 0
e_i_sini = 0

for filt in ['g', 'r', 'i']:
    mod=79
    with open(f'sinistro_runs/mcmc_state_job{mod}_{filt}.pkl', 'rb') as f:
        samples = pickle.load(f)
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        ndim = samples.shape[1]
        lister = []
        errlist = []
        for i in range(ndim):
            p14, p50, p86 = percentiles[:, i]
            if verbosity == 'y':
                print(f"Sinistro, filter {filt}, A_{i}: {p50:.5f} (+{p86-p50:.10f} / -{p50-p14:.10f})")
            lister.append(p50)
            errlist.append(np.mean([p86-p50, p50-p14]))
    a1, a2, cons = lister
    err1, errs2, conserrs = errlist
    cos_phases = np.cos(phi_1-(2*phi_2))
    if filt=='g':
        c_gband_sin = np.sqrt(a1**2 + a2**2 + (2*a1*a2*cos_phases))
        term1_deriv = 2*a2 + 2*a1*cos_phases
        term2_deriv = 2*a1 + 2*a2*cos_phases
        e_g_sin = np.sqrt((err1*term1_deriv)**2 + (errs2*term2_deriv)**2)
    elif filt=='r':
        c_rband_sin = np.sqrt(a1**2 + a2**2 + (2*a1*a2*cos_phases))
        term1_deriv = 2*a2 + 2*a1*cos_phases
        term2_deriv = 2*a1 + 2*a2*cos_phases
        e_r_sin = np.sqrt((err1*term1_deriv)**2 + (errs2*term2_deriv)**2)    
    elif filt=='i':
        c_iband_sin = np.sqrt(a1**2 + a2**2 + (2*a1*a2*cos_phases))
        term1_deriv = 2*a2 + 2*a1*cos_phases
        term2_deriv = 2*a1 + 2*a2*cos_phases
        e_i_sin = np.sqrt((err1*term1_deriv)**2 + (errs2*term2_deriv)**2)

c_gband = 0
c_rband = 0
c_iband = 0
c_zband = 0

e_g_musc = 0
e_r_musc = 0
e_i_musc = 0
c_z_musc = 0
for filt in ['g', 'r', 'i', 'z']:
    mod=79
    with open(f'muscat_runs/mcmc_state_job{mod}_{filt}.pkl', 'rb') as f:
        samples = pickle.load(f)
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        ndim = samples.shape[1]
        lister = []
        errlist = []
        for i in range(ndim):
            p14, p50, p86 = percentiles[:, i]
            if verbosity == 'y':
                print(f"MuSCAT, filter {filt}, A_{i}: {p50:.5f} (+{p86-p50:.10f} / -{p50-p14:.10f})")
            lister.append(p50)
            errlist.append(np.mean([p86-p50, p50-p14]))
    a1, a2 = lister
    err1, errs2 = errlist
    cos_phases = np.cos(phi_1-(2*phi_2))
    if filt=='g':
        c_gband = np.sqrt(a1**2 + a2**2 + (2*a1*a2*cos_phases))
        term1_deriv = 2*a2 + 2*a1*cos_phases
        term2_deriv = 2*a1 + 2*a2*cos_phases
        e_g_musc = np.sqrt((err1*term1_deriv)**2 + (errs2*term2_deriv)**2)
    elif filt=='r':
        c_rband = np.sqrt(a1**2 + a2**2 + (2*a1*a2*cos_phases))
        term1_deriv = 2*a2 + 2*a1*cos_phases
        term2_deriv = 2*a1 + 2*a2*cos_phases
        e_r_musc = np.sqrt((err1*term1_deriv)**2 + (errs2*term2_deriv)**2)
    elif filt=='i':
        c_iband = np.sqrt(a1**2 + a2**2 + (2*a1*a2*cos_phases))
        term1_deriv = 2*a2 + 2*a1*cos_phases
        term2_deriv = 2*a1 + 2*a2*cos_phases
        e_i_musc = np.sqrt((err1*term1_deriv)**2 + (errs2*term2_deriv)**2)
    elif filt=='z':
        c_zband = np.sqrt(a1**2 + a2**2 + (2*a1*a2*cos_phases))
        term1_deriv = 2*a2 + 2*a1*cos_phases
        term2_deriv = 2*a1 + 2*a2*cos_phases
        e_z_musc = np.sqrt((err1*term1_deriv)**2 + (errs2*term2_deriv)**2)


labels = ['T_{amb}', 'T_{spot}', 'f_{spot}', '\Delta f_{spot}']

tamb_sinistro, tspot_sinistro, fspot_sinistro, delfspot_sinistro = np.percentile(samples_sinistro[:, 0], [50]), np.percentile(samples_sinistro[:, 1], [50]), np.percentile(samples_sinistro[:, 2], [50]), np.percentile(samples_sinistro[:, 3], [50])

tamb, tspot, fspot, delfspot = np.percentile(samples_muscat[:, 0], [50]), np.percentile(samples_muscat[:, 1], [50]), np.percentile(samples_muscat[:, 2], [50]), np.percentile(samples_muscat[:, 3], [50])

bandpass=np.linspace(3800.,10000.,400)*u.angstrom


spotflux = get_phoenix_photons(temperature=tspot, logg=4.52, metallicity=0.12, wavelength=bandpass)[1].value
ambflux = get_phoenix_photons(temperature=tamb, logg=4.52, metallicity=0.12, wavelength=bandpass)[1].value

spotflux_sinistro = get_phoenix_photons(temperature=tspot_sinistro, logg=4.52, metallicity=0.12, wavelength=bandpass)[1].value
ambflux_sinistro = get_phoenix_photons(temperature=tamb_sinistro, logg=4.52, metallicity=0.12, wavelength=bandpass)[1].value


this_model_spectrum = fspot*spotflux + (1.0-fspot)*ambflux
this_model_spectrum_sinistro = fspot_sinistro*spotflux_sinistro + (1.0-fspot_sinistro)*ambflux_sinistro
  
                
d_lambda = (bandpass[1]-bandpass[0])
contrast = 1.-(spotflux/ambflux)
ds_over_s = -delfspot * ( contrast / ( 1.-fspot * contrast ) )
semi_amplitude = np.abs(ds_over_s)

numerator = np.nansum(semi_amplitude*this_model_spectrum*response_g*d_lambda)
denominator = np.nansum(this_model_spectrum*response_g*d_lambda)
g_S_m = numerator/denominator

numerator = np.nansum(semi_amplitude*this_model_spectrum*response_r*d_lambda)
denominator = np.nansum(this_model_spectrum*response_r*d_lambda)
r_S_m = numerator/denominator

numerator = np.nansum(semi_amplitude*this_model_spectrum*response_i*d_lambda)
denominator = np.nansum(this_model_spectrum*response_i*d_lambda)
i_S_m = numerator/denominator

numerator = np.nansum(semi_amplitude*this_model_spectrum*response_z*d_lambda)
denominator = np.nansum(this_model_spectrum*response_z*d_lambda)
z_S_m = numerator/denominator

contrast_sin = 1.-(spotflux_sinistro/ambflux_sinistro)
ds_over_s = -delfspot_sinistro * ( contrast_sin / ( 1.-fspot_sinistro * contrast_sin ) )
semi_amplitude = np.abs(ds_over_s)

numerator = np.nansum(semi_amplitude*this_model_spectrum_sinistro*response_g*d_lambda)
denominator = np.nansum(this_model_spectrum_sinistro*response_g*d_lambda)
g_S = numerator/denominator

numerator = np.nansum(semi_amplitude*this_model_spectrum_sinistro*response_r*d_lambda)
denominator = np.nansum(this_model_spectrum_sinistro*response_r*d_lambda)
r_S = numerator/denominator

numerator = np.nansum(semi_amplitude*this_model_spectrum_sinistro*response_i*d_lambda)
denominator = np.nansum(this_model_spectrum_sinistro*response_i*d_lambda)
i_S = numerator/denominator


cwls = [4770, 6230, 7630, 9130]
fig, ax = plt.subplots(figsize=(12,8), layout='constrained')
ax2 = ax.twinx()

label = r'$\frac{\Delta S}{S_{\mathrm{avg}}}$'
for cwl, filt, val, val2  in zip(cwls, ['g', 'r', 'i', 'z'], [g_S_m, r_S_m, i_S_m, z_S_m], [c_gband, c_rband, c_iband, c_zband]):
    if filt=='g':
        ax.scatter(cwl, val2, label='Data Guess (MuSCAT)', s=100, c='k', rasterized=True,marker='x')
        ax.scatter(cwl, val, label=r'Physical Model "50th %" (MuSCAT)', s=100, c='#05299E', rasterized=True)
    else:
        ax.scatter(cwl, val2, s=100, c='k', rasterized=True, marker='x')
        ax.scatter(cwl, val, s=100, c='#05299E', rasterized=True)
    
for cwl, filt, val, val2 in zip(cwls, ['g', 'r', 'i'], [g_S, r_S, i_S], [c_gband_sin, c_rband_sin, c_iband_sin]):
    if filt=='g': 
        ax.scatter(cwl-100, val2, s=100, c='k', rasterized=True, label='Data Guess (Sinistro)', marker='+')
        ax.scatter(cwl-100, val, s=100, c='r', label=r'Physical Model "50th %" (Sinistro)', rasterized=True)
    else:
        ax.scatter(cwl-100, val2, s=100, c='k', rasterized=True, marker='+')
        ax.scatter(cwl-100, val, s=100, c='r', rasterized=True)


for f, c, loc in zip('griz', ['g', 'r', 'm', 'brown'], [4770, 6230, 7630, 9130]):
    data = fetch_sdss_filter(f)
    if f =='g':
        ax2.fill(data[0], data[1], ec=c, fc=c, alpha=0.4)
    else:
        ax2.fill(data[0], data[1], ec=c, fc=c, alpha=0.4)
    ax2.text(loc, 0.02, f, color=c)

#ax.plot(bandpass, normed_model-0.6, lw=7, label="PHOENIX model")
ax.set_ylim(0, 0.1)
ax2.set_ylim(0, 0.5)
ax.set_title('MCMC Fits of Sinistro/MuSCAT (separately)', weight='bold')
ax.legend(fontsize=14)
ax.set_ylabel('Contrast')
ax.set_xlabel('Wavelength (Å)')
ax2.set_ylabel('Filter Sensitivity')

plt.savefig('../figures/fig2_contrasts.png')
plt.clf()
plt.close()
