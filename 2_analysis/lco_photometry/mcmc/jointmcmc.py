from astroML.datasets import fetch_sdss_filter
import argparse
from specutils.spectra import Spectrum
from IPython.display import display, Math
from specutils.fitting import fit_generic_continuum
from chromatic import *
import warnings
from corner import corner
from glob import glob
import os
import pickle
from sympy import pprint
import emcee
import matplotlib.pyplot as plt
import lightkurve as lk
import numpy as np
from speclite import filters
from astropy.constants import h, c, k_B
import astropy.units as u
import numpy as np
import pathlib
  
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--which', default = 'b', type = str, help = 'Do you want to fit contrasts for Sinistro (s), MuSCAT (m) or both (b, default)')
parser.add_argument('-c', '--checker', default=False, type = bool, help='If ran through subprocess, then skip the output of progress bars (True/False). Default False')
args = parser.parse_args()

checker = args.checker
if checker is True:
    progressor = False
else:
    progressor = True 
which_one = args.which

if checker != 'n':
    progresser = False
else:
    progresser = True
if which_one not in ['s', 'm', 'b']:
    raise ValueError("Invalid value for --which. Please choose 's', 'm', or 'b'.")


labels = ['T_{amb}', 'T_{spot}', 'f_{spot}', '\Delta f_{spot}']

os.environ["OMP_NUM_THREADS"] = "8"

if not os.path.exists("joint_contrasts"):
    os.makedirs("joint_contrasts/", exist_ok=True)
    if not os.path.exists("joint_contrasts/muscat"):
        os.makedirs("joint_contrasts/muscat", exist_ok=True)
    if not os.path.exists("joint_contrasts/sinistro"):
        os.makedirs("joint_contrasts/sinistro", exist_ok=True)

path = "../../../data/tessmcmc"
pathlist = sorted(glob(os.path.join(path, "*")))

os.chdir(pathlib.Path.cwd())
def set_rcparams():
    """Set Matplotlib.rcparam data"""
    tab = Table.read('corner_rcparams.txt', format='csv')
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
            if i==2:
                print(f"Sinistro, filter {filt}, constant offset: {p50:.5f} (+{p86-p50:.10f} / -{p50-p14:.10f})")
            else:
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

clist = [c_gband_sin, c_rband_sin, c_iband_sin]
clist2 = [c_gband, c_rband, c_iband, c_zband]

def emcee_modelfunction_sinistro(theta, wl):
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        tamb, tspot, fspot, del_fspot = theta
        spotflux = get_phoenix_photons(temperature=tspot, logg=4.52, metallicity=0.12,  R=100, wavelength=wl*u.Angstrom)[1].value
        ambflux = get_phoenix_photons(temperature=tamb, logg=4.52, metallicity=0.12,  R=100, wavelength=wl*u.Angstrom)[1].value
        this_model_spectrum = fspot*spotflux + (1.0-fspot)*ambflux
                    
        d_lambda = (wl[1]-wl[0])
        contrast = 1.-(spotflux/ambflux)
        ds_over_s = -del_fspot * ( contrast / ( 1.-fspot * contrast ) )
        semi_amplitude = np.abs(ds_over_s)
        
        numerator = np.nansum(semi_amplitude*this_model_spectrum*response_g*d_lambda)
        denominator = np.nansum(this_model_spectrum*response_g*d_lambda)
        g_S = numerator/denominator

        numerator = np.nansum(semi_amplitude*this_model_spectrum*response_r*d_lambda)
        denominator = np.nansum(this_model_spectrum*response_r*d_lambda)
        r_S = numerator/denominator

        numerator = np.nansum(semi_amplitude*this_model_spectrum*response_i*d_lambda)
        denominator = np.nansum(this_model_spectrum*response_i*d_lambda)
        i_S = numerator/denominator

        return g_S, r_S, i_S
    

def emcee_modelfunction_muscat(theta, wl):
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        tamb, tspot, fspot, del_fspot = theta
        spotflux = get_phoenix_photons(temperature=tspot, logg=4.52, metallicity=0.12,  R=100, wavelength=wl*u.Angstrom)[1].value
        ambflux = get_phoenix_photons(temperature=tamb, logg=4.52, metallicity=0.12,  R=100, wavelength=wl*u.Angstrom)[1].value
        this_model_spectrum = fspot*spotflux + (1.0-fspot)*ambflux
                    
        d_lambda = (wl[1]-wl[0])
        contrast = 1.-(spotflux/ambflux)
        ds_over_s = -del_fspot * ( contrast / ( 1.-fspot * contrast ) )
        semi_amplitude = np.abs(ds_over_s)
        
        numerator = np.nansum(semi_amplitude*this_model_spectrum*response_g*d_lambda)
        denominator = np.nansum(this_model_spectrum*response_g*d_lambda)
        g_S = numerator/denominator

        numerator = np.nansum(semi_amplitude*this_model_spectrum*response_r*d_lambda)
        denominator = np.nansum(this_model_spectrum*response_r*d_lambda)
        r_S = numerator/denominator

        numerator = np.nansum(semi_amplitude*this_model_spectrum*response_i*d_lambda)
        denominator = np.nansum(this_model_spectrum*response_i*d_lambda)
        i_S = numerator/denominator

        numerator = np.nansum(semi_amplitude*this_model_spectrum*response_z*d_lambda)
        denominator = np.nansum(this_model_spectrum*response_z*d_lambda)
        z_S = numerator/denominator

        return g_S, r_S, i_S, z_S
    
fspot_mu, del_fspot_mu = 0.41, 0.05
fspot_sigma, del_fspot_sigma = 0.05, 0.005
tambMin, tambMax, tspotMin, tspotMax = 3200, 12000, 2300, 3200

priors = [(tambMin, tambMax),
        (tspotMin, tspotMax),
        (fspot_mu, fspot_sigma),
        (del_fspot_mu, del_fspot_sigma)]

def logprior(theta):
    tamb, tspot, fspot, delfspot = theta
    prior = 0.0
    if  priors[0][0] < tamb < priors[0][1] and priors[1][0] < tspot < priors[1][1]:
        if (12000.0>=tamb>=tspot>=2300.0):
            if fspot is not None and not np.isnan(fspot):
                prior -= 0.5*((fspot- priors[2][0]) / priors[2][1])**2
            if delfspot is not None and not np.isnan(delfspot):
                prior -= 0.5*((delfspot - priors[3][0]) / priors[3][1])**2
            return prior
    else:
        return -np.inf

def lnlike_sinistro(theta, x, y, yerrs): #CHANGE
    """We want a RMS of the three parameters we are fitting together"""
    S_g, S_r, S_i = emcee_modelfunction_sinistro(theta, x) 
    cg, cr, ci = [val for val in y]
    g_error, r_error, i_error = [val for val in yerrs]
    chisq = np.nansum((cg-S_g)**2/(g_error)**2 + (cr-S_r)**2/(r_error)**2 + (ci-S_i)**2/(i_error)**2)
    err_weight = np.nansum(1./np.sqrt(2.*np.pi*(yerrs)))
    ln_like = (err_weight - 0.5*chisq)
    return ln_like

def lnprob_sinistro(theta, x, y, yerrs):
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_sinistro(theta, x, y, yerrs)

def lnlike_muscat(theta, x, y, yerrs): #CHANGE
    """We want a RMS of the three parameters we are fitting together"""
    S_g, S_r, S_i, S_z = emcee_modelfunction_muscat(theta, x) 
    cg, cr, ci, cz = [val for val in y]
    g_error, r_error, i_error, z_error = [val for val in yerrs]
    chisq = np.nansum((cg-S_g)**2/(g_error)**2 + (cr-S_r)**2/(r_error)**2 + (ci-S_i)**2/(i_error)**2 + (cz-S_z)**2/(z_error)**2)
    err_weight = np.nansum(1./np.sqrt(2.*np.pi*(yerrs)))
    ln_like = (err_weight - 0.5*chisq)
    return ln_like

def lnprob_muscat(theta, x, y, yerrs):
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_muscat(theta, x, y, yerrs)

if which_one == 'm':
    ## MuSCAT
    cerrs = np.array([e_g_musc, e_r_musc, e_i_musc, e_z_musc])*1.25

    data = (bandpass, clist2, cerrs)
    nwalkers = 100
    niter = 25000
    parameters_init_mc = [3998, 3093, 0.41, 0.05]

    initial = np.array(parameters_init_mc)
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

    def run_it(p0, nwalkers, niter, ndim, lnprob, data):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

        print("(MuSCAT) Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 5000, progress=progresser)
        sampler.reset()

        print("(MuSCAT) Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=progresser)

        return sampler, pos, prob, state

    sampler, pos, prob, state = run_it(p0, nwalkers, niter, ndim, lnprob_muscat, data)

    samples_muscat = sampler.flatchain
    with open(f'joint_contrasts/muscat/mcmc_state_job_joint.pkl', 'wb') as f:
        pickle.dump(samples_muscat, f)

    print("Autocorrelation time:", sampler.get_autocorr_time())

    print('MuSCAT Fit Parameters')
    for i in range(4):
        mcmc = np.percentile(samples_muscat[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        pprint(txt, use_unicode=False)
    
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), squeeze=True)
    fig.suptitle('MuSCAT', fontweight='bold', fontsize=20)
    corner(samples_muscat, fig=fig, hist_bin_factor=20, show_titles=True, plot_contours=True, use_math_text=True,
        labels=[r'T$_{amb}$', r'T$_{spot}$', r'f$_{spot}$', r'$\Delta$f$_{spot}$'], 
        label_kwargs={'fontsize': 15}, max_n_ticks=3, labelpad=0.25, rasterized=True)
    plt.savefig('joint_contrasts/muscatcorner.png', dpi=300)
    plt.close()
    plt.clf()    
elif which_one == 's':
## Sinistro

    cerrs = np.array([e_g_sin, e_r_sin, e_i_sin])*1.25

    data = (bandpass, clist, cerrs)
    nwalkers = 100
    niter = 25000
    parameters_init_mc = []
    with open('joint_contrasts/muscat/mcmc_state_job_joint.pkl', 'rb') as f:
        samples_muscat = pickle.load(f)

    for i in range(4): # Use the MuSCAT fits as the initial guesses for the Sinistro Sampler
        mcmc = np.percentile(samples_muscat[:, i], [50])
        p50 = mcmc[0]
        parameters_init_mc.append(p50)

    initial = np.array(parameters_init_mc)
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

    def run_it(p0, nwalkers, niter, ndim, lnprob, data):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

        print("(Sinistro) Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 5000, progress=progresser)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=progresser)

        return sampler, pos, prob, state

    sampler, pos, prob, state = run_it(p0, nwalkers, niter, ndim, lnprob_sinistro, data)

    samples_sinistro = sampler.flatchain
    with open(f'joint_contrasts/sinistro/mcmc_state_job_joint.pkl', 'wb') as f:
        pickle.dump(samples_sinistro, f)
    print("Autocorrelation time:", sampler.get_autocorr_time())

    print('Sinistro Fit Parameters')
    for i in range(4):
        mcmc = np.percentile(samples_sinistro[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        pprint(txt, use_unicode=False)

    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), squeeze=True)
    fig.suptitle('Sinistro', fontweight='bold', fontsize=20)
    corner(samples_sinistro, fig=fig, hist_bin_factor=20, show_titles=True, plot_contours=True, use_math_text=True,
        labels=[r'T$_{amb}$', r'T$_{spot}$', r'f$_{spot}$', r'$\Delta$f$_{spot}$'], 
        label_kwargs={'fontsize': 15}, max_n_ticks=3, labelpad=0.25, rasterized=True)
    plt.savefig('joint_contrasts/sinistrocorner.png', dpi=300)
    plt.close()
    plt.clf()
elif which_one == 'b':
    ## Run them both, first MuSCAT then Sinistro
    ## MuSCAT
    cerrs = np.array([e_g_musc, e_r_musc, e_i_musc, e_z_musc])*1.25

    data = (bandpass, clist2, cerrs)
    nwalkers = 100
    niter = 25000
    parameters_init_mc = [3998, 3093, 0.41, 0.05]

    initial = np.array(parameters_init_mc)
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

    def run_it(p0, nwalkers, niter, ndim, lnprob, data):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

        print("(MuSCAT) Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 5000, progress=progresser)
        sampler.reset()

        print("(MuSCAT) Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=progresser)

        return sampler, pos, prob, state

    sampler, pos, prob, state = run_it(p0, nwalkers, niter, ndim, lnprob_muscat, data)

    samples_muscat = sampler.flatchain
    with open(f'joint_contrasts/muscat/mcmc_state_job_joint.pkl', 'wb') as f:
        pickle.dump(samples_muscat, f)

    print("Autocorrelation time:", sampler.get_autocorr_time())

    print('MuSCAT Fit Parameters')
    for i in range(4):
        mcmc = np.percentile(samples_muscat[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        pprint(txt, use_unicode=False)
    
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), squeeze=True)
    fig.suptitle('MuSCAT', fontweight='bold', fontsize=20)
    corner(samples_muscat, fig=fig, hist_bin_factor=20, show_titles=True, plot_contours=True, use_math_text=True,
        labels=[r'T$_{amb}$', r'T$_{spot}$', r'f$_{spot}$', r'$\Delta$f$_{spot}$'], 
        label_kwargs={'fontsize': 15}, max_n_ticks=3, labelpad=0.25, rasterized=True)
    plt.savefig('joint_contrasts/muscatcorner.png', dpi=300)
    plt.close()
    plt.clf()

    cerrs = np.array([e_g_sin, e_r_sin, e_i_sin])*1.25

    data = (bandpass, clist, cerrs)
    nwalkers = 100
    niter = 25000
    parameters_init_mc = []
    with open('joint_contrasts/muscat/mcmc_state_job_joint.pkl', 'rb') as f:
        samples_muscat = pickle.load(f)

    for i in range(4): # Use the MuSCAT fits as the initial guesses for the Sinistro Sampler
        mcmc = np.percentile(samples_muscat[:, i], [50])
        p50 = mcmc[0]
        parameters_init_mc.append(p50)

    initial = np.array(parameters_init_mc)
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

    def run_it(p0, nwalkers, niter, ndim, lnprob, data):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

        print("(Sinistro) Running burn-in...")
        
        p0, _, _ = sampler.run_mcmc(p0, 5000, progress=progresser)
        sampler.reset()

        print("(Sinistro) Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=progresser)

        return sampler, pos, prob, state

    sampler, pos, prob, state = run_it(p0, nwalkers, niter, ndim, lnprob_sinistro, data)

    samples_sinistro = sampler.flatchain
    with open(f'joint_contrasts/sinistro/mcmc_state_job_joint.pkl', 'wb') as f:
        pickle.dump(samples_sinistro, f)
    print("Autocorrelation time:", sampler.get_autocorr_time())

    print('Sinistro Fit Parameters')
    for i in range(4):
        mcmc = np.percentile(samples_sinistro[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        pprint(txt, use_unicode=False)

    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), squeeze=True)
    fig.suptitle('Sinistro', fontweight='bold', fontsize=20)
    corner(samples_sinistro, fig=fig, hist_bin_factor=20, show_titles=True, plot_contours=True, use_math_text=True,
        labels=[r'T$_{amb}$', r'T$_{spot}$', r'f$_{spot}$', r'$\Delta$f$_{spot}$'], 
        label_kwargs={'fontsize': 15}, max_n_ticks=3, labelpad=0.25, rasterized=True)
    plt.savefig('joint_contrasts/sinistrocorner.png', dpi=300)
    plt.close()
    plt.clf()
