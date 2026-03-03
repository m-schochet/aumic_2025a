"""Run MCMC sampler to get sine-model amplitudes from Sinistro data"""
from glob import glob
import os
import pickle
import emcee
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mcmcfuncs import file_len, file_load, create_lc, binned_lc

os.environ["OMP_NUM_THREADS"] = "8"

if not os.path.exists("sinistro_im"):
    os.makedirs("sinistro_im/", exist_ok=True)
if not os.path.exists("sinistro_runs/"):
    os.makedirs("sinistro_runs", exist_ok=True)

pathtess = "../../../data/tessmcmc"
path59, path79, path97 = sorted(glob(os.path.join(pathtess, "*")))

muscatpath = "muscat_runs/"
muscatpathlist = sorted(glob(os.path.join(muscatpath, "*")))

commonpath_folded = '../../../data/lco_aumic/folded_sinistro/'
paths = sorted([os.path.join(commonpath_folded, specific) for \
                specific in os.listdir(commonpath_folded) if specific.endswith('.csv')])
foldedlcs = [pd.read_csv(path) for path in paths]

sinistrolcs = [] 
for band in foldedlcs:
    sinistrolcs.append(lk.LightCurve(time=band['time'], \
                                     flux=band['flux'], \
                                     flux_err=band['flux_err']).normalize())

twirled_path = '../../../data/lco_aumic/lcs_posttwirl/'
twirled_paths = sorted([os.path.join(twirled_path, specific) for \
                specific in os.listdir(twirled_path) if specific.endswith('.xls')])
# output order is B, U, V, gp, ip, rp


lens = [file_len(path) for path in [twirled_paths[1], twirled_paths[4]]]
datas = []
for file, cleans in zip(twirled_paths, [[None, None], [2, lens[0]-13],\
                                [None, None], [2, None],\
                                [28, lens[1]-1], [1, None]]):
    datum = file_load(file, cleanrange=cleans)
    datas.append(datum)

del paths, pathtess, path59, path97, commonpath_folded, 
del twirled_path, twirled_paths, lens, foldedlcs, muscatpath

BDATA, UDATA, VDATA, GDATA, IDATA, RDATA = datas

B_LC, BSNR = create_lc(*BDATA)
U_LC, USNR = create_lc(*UDATA)
V_LC, VSNR = create_lc(*VDATA)
G_LC, GSNR = create_lc(*GDATA)
I_LC, ISNR = create_lc(*IDATA)
R_LC, RSNR = create_lc(*RDATA)


lcs = [G_LC, R_LC, I_LC, U_LC, B_LC, V_LC]
binnedlcs, foldedlcs = [], []

for index, lc in enumerate(lcs):
    if index<3:
        binned, folded = binned_lc(lc.time.value, lc.flux.value, lc.flux_err, norm=True)
    else:
        binned, folded = binned_lc(lc.time.value, lc.flux.value, lc.flux_err)
    binnedlcs.append(binned)
    foldedlcs.append(folded)

for lc in foldedlcs:
    lc /= np.median(lc[(lc['time'].value > 0.5) & (lc['time'].value < 1.0)]['flux'])

G_FOLD, R_FOLD, I_FOLD, U_FOLD, B_FOLD, V_FOLD = foldedlcs
G_BIN, R_BIN, I_BIN, U_BIN, B_BIN, V_BIN = binnedlcs

def emcee_func(theta, asserted, xlist):
    """A 2-sine wave function with parameters for variable phase, total offset, and frequency.
    The second sine wave has a harmonic frequency (frequency/2).

    Args:
        theta (list): List of free parameters [amplitude1, amplitude2].
        asserted (list): List of asserted parameters [frequency, phase1, phase2].
        xlist (list): List of x-values to generate the 2-sine wave model over.

    Returns:
        list: f(x), the value of the 2-sine wave model at each x-value in X.
    """
    f, p1, p2 = asserted
    a1, a2, constant = theta
    return a1 * np.sin(f * xlist + p1) + a2 * np.sin(f/2 * xlist + p2) + constant

def main():
    lister = []
    with open(path79, "rb") as f:
        state = pickle.load(f)
        percentiles = np.percentile(state, [50], axis=0)
        ndim = state.shape[1]
        lister = []
        for i in range(ndim):
            p50 = percentiles[:, i]
            lister.append(p50)
    freq, a1given, a2given, phi_1, phi_2 = [val[0] for val in lister]
    asserted_params = [freq, phi_1, phi_2]
    with open(muscatpathlist[0], "rb") as gfilt:
        state = pickle.load(gfilt)
        percentiles = np.percentile(state, [50], axis=0)
        ndim = state.shape[1]
        lister2 = []
        for i in range(ndim):
            p50 = percentiles[:, i]
            lister2.append(p50)
        a1_g, a2_g = lister2
    with open(muscatpathlist[2], "rb") as rfilt:
        state = pickle.load(rfilt)
        percentiles = np.percentile(state, [50], axis=0)
        ndim = state.shape[1]
        lister3 = []
        for i in range(ndim):
            p50 = percentiles[:, i]
            lister3.append(p50)
        a1_r, a2_r = lister3
    with open(muscatpathlist[1], "rb") as ifilt:
        state = pickle.load(ifilt)
        percentiles = np.percentile(state, [50], axis=0)
        ndim = state.shape[1]
        lister4 = []
        for i in range(ndim):
            p50 = percentiles[:, i]
            lister4.append(p50)
        a1_i, a2_i = lister4
    num = 79
    model_avg = 0
    cons_guess = 0
    for filt, data, alister in zip(['g', 'r', 'i'], [G_FOLD, R_FOLD, I_FOLD], \
                                    [[a1_g, a2_g], [a1_r, a2_r], [a1_i, a2_i]]):
        xs, ys, errs = data.time.value, \
            data.flux.value - np.median(data.flux.value), data.flux_err.value
        amp1_mu, amp2_mu, a_sigs = alister[0], alister[1], 0.5
        const_low, constant_upper = -0.1, 0.1
        priors = [(amp1_mu, a_sigs),
                    (amp2_mu, a_sigs),
                    (const_low, constant_upper)]

        def logprior(theta):
            lprior = 0
            for i, val in enumerate(theta):
                if i==2 and (val > const_low and val < constant_upper):
                    pass
                elif val is not None or val is not np.nan:
                    lprior -= 0.5*((val - priors[i][0]) / priors[i][1])**2
                else:
                    return -np.inf
            return lprior

        def lnlike(theta, x, y, yerr):
            return -0.5 * np.sum(((y - emcee_func(theta, asserted_params, x))/yerr) ** 2)

        def lnprob(theta, x, y, yerr):
            lp = logprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)

        data = (xs, ys, errs)
        nwalkers = 200
        niter = 50000
        parameters_init_mc = [a1given, a2given, cons_guess]

        initial = np.array(parameters_init_mc)
        ndim = len(initial)
        p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

        def run_it(p0, nwalkers, niter, ndim, lnprob, data):
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, 2500, progress=False)
            sampler.reset()

            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

            return sampler, pos, prob, state

        sampler, pos, prob, state = run_it(p0,nwalkers,niter,ndim,lnprob,data)
        del pos, prob
        flat_samples = sampler.get_chain(thin=15, flat=True)

        # 2. Calculate 14th, 50th (median), and 86th percentiles for each parameter
        percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)

        ndim = flat_samples.shape[1]
        lister_final = []
        for i in range(ndim):
            p14, p50, p86 = percentiles[:, i]
            print(f"A_{i}: {p50:.5f} (+{p86-p50:.10f} / -{p50-p14:.10f})")
            lister_final.append(p50)

        samples = sampler.flatchain
        with open(f'sinistro_runs/mcmc_state_job{num}_{filt}.pkl', 'wb') as f:
            pickle.dump(samples, f)

        xlist = np.linspace(np.min(xs), np.max(xs), 1000)
        best_fit_model_plot = emcee_func(lister_final, asserted_params, xlist)
        best_fit_model_comparison = emcee_func(lister_final, asserted_params, xs)

        plt.plot(xlist, best_fit_model_plot, c='orange', \
                 label='Highest Likelihood MCMC Model', rasterized=True)
        plt.scatter(xs, ys, c='b', label=f'Sinistro LightCurve {filt}', rasterized=True)

        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.legend(fontsize=10, loc='upper right')
        plt.savefig(f"sinistro_im/emceefit_{num}_{filt}", dpi=300)
        plt.clf()
        plt.close()

        print(f"Finished MCMC for {filt} filter of job {num}.\n")
        model_avg += np.abs(np.sum(ys - best_fit_model_comparison)/len(ys))
    print(f'Average difference, given as SUM(real vals - fitted vals)/Nvals across all 3 filters.')
    print(f'Model ({num}) average difference (real fluxes - model) across all 3 filters is: \
          {model_avg}, or {model_avg/3} if averaged\n')
main()
