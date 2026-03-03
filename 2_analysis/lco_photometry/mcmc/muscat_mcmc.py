"""Run MCMC sampler to get sine-model amplitudes from MuSCAT data"""
from glob import glob
import os
import pickle
import emcee
import matplotlib.pyplot as plt
from astropy.io import ascii as asc
import numpy as np
from mcmcfuncs import muscat_lks

os.environ["OMP_NUM_THREADS"] = "8"

if not os.path.exists("muscat_im"):
    os.makedirs("muscat_im/", exist_ok=True)
if not os.path.exists("muscat_runs"):
    os.makedirs("muscat_runs", exist_ok=True)

PATH = "../../../data/tessmcmc"
pathlist = sorted(glob(os.path.join(PATH, "*")))

PATH59, PATH79, PATH97 = pathlist
del PATH, pathlist

COMMONPATH = '../../../data/lco_aumic/muscat'
G, R, I, Z = [sorted(glob(os.path.join(COMMONPATH, FIL))) for \
              FIL in ['muscat_gp*', 'muscat_rp*', 'muscat_ip*', 'muscat_zs_*']]
removals = [[0, 43, 1, 1, 0, 0],  [0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0],  [0, 0, 12, 0, 0, 0]]
lightcurves = []
for the_variables in zip(['G', 'R', 'I', 'Z'], [G, R, I, Z], removals):
    filt, filelist, removelist = the_variables
    filt_objs = [asc.read(f) for f in filelist]
    [filt_objs[i] == filt_objs[i].sort(keys='rel_flux_T1') for i in range(len(filt_objs))]
    for index, j in enumerate(removelist):
        filt_objs[index] = filt_objs[index][j:]
    lightcurves.append(muscat_lks(filt_objs, normalize=True))
GPLC, RPLC, IPLC, ZSLC = lightcurves

del COMMONPATH, G, R, I, Z, removals

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
    a1, a2 = theta
    return a1 * np.sin(f * xlist + p1) + a2 * np.sin(f/2 * xlist + p2)

def main(path, num):
    with open(path, "rb") as f:
        state = pickle.load(f)
        percentiles = np.percentile(state, [50], axis=0)
        ndim = state.shape[1]
        lister = []
        for i in range(ndim):
            p50 = percentiles[:, i]
            lister.append(p50)
    freq, a1given, a2given, phi_1, phi_2 = [val[0] for val in lister]
    asserted_params = [freq, phi_1, phi_2]
    model_avg = 0
    for filt, data in zip(['g', 'r', 'i', 'z'], [GPLC, RPLC, IPLC, ZSLC]):
        xs, ys, errs = data.time.value, \
            data.flux.value - np.median(data.flux.value), data.flux_err.value
        amp1_mu, amp2_mu, a_sigs = a1given, a2given, 0.1

        priors = [(amp1_mu, a_sigs),
                    (amp2_mu, a_sigs)]

        def logprior(theta):
            lprior = 0
            for i, val in enumerate(theta):
                if val is not None or val is not np.nan:
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
        nwalkers = 250
        niter = 10000
        parameters_init_mc = [a1given, a2given]
        initial = np.array(parameters_init_mc)
        ndim = len(initial)
        p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]

        def run_it(p0, nwalkers, niter, ndim, lnprob, data):
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, 1000, progress=False)
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
        lister = []
        for i in range(ndim):
            p14, p50, p86 = percentiles[:, i]
            print(f"A_{i}: {p50:.5f} (+{p86-p50:.10f} / -{p50-p14:.10f})")
            lister.append(p50)

        samples = sampler.flatchain

        with open(f'muscat_runs/mcmc_state_job{num}_{filt}.pkl', 'wb') as f:
            pickle.dump(samples, f)

        xlist = np.linspace(np.min(xs), np.max(xs), 1000)
        best_fit_model_plot = emcee_func(lister, asserted_params, xlist)
        best_fit_model_comparison = emcee_func(lister, asserted_params, xs)

        plt.plot(xlist, best_fit_model_plot, c='orange', \
                 label='Highest Likelihood MCMC Model', rasterized=True)
        plt.scatter(xs, ys, c='b', label=f'MuSCAT LightCurve {filt}', rasterized=True)

        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.legend(fontsize=10, loc='upper right')
        plt.savefig(f"muscat_im/emceefit_{num}_{filt}", dpi=300)
        plt.clf()
        plt.close()

        print(f"Finished MCMC for {filt} filter of job {num}.")
        model_avg += np.abs(np.median(ys - best_fit_model_comparison))
    print(f'Model ({num}) average difference (real fluxes - model) across all 4 filters is: \
          {model_avg}, or {model_avg/4} if averaged\n')

main(PATH79, 79)
