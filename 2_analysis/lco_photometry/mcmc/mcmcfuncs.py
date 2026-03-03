"""Copied from `lco_functions.py` to bring all the 
necessary functions for our MCMC samplers into the folder"""

from astropy.io import ascii as asc
import lightkurve as lk
import numpy as np
import pandas as pd

def file_len(path):
    """Get the length of a light curve for trimming purposes

    Args:
        path (str): File path to the light curve data.

    Returns:
        int: Length of the light curve data.
    """
    data = asc.read(path)
    return len(data)

def file_load(path, cleanrange=None, full=False):
    """Load a path of LCO .fits file to plot the light curve

    Args:
        path (str):  filepath to .xls file
        cleanrange (tuple, optional): Range to 'clean' flux values. 
            Defaults to None.
        full (bool, optional): Whether to return the full data 
            instead of light-curve information.

    Returns:
        tuple: Arrays of JD, relative flux, flux error, and flux SNR.
    """
    data = asc.read(path)
    data.sort(keys='rel_flux_T1')
    if cleanrange:
        if cleanrange[0] is None or cleanrange[1] is None:
            if cleanrange[0] is None:
                data = data[:cleanrange[1]]
            else:
                data = data[cleanrange[0]:]
        else:
            data = data[cleanrange[0]:cleanrange[1]]
    data.sort(keys='J.D.-2400000')
    if full is True:
        return data
    return data['J.D.-2400000'], data['rel_flux_T1'], \
        data['rel_flux_err_T1'], data['rel_flux_SNR_T1']

def create_lc(days, fluxes, errs, snr_list):
    """Create a light curve object from time, 
    flux, error, and SNR arrays.

    Args:
        days (array): Array of time values (JD).
        fluxes (array): Array of relative flux values.
        errs (array): Array of flux error values.
        snr_list (array): Array of flux SNR values.
    Returns:
        LightCurve: Light curve object.
    """
    lc = lk.LightCurve(time=days, flux=fluxes, flux_err=errs)
    return lc, snr_list


def muscat_lks(filelist, normalize = False):
    """Create lightcurves from MuSCAT LCO imaging
    
    Args:
        filelist (list): List of astropy tables with LCO MuSCAT data.
        normalize (bool, optional): Whether to normalize the light curve. Defaults to False.
        
    Returns:
        LightCurve: Light curve object."""
    d, fl, err = [], [], []
    day_list = [(day['J.D.-2400000']-57000).tolist() for day in filelist]
    flux_list = [flux['rel_flux_T1'].tolist() for flux in filelist]
    fluxerr_list = [fluxerr['rel_flux_err_T1'].tolist() for fluxerr in filelist]
    for day, flux, error in zip(day_list, flux_list, fluxerr_list):
        d = d + day
        fl = fl + flux
        err = err + error
    if normalize is True:
        return lk.LightCurve(time=d, flux=fl, flux_err=err).normalize()
    return lk.LightCurve(time=d, flux=fl, flux_err=err)

def binned_lc(tlist, flist, elist, norm=False):
    """Bin a light curve (where a new day is asserted if an image timestamp occurs
    0.45 JD = 10.8hrs after the previous one

    Args:
        tlist (list-like): light curve times
        flist (list-like): light curve fluxes
        elist (list-like): flux errors
        norm (bool, optional): Whether to normalize the light curve. Defaults to False.
    Returns:
        final (lk.LightCurve): binned light curve
        folded_fin (lk.LightCurve): phase-folded and binned light curve
    """
    returned_binnedlc = pd.DataFrame(data={"flux":[], "times":[], "errs":[]})
    date_counter, flux_counter, num_counter, error_counter  = 0, 0, 0, 0
    for num, dates in enumerate(tlist):
        if num==0:
            flux_counter+=flist[num]
            date_counter+=dates
            error_counter+=elist[num]
            num_counter+=1
        else:
            datediffs = np.diff(tlist)
            if datediffs[num-1] > 0.45:
                if flux_counter!=0 and num_counter!=0:
                    returned_binnedlc.loc[len(returned_binnedlc)] = \
                        [flux_counter/num_counter, date_counter/num_counter, \
                         error_counter/num_counter]
                date_counter, flux_counter, num_counter, error_counter  = 0, 0, 0, 0
            else:
                flux_counter+=flist[num]
                date_counter+=dates
                error_counter+=elist[num]
                num_counter+=1
    flist = list(returned_binnedlc['flux'].values)
    elist = list(returned_binnedlc['errs'].values)
    tlist = list(returned_binnedlc['times'].values)
    final = lk.LightCurve(time=tlist, flux=flist, flux_err=elist)
    if norm:
        final = final.normalize()
    folded_fin = final.fold(4.86, epoch_time=2400000, epoch_phase=1)
    return final, folded_fin
