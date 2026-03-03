"""All relevant functions that are used several times in our analysis 
python scripts"""

from glob import glob
import os
import pathlib
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

def find_lost(datalist, filter_type, telescope='sinitro'):
    """ Finds the files that were not used in the cleaned AIJ photometry

    NOTE: Requires access to original .fits files on external drive, here
    that path is assumed to be /Volumes/harddrive/

    Args:
        datalist (pd.DataFrame): list of files with a 'Label" column
        filter_type (str): 'gp', 'ip', 'rp', 'U', 'B', V'
        telescope (str): 'sinistro' or 'muscat;'
    """
    files_good_aij = sorted([str(val) for val in list(datalist['Label'])])
    aij_files = pd.DataFrame({"files": files_good_aij})
    if filter_type=='U':
        files_preaij = sorted(glob("/Volumes/harddrive/U_notwirl/*.fits.fz"))
    elif filter_type=='B':
        files_preaij = sorted(glob("/Volumes/harddrive/B_notwirl/*.fits.fz"))
    else:
        files_preaij = sorted(glob(f"/Volumes/harddrive/{filter_type}/aligned/*.fits"))

    lco_files = pd.DataFrame({"files": files_preaij})
    lco_filenames = [str(os.path.basename(val)) for val in (lco_files['files'])]
    lco_filenames_pd = pd.DataFrame({"files": lco_filenames})
    combined = [bool(np.isnan(val)) for val in \
                    lco_filenames_pd['files'].value_counts() - aij_files['files'].value_counts()]

    badfiles = [str(val) for val in lco_files[combined]['files'].values.tolist()]
    print('The number of bad files in the', filter_type, 'filter are:', len(badfiles))
    path = pathlib.Path('bad_files/')
    if not os.path.exists(path):
        path.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(os.path.join(path, 'sinistro')):
            os.path.join(path, 'sinistro').mkdir(parents=True, exist_ok=True)
        if not os.path.exists(os.path.join(path, 'muscat')):
            os.path.join(path, 'muscat').mkdir(parents=True, exist_ok=True)
    if telescope=='sinistro':
        with open(f'bad_files/sinistro/bad_{filter_type}files.txt', 'w') as f:
            for item in badfiles:
                new_string = item.replace("aligned_", "")
                f.write(new_string + '\n')
        return filter_type
    with open(f'bad_files/muscat/bad_{filter_type}files.txt', 'w') as f:
        for item in badfiles:
            #new_string = item.replace("aligned_", "")
            f.write(item + '\n')
    return filter_type


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
                if (flux_counter!=0) and (num_counter!=0):
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
