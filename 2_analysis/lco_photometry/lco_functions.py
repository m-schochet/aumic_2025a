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
    if normalize==True:
        return lk.LightCurve(time=d, flux=fl, flux_err=err).normalize()
    return lk.LightCurve(time=d, flux=fl, flux_err=err)

def find_lost(datalist, filter_type):
    """ Finds the files that were not used in the cleaned AIJ photometry

    NOTE: Requires access to original .fits files on external drive, here
    that path is assumed to be /Volumes/harddrive/

    Args:
        datalist (pd.DataFrame): list of files with a 'Label" column
        filter_type (str): 'gp', 'ip', 'rp', 'U', 'B', V'
    """ 
    files_goodAIJ = sorted([str(val) for val in list(datalist['Label'])])
    AIJ_FILES = pd.DataFrame({"files": files_goodAIJ})
    if(filter_type=='U'):
        files_preaij = sorted(glob(f"/Volumes/harddrive/U_notwirl/*.fits.fz"))
    elif(filter_type=='B'):
        files_preaij = sorted(glob(f"/Volumes/harddrive/B_notwirl/*.fits.fz"))
    else:
        files_preaij = sorted(glob(f"/Volumes/harddrive/{filter_type}/aligned/*.fits"))
    
    LCO_FILES = pd.DataFrame({"files": files_preaij})
    LCO_FILENAMES = [str(os.path.basename(val)) for val in (LCO_FILES['files'])]
    LCO_FILENAMES_PD = pd.DataFrame({"files": LCO_FILENAMES})
    combined = [bool(np.isnan(val)) for val in 
                    LCO_FILENAMES_PD['files'].value_counts() - AIJ_FILES['files'].value_counts()]

    badfiles = [str(val) for val in LCO_FILES[combined]['files'].values.tolist()]
    print('The number of bad files in the', filter_type, 'filter are:', len(badfiles))
    path = pathlib.Path('bad_files/')
    if not os.path.exists(path):
        path.mkdir(parents=True, exist_ok=True)
    with open(f'bad_files/bad_{filter_type}files.txt', 'w') as f:
        for item in badfiles:
            new_string = item.replace("aligned_", "")
            f.write(new_string + '\n')
    return(filter_type)


def binned_lc(tlist, flist, elist):
    """Bin a light curve (where a new day is asserted if an image timestamp occurs
    0.45 JD = 10.8hrs after the previous one

    Args:
        tlist (list-like): light curve times
        flist (list-like): light curve fluxes
        elist (list-like): flux errors

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
                if ((flux_counter!=0) & (num_counter!=0)):
                    returned_binnedlc.loc[len(returned_binnedlc)] = [flux_counter/num_counter, date_counter/num_counter, error_counter/num_counter]
                date_counter, flux_counter, num_counter, error_counter  = 0, 0, 0, 0
            else:
                flux_counter+=flist[num]
                date_counter+=dates
                error_counter+=elist[num]     
                num_counter+=1
    flist = [val for val in returned_binnedlc['flux'].values]
    elist = [val for val in returned_binnedlc['errs'].values]
    tlist = [val for val in returned_binnedlc['times'].values]
    final = lk.LightCurve(time=tlist, flux=flist, flux_err=elist)
    folded_fin = final.fold(4.86, epoch_time=2400000, epoch_phase=1)
    return final, folded_fin