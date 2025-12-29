from astropy.wcs.utils import proj_plane_pixel_scales
from twirl import compute_wcs, gaia_radecs, find_peaks
from requests import HTTPError
from twirl.geometry import sparsify
from astropy.wcs import WCS
from astropy.io import fits
from datetime import datetime
import pytz
import numpy as np
import json
from glob import glob
import sys
import os

key_path = 'wcs_headers.json'
wcs_keys = []

with open(key_path, 'r') as file:
    wcs_keys = json.load(file)

def run(path):
    os.makedirs(os.path.join(path, "twirl_failed"), exist_ok = True)
    filelist = sorted(glob(os.path.join(path, "*.fits")))
    for file in filelist:
        try:
            with fits.open(file, mode='update') as hdu:
                hdu1 = hdu[0]
                if(hdu1.header['EXPTIME']<1):
                    print(f"Exposure <1 seconds, so file {file} should be skipped", file=sys.stdout)
                    os.rename(file, os.path.join(path, "twirl_failed", os.path.split(file)[1]))
                    continue
                data, original_wcs = hdu1.data, WCS(hdu1.header)
                xy = find_peaks(data)[0:20]
                time = datetime.strptime(hdu1.header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
                time_aware = pytz.utc.localize(time)
                fov = (data.shape * proj_plane_pixel_scales(original_wcs))[0]
                center = original_wcs.pixel_to_world(*np.array(data.shape) / 2)
                radecs = gaia_radecs(center, 1.2*fov, limit=50, dateobs=time_aware)
                radecs = sparsify(radecs, 0.01)
                wcs = compute_wcs(xy, radecs[0:20], tolerance=5)
                new_wcs = wcs.to_fits()
                for key in wcs_keys:
                    if(key=="PC1_1"):
                        rep = "CD1_1"
                    elif(key=="PC1_2"):
                        rep = "CD1_2"
                    elif(key=="PC2_1"):
                        rep = "CD2_1"
                    elif(key=="PC2_2"):
                        rep = "CD2_2"
                    else:
                        rep = str(key)
                    hdu1.header[rep] = new_wcs[0].header[key]
                hdu.flush()
        except (ValueError, HTTPError) as e:
            print(f"Error {e} so file {file} should be skipped", file=sys.stdout)
            os.rename(file, os.path.join(path, "twirl_failed", os.path.split(file)[1]))


if __name__ == "__main__":
    path = str(sys.argv[1])
    run(path)
