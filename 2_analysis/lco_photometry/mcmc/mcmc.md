## This section of the Github describes Sections 3.x-3.y in Schochet and Feinstein

### The `muscat_mcmc.py` file is meant to be ran first [Sec 3.xx] (assuming you have downloaded the original TESS-fit files in the `data/tessmcmc` folder). It will fit and save files in this `mcmc` folder.

### After this you can run the `sinistro_mcmc.py` to get a fit for the same parameters in the Sinistro data by using the MuSCAT fits as a prior [Sec 3.xy]

### Finally you can run the code in `mcmc_joint.py` which runs a comprehensive fit for a physical model of the filter-dependent spot contrasts to match the fit amplitudes from above