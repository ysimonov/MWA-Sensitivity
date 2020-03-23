# MWA-Sensitivity
Calculation of MWA Sensitivity based on Daniel Ung's theory

Cite as:

Ung, D.C.X., 2019. Determination of Noise Temperature and Beam Modelling 
  of an Antenna Array with Example Application using MWA (Doctoral dissertation, Curtin University).

Extract spherical modal coefficients from archive (MWA_FEE_rev3_meas_LNA_imp.7z).

Usage (open constants.py):

1. Choose polarization ('X' or 'Y');

2. Specify gridpoint number from .csv file;

3. Provide phi- and theta- angles (spherical coordinates) of sky object that's been observed on the sky semisphere;

4. Give the date of observation and local coordinates of the observer.

Run: 

OMP_THREAD_NUM=4 python3 MWA_sensitivity.py 
(multithreaded) 

OR

python3 MWA_sensitivity.py
