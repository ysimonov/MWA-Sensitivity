# MWA-Sensitivity
Calculation of MWA Sensitivity based on Daniel Ung's theory

## Cite as:

## **[Ung, D.C.X., 2019. *Determination of Noise Temperature and Beam Modelling of an Antenna Array with Example Application using MWA* (Doctoral dissertation, Curtin University).](https://espace.curtin.edu.au/handle/20.500.11937/77989)** 

Simulated frequencies: 

    f = [49.92,327.68] MHz with df = 1.28 MHz

Extract spherical modal coefficients from archive (MWA_FEE_rev3_meas_LNA_imp.7z).

Usage (**input_data.py**):

1. Choose polarization ('X' or 'Y');

2. Specify gridpoint number from .csv file (See Data foulder); 

3. Provide ϕ- and θ- angles (spherical coordinates) of sky object that's been observed on the sky semisphere;


    Conversion from Elevation and Azimuth coordinates to spherical θ and ϕ angles:

        θ=arccos[cos(el)cos(az)]

        ϕ=arctan[tan(el)/sin(az)]
      
      
    Conversion from Astronomical u,v coordinates to spherical θ and ϕ angles:
    
        θ=arcsin[sqrt(u^2+v^2)]

        ϕ=arctan[u/v]

4. Give the date of observation and local coordinates of the observer.


Run: 

    OMP_THREAD_NUM=4 python3 MWA_sensitivity.py 
    (multithreaded) 

OR

    python3 MWA_sensitivity.py
  

Required modules:

***datetime, numpy, pandas, scipy, matplotlib, h5py, scikit-rf, astropy, aenum***

Installation: 

Linux:

pip3 install datetime, numpy, pandas, scipy, matplotlib, h5py, scikit-rf, astropy, aenum


Windows:

Download Latest Release of Python from https://www.python.org/downloads/windows/

-Open the Control Panel.
-Click System and Security, then System
-Click Advanced system settings on the left
-Inside the System Properties window, click the Environment Variables button.
-Under 'System Variables', click on PATH or path -> edit
-Specify location of python foulder, for example, 'C:\Python' and save
-Click edit again, and specify location of 'C:\Python\Scripts' where pip is installed, then save
-Open Windows Power Shell and type python. The power shell should recognize this command and show the version of Python.
-Enter 'python -m pip install –upgrade pip' command to upgrade pip
-Type pip install datetime, numpy, pandas, scipy, matplotlib, h5py, scikit-rf, astropy, aenum
-Open MWA_Sensitivity.py (open with), select IDLE Python

Press F5 to start the script. 



