# MWA-Sensitivity

## Cite as:

## **[Ung, D.C.X., 2019. *Determination of Noise Temperature and Beam Modelling of an Antenna Array with Example Application using MWA* (Doctoral dissertation, Curtin University).](https://espace.curtin.edu.au/handle/20.500.11937/77989)** 


The program computes:

- Antenna noise temperature due to local sky image (Haslam 408 MHz all sky map) (K)
- Effective area of 4x4 MWA array tile (m^2)
- Radiation efficiency (%) 
- Realised area (m^2)
- Receiver noise temperature (K)
- System noise temperature (K)
- System Equivalent Flux Density (SEFD) in kJy
- Sensitivity (m^2.K^-1)

Simulated frequencies: 

    f = [49.92,327.68] MHz with df = 1.28 MHz

Extract spherical modal coefficients from archive (MWA_FEE_rev3_meas_LNA_imp.7z) in the same folder (Data/).

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

Windows / Linux / MacOS:

- Download Anaconda for Windows (Select Latest Version): https://www.anaconda.com/distribution/#download-section
- Open Anaconda Navigator
- Open Spyder
- In the Console Line type: pip instal aenum, scikit-rf
- Download Zip folder and extract archive
- Go to /Data/ folder and extract MWA_FEE_rev3_meas_LNA_imp.h5 file
- Open MWA_Sensitivity.py in Spyder or type, for example, "cd C:\Users\Desktop\MWA-Sensitivity-master" in the Console Line
- Enter 'run MWA_sensitivity' in the Console Line to run the main script
- Alternatively, skip the above two options, and use key combination Ctrl+O, navigate to the folder where MWA_Sensitivity.py is located and press F5 to run the script




