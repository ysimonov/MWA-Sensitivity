from aenum import Constant

class CONST(Constant):

    #===============VARIABLE DECLARATION===============#

    POL = 'X' #SPECIFY POLARISATION HERE

    DEGS_PER_PIXEL = 0.5 #deg/pix - number of degrees per pixel (used to construct angular grid). The lower the number,
                        #          the more dense anglular grid is which may or may not improve accuracy of integration
    GRIDPOINT = 19 #selected MWA gridpoint (0 to 196) (see .csv file)

    PHI_POINTING = 153.4349 #151.46 #measured 'phi' angle on sky semisphere
    THETA_POINTING = 15.3729 #18.3 #measured 'theta' angle on sky semisphere

    #===============DATE OF OBSERVATION=================#

    YEAR = 2016 
    MONTH = 12
    DAY = 7 
    UTC_HR = 10
    UTC_MIN = 28
    UTC_SEC = 9
    FREQ_MIN = 50e6

    #===============LOCAL COORDINATES===================#

    LON = 116+ 40/60 +14.93/3600 # DEGREES,MINUTES,SECONDS
    LAT = -(26 + 42/60 + 11.95/3600) # DEGREES,MINUTES,SECONDS
    ALT = 377.8*1e-3

    #===============NAMES OF DATA FILES=================#

    #select one of measured SEFDs for comparison
    SEFD_DIR = 'Data/MWA_SEFD_'+str(POL)+'pol.mat'

    #multiport scattering parameters of simulated array (choose one polarisation)
    S_ARRAY_DIR = 'Data/MWA_Sparameter_from_Port_currents_rev3_'+str(POL)+'pol.s16p'

    #measured LNA's scattering parameters
    SDS_DUT_DIR = 'Data/EDA_Sds_DUT_comp.s2p'

    #spherical modal coefficients
    H5_DIR = 'Data/MWA_FEE_rev3_meas_LNA_imp.h5'

    #beamformer delays and measured angles
    BEAMFORMER_DELAYS_DIR = 'Data/MWA_sweet_spot_gridpoints.csv'

    #noise invariant parameters
    NOISE_PARA_DIR = 'Data/EDA_LNA_DISO_noisepar_170705.mat'
    
    #fits image containing map of the sky
    FITS_IMAGE = 'Data/radio408_RaDec.fits'

    #===========PHYSICAL CONSTANTS======================#

    #This modules contains definition of Global Constants
    T0 = 290 #Kelvin - Reference ambient temperature
    KB = 1.38064852e-23 #Joule/K - Boltzmann constant
    C0 = 299792458 #m/s - Speed of light in vacuum
    EPS0 = 8.85418781761e-12 #Farad/m - Permittivity of free space
    MU0 = 1.25663706143592e-6 #Henry/m - Permeability of free space
    ZF = 376.7303134619917 #Ohm - Impedance of free space
    SQRT_FAC = 7.743286873158916 #Ohm^0.5 - Factor that converts Jones matrix entries to Electric fields

    #==========DEFAULT DATA AND CONVERSION FACTORS======#

    N_ANT = 16 #number of antennas in the tile
    N_DIM = 32 #dimension of matrix elements
    F_NUM = 218 #number of frequency points
    RAD2DEG = 57.295779513082321 #rad/deg - multiplicative conversion factor to convert from radias to degrees
    DEG2RAD = 0.0174532925199433 #deg/rad - multiplicative conversion factor to convert from degrees to radians
    FLUXSITOJANSKY = 1.0e+26 #Jy/(W.m^-2.Hz^-1) - multiplicative conversion factor from Flux in SI units to Jansky
    JYTOKJY = 1.0e-3
    HZTOMHZ = 1.0e-6 #Hz/MHz - multiplicative conversion factor from Hz to MHz


