import math
import numpy as np
#import matplotlib.pyplot as plt #2D plots
from scipy.interpolate import griddata #2D interpolation of meshgrid data
from astropy.io import fits #used for reading .fits images
from input_data import CONST

def tand(angle_deg):
    return np.tan(angle_deg * np.pi / 180)

def sind(angle_deg):
    return np.sin(angle_deg * np.pi / 180)

def cosd(angle_deg):
    return np.cos(angle_deg * np.pi / 180)

def wrapto360(lon):
    return lon % 360

def wrapto180(lon):
    return np.arctan2(sind(lon), cosd(lon)) * 180 / np.pi #between -180 and 180

def julian_datetime(date):
    julian_date = math.floor(365.25 * (date.year + 4716.0)) + \
                  math.floor(30.6001 * (date.month + 1.0)) + 2.0 - \
                  math.floor(date.year / 100.0) + \
                  math.floor(math.floor(date.year / 100.0) / 4.0) + \
                  date.day - 1524.5 + (date.hour + date.minute / 60.0 + \
                  date.second / 3600.0) / 24.0
    return julian_date

def skyatlocalcoord(freq, phi_arr, theta_arr, observation_time, lon, lat):

    """
    This function creates the interpolated skymap for a signle lon, lat 
    and observation time. 
    """

    #get time of observation
    hour = observation_time.hour
    minute = observation_time.minute
    second = observation_time.second

    #create meshgrid from the arrays provided
    phi, theta = np.meshgrid(phi_arr, theta_arr, indexing='ij')
    FITS_IMAGE = fits.open(CONST.FITS_IMAGE)

    #retrieve image data
    image_data = np.array(FITS_IMAGE[0].data[0], dtype=np.float64)

    #Galactic longitude (Right Ascension) at reference point (DEG)
    crval1 = FITS_IMAGE[0].header['CRVAL1']

    #Galactic latitude (Declination) at reference point (DEG)
    crval2 = FITS_IMAGE[0].header['CRVAL2']

    #Pixel coordinate of reference point (RA)
    crpix1 = FITS_IMAGE[0].header['CRPIX1']

    #Pixel coordinate of reference point (DEC)
    crpix2 = FITS_IMAGE[0].header['CRPIX2']

    #x-scale, increment along axis
    cdelt1 = FITS_IMAGE[0].header['CDELT1']

    #y-scale, increment along axis
    cdelt2 = FITS_IMAGE[0].header['CDELT2']

    #apply appropriate scaling to sky image
    scaling = (freq / 408e6)**(-2.55)

    #scale by 0.1 to match the actual sky temperature
    skymap = scaling * np.float64(0.1 * image_data).T

    #arrange right ascention and declination points in ascending order
    arr_x = np.arange(1, np.size(skymap, 0) + 1)
    arr_y = np.arange(1, np.size(skymap, 1) + 1)

    #create right ascention and declination arrays
    ra_1d = crval1 + (arr_x - crpix1) * cdelt1
    dec_1d = crval2 + (arr_y - crpix2) * cdelt2

    #form grid from RA and DEC
    ra2d, dec2d = np.meshgrid(ra_1d, dec_1d, indexing='ij')

    #estimate Julian time
    julian_time = julian_datetime(observation_time)

    julian_date = julian_time - 2451543.5

    #Longitude of perihelion degrees
    longitude_perihelion = 282.9404 + 4.70935e-5 * julian_date

    #Mean anomaly degrees
    mean_anomaly = (356.0470 + 0.9856002585 * julian_date) % 360

    #Sun's mean longitude degrees
    mean_longitude = longitude_perihelion + mean_anomaly

    #UTH time in hours
    uth = hour + minute / 60 + second / 3600

    #get global mean siderial time
    gmst0 = ((mean_longitude + 180) % 360) / 15

    #Local Siderial time
    sidtime = gmst0 + uth + lon / 15

    #calculate hour angle
    hour_angle = wrapto180(sidtime * 15) - ra2d

    #project from astronomical coordinates onto Cartesian coordinates
    x_coord = cosd(hour_angle) * cosd(dec2d)
    y_coord = sind(hour_angle) * cosd(dec2d)
    z_coord = sind(dec2d)

    #Rotate along an axis going east-west
    xhor = x_coord * cosd(90.0 - lat) - z_coord * sind(90.0 - lat)
    yhor = y_coord
    zhor = x_coord * sind(90.0 - lat) + z_coord * cosd(90.0 - lat)

    #FIND AZ and EL and Limit Azimuth range from -180 to 180 deg
    az2d = np.arctan2(yhor, xhor) * CONST.RAD2DEG + 180
    el2d = np.arcsin(zhor) * CONST.RAD2DEG

    #Limit Azimuth range from -180 to 180 deg
    az_wrap = wrapto180(az2d)

    #create a mask to fiter out negative elevation angles that are below horizon
    mask = el2d < 0

    kx_sky = cosd(el2d) * sind(az_wrap)
    ky_sky = cosd(el2d) * cosd(az_wrap)

    #apply mask on kx and ky coordinates
    kx_sky_new = kx_sky[~mask].ravel()
    ky_sky_new = ky_sky[~mask].ravel()

    #collect coordinates into columns for interpolation
    kxky_sky_col = np.column_stack((kx_sky_new, ky_sky_new))

    #apply mask on skymap and flatten array for interpolation
    skymap_new = skymap[~mask].ravel()

    #create azimuth and elevation coordinates from phi and theta arrays used in the main program
    az_beam = wrapto360(90 - (phi * CONST.RAD2DEG))
    el_beam = 90 - (theta * CONST.RAD2DEG)

    #create 'artificial' sky coordinates for interpolation
    kx_beam = cosd(el_beam) * sind(az_beam)
    ky_beam = cosd(el_beam) * cosd(az_beam)

    #interpolate onto kx_beam and ky_beam where evevation angles >=0
    local_sky = griddata(kxky_sky_col, skymap_new, (kx_beam, ky_beam), 'cubic')

    #apply mask to remove NaNs from interpolated data
    mask2 = np.isnan(local_sky)

    #create new coordinates that do not contain invalid points, i.e., points below horizon
    kx_beam_new = kx_beam[mask2]
    ky_beam_new = ky_beam[mask2]

    #fill 'missing' values with interpolated values
    temp = griddata(kxky_sky_col, skymap_new, (kx_beam_new, ky_beam_new), 'nearest')

    local_sky[mask2] = temp

#    contours = plt.contour(kx_beam, ky_beam, local_sky, 1000, cmap='gist_stern')
#    plt.xlabel('kx')
#    plt.ylabel('ky')
#    plt.title('Sky Map at '+str(freq/1e6)+' MHz')
#    plt.colorbar()
#    plt.show()

    return local_sky
