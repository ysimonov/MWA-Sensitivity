#Author: Yevgeniy Simonov (simonov.yevgeniy@gmail.com)
#Date: March 2020

#Ref: Ung, D.C.X., 2019. Determination of Noise Temperature and Beam Modelling
#                        of an Antenna Array with Example Application using MWA
#                        (Doctoral dissertation, Curtin University).

#MATLAB script is written by Daniel Chu Xin Ung (daniel.ung@curtin.edu.au)

import os
import sys
import time #measure time of code execution
import datetime #object that holds current date, used to specify observation time
import math
import numpy as np
from numpy.linalg import multi_dot
import pandas as pd #used for reading .csv files
import scipy.io as sio #used for reading .mat files
import matplotlib.pyplot as plt #2D plots
import h5py #open .h5 files
import warnings

from input_data import CONST #import all constants defined in the module
from noise_mat import noise_matrices
from integral2D import integrate_on_sphere
from beam_pattern import get_beam_pattern, max_mode_size
from interpSky import skyatlocalcoord
from E_field import LegendreP

if sys.platform == 'win32':
    # On Windows, the best timer is time.clock
    default_timer = time.clock
else:
    # On most other platforms the best timer is time.time
    default_timer = time.time

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    OBSERVATION_TIME = datetime.datetime(CONST.YEAR, CONST.MONTH, CONST.DAY, \
                                         CONST.UTC_HR, CONST.UTC_MIN, CONST.UTC_SEC)

    N_ANT = CONST.N_ANT
    F_NUM = CONST.F_NUM
    PHI_POINTING = CONST.PHI_POINTING * CONST.DEG2RAD
    THETA_POINTING = CONST.THETA_POINTING * CONST.DEG2RAD

    N_THETA = int(90.0 / CONST.DEGS_PER_PIXEL) + 1 #number of 'theta' points
    N_PHI = int(360.0 / CONST.DEGS_PER_PIXEL) + 1 #number of 'phi' points

    # extract csv time delays for 4x4 array and angles of observation
    FILE_PATH = open(CONST.BEAMFORMER_DELAYS_DIR)
    data_csv = pd.read_csv(FILE_PATH, sep=',', comment='#', \
    skipinitialspace=True, skip_blank_lines=True, \
    error_bad_lines=False, warn_bad_lines=True).sort_index()

    data_csv = np.array(data_csv, dtype=np.float64) #convert to numpy object

    #allocate arrays
    w = np.ones((F_NUM, N_ANT), dtype=np.complex128)
    p_int = np.zeros((F_NUM), dtype=np.complex128)
    p_ext = np.zeros((F_NUM), dtype=np.complex128)
    p_rcv = np.zeros((F_NUM), dtype=np.float64)
    p_in = np.zeros((F_NUM), dtype=np.complex128)
    p_out = np.zeros((F_NUM), dtype=np.complex128)
    tau = np.zeros((F_NUM), dtype=np.complex128)
    rcv_temp = np.zeros((F_NUM), dtype=np.float64)

    ffout = np.zeros((F_NUM, N_PHI, N_THETA), dtype=np.float64)
    ffout2 = np.zeros((F_NUM), dtype=np.float64)
    int_p_lna = np.zeros((F_NUM), dtype=np.float64)
    int_t_ext = np.zeros((F_NUM), dtype=np.float64)
    area_realised = np.zeros((F_NUM), dtype=np.float64)

    #start timer
    start = default_timer()

    freq_array = np.linspace(49920000, 327680000, 218)

    #create grid and project observed sky with different temperature values

    phi_1d = np.arange(0, N_PHI) * CONST.DEGS_PER_PIXEL * CONST.DEG2RAD
    theta_1d = np.arange(0, N_THETA) * CONST.DEGS_PER_PIXEL * CONST.DEG2RAD

    print("----- extracting and interpolating sky map ------")
    sky_temp = skyatlocalcoord(CONST.FREQ_MIN, phi_1d, theta_1d, \
                               OBSERVATION_TIME, CONST.LON, CONST.LAT)
    end1 = default_timer()
    print("Time elapsed = %.3f seconds" %(end1-start))

    print('\n')
    print('----- calculating receiver noise parameters ------')

    #compute receiver's noise temperature ant parameter Tau
    mnm, msesm, mem, z_lna = noise_matrices()

    w_amp = 1.0 / math.sqrt(N_ANT)
    for i in range(0, N_ANT):
        w[:, i] = w_amp * np.exp(1j * 2.0 * np.pi * freq_array[:] * \
        (-data_csv[CONST.GRIDPOINT-1][i+3]) * 435e-12)
    wc = np.conj(w)

    for i in range(0, F_NUM):
        p_int[i] = multi_dot([w[i, :], mnm[i, 1::2, 1::2], wc[i, :]])
        p_ext[i] = multi_dot([w[i, :], msesm[i, 1::2, 1::2], wc[i, :]])

    rcv_temp[:] = np.real(CONST.T0 * p_int[:] / p_ext[:])

    for i in range(0, F_NUM):
        p_in[i] = multi_dot([w[i, :], mem[i, 0::2, 0::2], wc[i, :]])
        p_out[i] = multi_dot([w[i, :], msesm[i, 0::2, 0::2], wc[i, :]])

    tau[:] = p_in[:] - p_out[:]

    end2 = default_timer()
    print("Time elapsed = %.3f seconds" %(end2-end1))
    print('\n')

    # contains spherical wave expansion coefficients used in calculation of Jones matrix (E-fields)
    H5_FILE = h5py.File(CONST.H5_DIR, 'r')

    # find the maximum order of beam modes for computation of Legendre polynomials
    Leg_order = max_mode_size(H5_FILE)

    print('----- calculating Associated Legendre polynomials -----')
    #Pre-compute Legendre polynomials
    leg_deriv, leg_sin, costheta, sintheta, leg_deriv1, \
    leg_sin1 = LegendreP(theta_1d, Leg_order, THETA_POINTING)

    end3 = default_timer()
    print("Time elapsed = %.3f seconds" %(end3-end2))
    print("\n")

    print('----- calculating far field patter at every frequency -----')
    #Compute far field pattern for every frequency

    quarter_loop = int(F_NUM/4)
    half_loop = int(F_NUM/2)
    almost_there = quarter_loop*3

    for i in range(0, F_NUM):

        ffout[i, :, :] = get_beam_pattern(H5_FILE, freq_array[i], w[i, :], \
                         leg_deriv, leg_sin, phi_1d, theta_1d, False)

        int_p_lna[i] = integrate_on_sphere(phi_1d, theta_1d, sintheta, ffout[i])

        int_t_ext[i] = integrate_on_sphere(phi_1d, theta_1d, sintheta, ffout[i], sky_temp)

        ffout2[i] = get_beam_pattern(H5_FILE, freq_array[i], w[i, :], \
                    leg_deriv1, leg_sin1, PHI_POINTING, THETA_POINTING, True)

        if(i == quarter_loop):
            print('25% completed ...')
        elif(i == half_loop):
            print('50% completed ...')
        elif(i == almost_there):
            print('75% completed ...')

    end4 = default_timer()
    print("Time elapsed = %.3f seconds" %(end4-end3))
    print('\n')

    scaling = -2.55
    scaling_factor = np.power((freq_array / CONST.FREQ_MIN), scaling)

    pfactor = 4.0 * np.real(z_lna) / CONST.ZF
    p_lna_prime = pfactor * int_p_lna
    area_realised = 0.5 * pfactor * np.square(CONST.C0 / freq_array) * ffout2 # 
    ant_temp = pfactor * scaling_factor * int_t_ext
    eff_rad = np.real(p_lna_prime / tau)
    sys_temp = np.real(ant_temp + tau * (rcv_temp  + CONST.T0 * (1.0 - eff_rad)))
    sefd = CONST.KB * sys_temp / area_realised * CONST.FLUXSITOJANSKY * CONST.JYTOKJY

    end = default_timer()
    print("Total run time = %.3f seconds" %(end-start))

    #prepare plots of calculated vs measured SEFD
    sefd_measured = sio.loadmat(CONST.SEFD_DIR)['masterfile'] #Jansky
    freq_measured = sio.loadmat(CONST.SEFD_DIR)['master_freq'] #MHz

    #clean zero points (invalid measurements)
    mask = sefd_measured == 1
    sefd_measured[mask] = None
    freq_measured[mask] = None

    #clean lower outliers
    mask2 = sefd_measured > 3e7
    sefd_measured[mask2] = None
    freq_measured[mask2] = None

    #clean upper outliers
    mask3 = sefd_measured < 1e4
    sefd_measured[mask3] = None
    freq_measured[mask3] = None

    #divide data into "bunches" of smaller data for averaging
    nsamples = 896*8
    master_freq = np.reshape(freq_measured.T, (nsamples, int(np.prod(np.size(freq_measured))/nsamples)))
    master_sefd = np.reshape(sefd_measured.T, (nsamples, int(np.prod(np.size(sefd_measured))/nsamples)))

    #average frequency and sefd scattered data
    freq_avg = np.nanmean(master_freq, axis=1, dtype=np.float64)
    sefd_avg = np.nanmean(master_sefd, axis=1, dtype=np.float64) * CONST.JYTOKJY

    #change Hz to MHz for plotting
    freq_array = freq_array * CONST.HZTOMHZ

    #create a directory (if it does not exist)
    path_directory = "Results/"

    # define the access rights
    access_rights = 0o755

    #check if directory exists
    if(os.path.exists(path_directory) is not True):
        try:
            os.mkdir(path_directory, access_rights)
        except OSError:
            print ("Creation of the directory %s failed" % path_directory)
        else:
            print ("Successfully created the directory %s " % path_directory)

    #create sub-directory for results
    path_to_results = path_directory + \
    CONST.POL + "_POL_" + str(CONST.PHI_POINTING) + "_" + str(CONST.THETA_POINTING) + "/"

    if(os.path.exists(path_to_results) is not True):
        try:
            os.mkdir(path_to_results, access_rights)
        except OSError:
            print ("Creation of the directory %s failed" % path_to_results)
        else:
            print ("Successfully created the directory %s " % path_to_results)

    #create plots
    plot_names = ['SEFD', 'radiation_efficiency', 'effective_area', \
                  'antenna_temperature', 'receiver_temperature', \
                  'system_temperature', 'realised_area']

    plot_format = '.png'

    ylabels = ['System Equivalent Flux Density, kJy', \
            'Radiation Efficiency, %', r'Effective Area, $m^{2}$', \
            'Antenna Noise Temperature, K', 'Receiver Noise Temperature, K', \
            'System Noise Temperature, K', r'Realised Area, $m^{2}$']

    #create a dictionary for plotting
    dict_arrays = {0 : sefd, \
                   1 : eff_rad * 100, \
                   2 : area_realised / tau, \
                   3 : ant_temp / tau, \
                   4 : rcv_temp, \
                   5 : sys_temp, \
                   6 : area_realised}

    axes_scales = ['log', 'linear', 'linear', 'log', 'log', 'log', 'linear']

    for i in range(0, 7):
        fig = plt.figure(dpi=1000)
        ax = plt.gca()
        if(i != 0):
            plt.plot(freq_array, dict_arrays[i], '-', lw=2)
        else:
            ax.scatter(freq_avg, sefd_avg, marker = "x", label=\
            r'SEFD Measured at $\phi = 151.46^{o}, \theta = 18.3^{o}$', s=5)
            plt.plot(freq_array, dict_arrays[i], '-', \
                     lw=2, color='r', label='SEFD Calculated')
            ax.legend()
        plt.xlabel('Frequency, MHz')
        plt.ylabel(ylabels[i])
        ax.set_axisbelow(True)
        ax.set_yscale(axes_scales[i])
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.tick_params(which='both', # Options for both major and minor ticks
                       top='off', # turn off top ticks
                       left='off', # turn off left ticks
                       right='off',  # turn off right ticks
                       bottom='off') # turn off bottom ticks
        fig.savefig(path_to_results + plot_names[i] + plot_format)
