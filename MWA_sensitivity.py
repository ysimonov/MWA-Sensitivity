#Author: Yevgeniy Simonov (simonov.yevgeniy@gmail.com)
#Date: March 2020

#Ref: Ung, D.C.X., 2019. Determination of Noise Temperature and Beam Modelling
#                        of an Antenna Array with Example Application using MWA
#                        (Doctoral dissertation, Curtin University).

#MATLAB script is written by Daniel Chu Xin Ung (daniel.ung@curtin.edu.au)

import time #measure time of code execution
import datetime #object that holds current date, used to specify observation time
import math
import numpy as np
from numpy.linalg import multi_dot
from datetime import date
import pandas as pd #used for reading .csv files
import scipy.io as sio #used for reading .mat files
import matplotlib.pyplot as plt #2D plots
import h5py #open .h5 files

from constants import CONST #import all constants defined in the module
from noise_mat import noise_matrices
from integral2D import integrate_on_sphere
from beam_pattern import get_beam_pattern, max_mode_size
from interpSky import skyatlocalcoord
from E_field import LegendreP

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

    start = time.time()

    freq_array = np.linspace(49920000, 327680000, 218)

    #create grid and project observed sky with different temperature values

    phi_1d = np.arange(0, N_PHI) * CONST.DEGS_PER_PIXEL * CONST.DEG2RAD
    theta_1d = np.arange(0, N_THETA) * CONST.DEGS_PER_PIXEL * CONST.DEG2RAD

    sky_temp = skyatlocalcoord(CONST.FREQ_MIN, phi_1d, theta_1d, \
                               OBSERVATION_TIME, CONST.LON, CONST.LAT)

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

    # contains spherical wave expansion coefficients used in calculation of Jones matrix (E-fields)
    H5_FILE = h5py.File(CONST.H5_DIR, 'r')

    # find the maximum order of beam modes for computation of Legendre polynomials
    Leg_order = max_mode_size(H5_FILE)

    #Pre-compute Legendre polynomials
    leg_deriv, leg_sin, index, costheta, sintheta, leg_deriv1, \
    leg_sin1 = LegendreP(theta_1d, Leg_order, THETA_POINTING)

    #Compute far field pattern for every frequency
    for i in range(0, F_NUM):

        ffout[i, :, :] = get_beam_pattern(H5_FILE, freq_array[i], w[i, :], \
        index, leg_deriv, leg_sin, phi_1d, theta_1d, False)

        int_p_lna[i] = integrate_on_sphere(phi_1d, theta_1d, sintheta, ffout[i])

        int_t_ext[i] = integrate_on_sphere(phi_1d, theta_1d, sintheta, ffout[i], sky_temp)

        ffout2[i] = get_beam_pattern(H5_FILE, freq_array[i], w[i, :], \
        index, leg_deriv1, leg_sin1, PHI_POINTING, THETA_POINTING, True)

    scaling = -2.55
    scaling_factor = np.power((freq_array / CONST.FREQ_MIN), scaling)

    pfactor = 4.0 * np.real(z_lna) / CONST.ZF
    p_lna_prime = pfactor * int_p_lna
    area_realised = 0.5 * pfactor * np.square(CONST.C0 / freq_array) * ffout2 #
    ant_temp = pfactor * scaling_factor * int_t_ext
    eff_rad = np.real(p_lna_prime / tau)
    sys_temp = ant_temp + np.real(tau) * (rcv_temp  + CONST.T0 * (1.0 - eff_rad))
    sefd = CONST.KB * sys_temp / area_realised * CONST.FLUXSITOJANSKY * CONST.JYTOKJY

    sefd_measured = np.ravel(sio.loadmat(CONST.SEFD_DIR)['masterfile']) * CONST.JYTOKJY
    freq_measured = np.ravel(sio.loadmat(CONST.SEFD_DIR)['master_freq']) #MHz

    threshold_min = 1e1
    threshold_max = 1e4

    for i in range(0, np.size(sefd_measured)):
        if(sefd_measured[i] > threshold_max or sefd_measured[i] < threshold_min):
            sefd_measured[i] = None

    freq_array = freq_array * CONST.HZTOMHZ

    end = time.time()
    print("Time elapsed = %.3f seconds" %(end-start))

    fig1 = plt.figure()
    ax = plt.gca()
    ax.set_yscale('log')
    ax.scatter(freq_measured, sefd_measured, label='SEFD Measured', marker='.')
    ax.scatter(freq_array, sefd, label='SEFD Calculated', marker='.')
    plt.xlabel('Frequency,MHz')
    plt.ylabel('System Equivalent Flux Density, kJy')
    plt.grid()
    plt.legend()
    fig1.savefig("SEFD_"+CONST.POL+"_POL.png")

    fig2, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig2.subplots_adjust(wspace=0, hspace=0)
    ax1.scatter(freq_array, eff_rad * 100, marker='.')
    ax2.scatter(freq_array, area_realised / tau, marker='.')
    ax2.set_xlabel('Frequency, MHz')
    ax1.set_ylabel(r'$\eta_{rad}$, %')
    ax2.set_ylabel(r'$\frac{A_{array}^{r}}{\tau}$, $m^{2}$')
    ax1.grid()
    ax2.grid()
    fig2.savefig("A_ARRAY_ETA_"+CONST.POL+"_POL.pdf")

    fig3, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    fig3.subplots_adjust(wspace=0, hspace=0)
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax1.scatter(freq_array, ant_temp / tau, marker='.')
    ax2.scatter(freq_array, rcv_temp, marker='.')
    ax3.scatter(freq_array, sys_temp, marker='.')
    ax3.set_xlabel('Frequency, MHz')
    ax1.set_ylabel(r'$T_{ant}$, K')
    ax2.set_ylabel(r'$T_{rcv}$, K')
    ax3.set_ylabel(r'$T_{sys}$, K')
    ax1.grid()
    ax2.grid()
    ax3.grid()
    fig3.savefig("TEMPERATURES_"+CONST.POL+"_POL.pdf")
