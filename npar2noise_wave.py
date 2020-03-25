import numpy as np
import scipy.io as sio #used for reading .mat files

from input_data import CONST

def abs_sqr(a_arr):
    return np.square(np.abs(a_arr))

def noise_waves(s11, s21):
    """
    computes correlated noise waves based on measured noise parameters Tmin,N,Gamma_opt
    input arrays: s11 and s21 have dimensions (0:F_NUM) are scattering 
    parameters of the LNA (DUT)
    """
    noise_par = \
    sio.loadmat(CONST.NOISE_PARA_DIR)#['noise_par'][0][0]

    tmin = np.array(noise_par['Tmin'][:, 0], dtype=np.float64)
    gamma_opt = np.array(noise_par['Gamma_opt'][:, 0], dtype=np.complex128)
    n_opt = np.array(noise_par['N'][:, 0], dtype=np.float64)

    tmin_t0 = tmin / CONST.T0

    tmp = 1.0-abs_sqr(gamma_opt)
    c11_kt0 = 4.0*n_opt*abs_sqr(1.0-s11*gamma_opt)/tmp-tmin_t0*(1.0-abs_sqr(s11))
    c22_kt0 = abs_sqr(s21)*(tmin_t0+4.0*n_opt*abs_sqr(gamma_opt)/tmp)
    c12_kt0 = -4.0*n_opt*np.conj(s21)*np.conj(gamma_opt)/tmp+s11*c22_kt0/s21
    c12_kt0_conj = np.conj(c12_kt0)

    #noise correlation matrix
    n_hat = np.zeros((CONST.F_NUM, CONST.N_DIM, CONST.N_DIM), dtype=np.complex128)
    for i in range(0, CONST.N_DIM, 2):
        j = i + 1
        n_hat[:, i, i] = c11_kt0[:]
        n_hat[:, j, j] = c22_kt0[:]
        n_hat[:, i, j] = c12_kt0[:]
        n_hat[:, j, i] = c12_kt0_conj[:]

    return n_hat
