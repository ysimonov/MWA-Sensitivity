import numpy as np
from numpy.linalg import inv, multi_dot
from numpy import matmul, conj, transpose
from skrf import Network #used for reading .snp files

from constants import CONST #import all constants defined in the module
from npar2noise_wave import noise_waves

def conjt(m_arr): #conjugate transpose
    return transpose(conj(m_arr))

def mat_prod(a_mat,b_mat):
    return matmul(a_mat,b_mat)

def noise_matrices():

    #Specify data foulders
    sds_dut = Network(CONST.SDS_DUT_DIR).s
    s_array_pol = Network(CONST.S_ARRAY_DIR).s

    shape = (CONST.F_NUM, CONST.N_DIM, CONST.N_DIM)

    #matrix that holds scattering parameters of LNA
    s_lna = np.zeros(shape, dtype=np.complex128)
    s_array = np.zeros(shape, dtype=np.complex128)
    e_mat = np.zeros(shape, dtype=np.complex128)
    m_mat = np.zeros(shape, dtype=np.complex128)
    mp_mat = np.zeros(shape, dtype=np.complex128)
    msesm = np.zeros(shape, dtype=np.complex128)
    mem = np.zeros(shape, dtype=np.complex128)
    mnm = np.zeros(shape, dtype=np.complex128)
    imat = np.identity(CONST.N_DIM, dtype=np.float64)

    #Scattering parameters of DUT (LNA)
    s11_lna = sds_dut[:, 0, 0]
    s12_lna = sds_dut[:, 0, 1]
    s21_lna = sds_dut[:, 1, 0]
    s22_lna = sds_dut[:, 1, 1]

    z_load = 100.0 * (1.0 + s11_lna) / (1.0 - s11_lna) #100 is the characteristic impedance

    #Create noise correlation matrix
    n_mat = noise_waves(s11_lna, s21_lna)

    for i in range(0, CONST.N_DIM, 2):
        j = i + 1
        s_lna[:, i, i] = s11_lna[:]
        s_lna[:, j, j] = s22_lna[:]
        s_lna[:, i, j] = s12_lna[:]
        s_lna[:, j, i] = s21_lna[:]

    s_array[:, 0::2, 0::2] = s_array_pol[:, :, :]

    for i in range(0, CONST.F_NUM):

        e_mat[i, :, :] = imat[:, :] - mat_prod(s_array[i, :, :], conjt(s_array[i, :, :]))

        m_mat[i, :, :] = inv(imat[:, :] - mat_prod(s_lna[i, :, :], s_array[i, :, :]))

        mp_mat[i, :, :] = inv(imat[:, :] - mat_prod(s_array[i, :, :], s_lna[i, :, :]))

        msesm[i, :, :] = multi_dot([m_mat[i, :, :], s_lna[i, :, :], e_mat[i, :, :], \
                                  conjt(s_lna[i, :, :]), conjt(m_mat[i, :, :])])

        mem[i, :, :] = multi_dot([mp_mat[i, :, :], e_mat[i, :, :], conjt(mp_mat[i, :, :])])

        mnm[i, :, :] = multi_dot([m_mat[i, :, :], n_mat[i, :, :], conjt(m_mat[i, :, :])])

    return mnm, msesm, mem, z_load
