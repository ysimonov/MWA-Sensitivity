import numpy as np
import h5py #used for reading .h5 files (spherical wave coefficients)

from E_field import construct_FF, construct_FF1, extract_modes

def abs_sqr(array_x):
    return np.square(np.abs(array_x))

#Determine maximum orders for computation of Legendre Polynomials
def max_mode_size(h5_file):
    q_modes_all = h5_file['modes'][()].T
    n_max = int(np.amax(q_modes_all[:, 2]))
    return n_max

def get_beam_pattern(h5_file, freq, beamformer_coeff, index, \
                     leg_deriv, leg_sin, phi, theta, single=False):

    """
    This routine returns Far-field pattern |E_theta|^2 + |E_phi|^2
    phi - angles of phi array in radians
    theta - angles of theta array in radians
    single - if True, calculates beam pattern for a single sky pointing only
    index - array containing array indexes for combination [n,m] of 
    Associated Legendre polynomials
    """

    target_frequency = int(freq)

    beam_modes = {}
    beam_modes = extract_modes(h5_file, beamformer_coeff, target_frequency) #returns beam modes

    #calculate the electric field components for a single point on the sky
    if single is True:

        efield_theta, efield_phi = construct_FF1(phi, theta, index, leg_deriv, leg_sin, beam_modes) 

    #calculate the electric field components for the entire visible sky (hemisphere)
    else:

        efield_theta, efield_phi = construct_FF(phi, theta, index, leg_deriv, leg_sin, beam_modes)

    return abs_sqr(efield_theta) + abs_sqr(efield_phi)

