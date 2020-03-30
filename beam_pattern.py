import numpy as np
import h5py #used for reading .h5 files (spherical wave coefficients)

from E_field import construct_FF, construct_FF1, extract_modes

def abs_sqr(array_x):
    return np.square(np.abs(array_x))

#Determine maximum orders for computation of Legendre Polynomials
def max_mode_size(h5_file):
    '''

    Parameters
    ----------
    h5_file : dictionary
        file containing degrees and orders of spherical harmonics.

    Returns
    -------
    n_max : integer
        highest order of spherical harmonics found in h5_file.

    '''
    q_modes_all = h5_file['modes'][()].T
    n_max = int(np.amax(q_modes_all[:, 2]))
    return n_max

def get_beam_pattern(h5_file, freq, beamformer_coeff, \
                     leg_deriv, leg_sin, phi, theta, single=False):
    '''

    Parameters
    ----------
    h5_file : dictionary
        file containing spherical modal coefficients,
        degrees and orders of spherical harmonics.
    freq : float64
        a single frequency point in Hz
    beamformer_coeff : complex128
        the values of complex beamformer coefficients at frequency freq
    leg_deriv : float64
        derivatives of (normalised) associated legendre polynomials.
    leg_sin : float64
        values of (normalised) associated legendre over sin(theta).
    phi : float64
        array of phi values in spherical coordinate system (rad).
    theta : float64
        array of theta values in spherical coordinate system (rad).

    Returns
    -------
    e_theta^2 + e_phi^2 : float64
        magnitude of the electric field squared.

    '''
    target_frequency = int(freq)

    beam_modes = {}
    beam_modes = extract_modes(h5_file, beamformer_coeff, target_frequency) #returns beam modes

    #calculate the electric field components for a single point on the sky
    if single is True:

        efield_theta, efield_phi = construct_FF1(phi, theta, leg_deriv, leg_sin, beam_modes) 

    #calculate the electric field components for the entire visible sky (hemisphere)
    else:

        efield_theta, efield_phi = construct_FF(phi, theta, leg_deriv, leg_sin, beam_modes)

    return abs_sqr(efield_theta) + abs_sqr(efield_phi)

