#from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline

def integrate_on_sphere(phi, theta, sintheta, far_field_pattern, sky_temp=None):
    '''

    Parameters
    ----------
    phi : float64 (array)
        array of phi angles in spherical coordinate system in radians.
    theta : float64 (array)
        array of theta angles in spherical coordinate system in radians.
    sintheta : float64 (array)
        sin(theta).
    far_field_pattern : float64 (array)
        magnitude of electric fields squared as a function of angles (phi,theta).
    sky_temp : float64
        default - none. Sky temperatures extracted from Haslam map.

    Returns
    -------
    integral : float64
        integrated value of far_field_pattern (and sky_temp)

    '''
    if sky_temp is None: 
        integrand = far_field_pattern * sintheta
    else:
        integrand = far_field_pattern * sky_temp * sintheta

    #this method is supposed to be more accurate than simpson's rule, but ~10 seconds slower
    integral = RectBivariateSpline(phi, theta, integrand, \
    kx=5,ky=5).integral(phi[0],phi[-1],theta[0],theta[-1])

    # dtheta = theta[1] - theta[0]
    # dphi = phi[1] - phi[0]
#    integral = simps(simps(integrand, theta, dtheta, 1), phi, dphi, 0)

    return integral
