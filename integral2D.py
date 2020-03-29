#from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline

def integrate_on_sphere(phi, theta, sintheta, far_field_pattern, sky_temp=None):

    if sky_temp is None: 
        integrand = far_field_pattern * sintheta
    else:
        integrand = far_field_pattern * sky_temp * sintheta

    #this method is supposed to be more accurate than simpson's rule, but ~10 seconds slower
    integral = RectBivariateSpline(phi, theta, integrand, \
    kx=5,ky=5).integral(phi[0],phi[-1],theta[0],theta[-1])


#    dtheta = theta[1] - theta[0]
#    dphi = phi[1] - phi[0]    
#    integral = simps(simps(integrand, theta, dtheta, 1), phi, dphi, 0)

    return integral
