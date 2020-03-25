import math
import decimal
from decimal import Decimal
import numpy as np
import h5py
from input_data import CONST
decimal.getcontext().prec = CONST.LEGENDRE_DECIMAL_PRECISION #decimal point precision

#AUTHOR: Yevgeniy Simonov, 2020

#computes Pochhammer's symbol z_m
#see: https://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.special.poch.html
def pochs(z, m):
    """
    pochs(z,m) = (z+m-1)!/(z-1)! = Gamma(z+m)/Gamma(z) for positive z,m
    """
    prod = Decimal(z) 
    if m > 0:
        j = Decimal(z)
        while j < (z + m - 1):
            prod *= Decimal(j + 1)
            j += Decimal(1)
        return prod
    elif m == 0.: #zero
        return Decimal(1)

#factorial in Decimal Precision
#see: https://docs.python.org/3/library/decimal.html
def fact(k):
    """
    extended precision factorial function for positive arguments k>=0
    """
    if(k<0):
        return None
    elif(k==1 or k==0):
        return Decimal(1)
    else:
        prod = Decimal(1)
        j = Decimal(1)
        while(j <= k):
            prod = prod * Decimal(j)
            j += Decimal(1)
        return prod

#this function returns the index of Legendre Polynomials
def PT(l,m):
    """
    function that returns combined indexes of Associated Legendre polynomials
    and their derivatives.
    """
    return int((m)+((l)*((l)+1))/2)

#For more details regarding computation of fully normalized Associated Legendre polynomials and 
#Spherical harmonics, refer to the article below:
#Limpanuparb, T. and Milthorpe, J., 2014. 
#Associated Legendre Polynomials and Spherical Harmonics Computation for Chemistry Applications. 
#arXiv preprint arXiv:1410.1748.
def computeP(L, idx, A, B, P, x):
    """
    function that computes Normalised Associated Legendre polynomials
    using recurrence relations, for a single argument x = cos(theta)
    for all orders n=0..L, |m|=0..L
    ->There is no phase term (-1)^M in this implementation
    """
    sintheta = math.sqrt(1.-x*x)
    temp = math.sqrt(0.5)
    P[idx[0, 0]] = 0.7071067811865475244
    if(L>0):
        SQRT3 = 1.7320508075688772935
        P[idx[1, 0]] = x*SQRT3*temp
        SQRT3DIV2 = -1.2247448713915890491
        temp = SQRT3DIV2*sintheta*temp
        P[idx[1, 1]]=temp
        for l in range(2, L+1):
            for m in range(0, l-1):
                P[idx[l, m]]=A[idx[l, m]]*(x*P[idx[l-1, m]] + \
                             B[idx[l, m]]*P[idx[l-2, m]])
            P[idx[l, l-1]]=x*math.sqrt(2*(l-1)+3)*temp
            temp=-math.sqrt(1.0+0.5/l)*sintheta*temp
            P[idx[l, l]]=temp
    return P

#For more information regarding P/sin term, read
#Li, P. and Jiang, L.J., 2012. 
#The far field transformation for the antenna modeling based on spherical electric field measurements. 
#Progress In Electromagnetics Research, 123, pp.243-261.
def LegendreP(theta, N_max, source_theta=None):
    '''
    This routine generates derivatives of normalised Associated Legendre polynomials
    and Legendre / sin(theta) for evaluation of multipole expansion of electric fields
    source_theta is an optional parameter that permits evaluation of Associated Legendre
    functions for this particular angle (source_theta)
    '''
    NPTS = len(theta)
    safety_factor = 5
    LL = N_max + 1 + safety_factor

    x = np.cos(theta)
    y = np.sin(theta)

    #Size of n,m combined 
    size = PT(LL+1, LL+1)

    #Indexes of Legendre polynomials
    idx = np.zeros((LL+1, LL+1), np.int64)
    for l in range(0, LL+1):
        for m in range(0, LL+1):
            idx[l, m] = PT(l, m) #indexes of Legendre polynomials

    #Temporary arrays used in recursion
    A = np.zeros((size), dtype=np.float64) 
    B = np.zeros((size), dtype=np.float64)
   
    #Legendre Polynomials
    P = np.zeros((size), dtype=np.float64) 

    #Compute temporary arrays 
    for l in range(2, LL+1):
        ls = l*l
        lm1s = (l-1)*(l-1)
        for m in range(0, l-1):
            ms = m*m
            A[idx[l, m]] = math.sqrt((4*ls-1.)/(ls-ms))
            B[idx[l, m]] = -math.sqrt((lm1s-ms)/(4*lm1s-1.))

    #Allocate space for Legendre polynomials
    Leg = np.zeros((size, NPTS), dtype=np.float64) # [size=(L,M),Theta]

    #Compute all Associated legendre functions on x-th grid
    for i in range(0, len(x)):
        Leg[:, i] = computeP(LL, idx, A, B, P, x[i])

    #Compute P/sin functions, general case:
    Leg_sin = np.zeros((size, NPTS), np.float64)

    #consider asymptotic cases when theta = 0 or theta = pi
    eps = 1e-12
    mask1 = (abs(theta-0.)<eps)
    mask2 = (abs(theta-math.pi)<eps)

    #combine masks
    mask = np.ma.mask_or(mask1,mask2)

    #find location of asymptotes
    asymptote = np.where(mask)[0]

    #find location of regular points 
    regular = np.where(~mask)[0]

    #evaluate function at regular points
    Leg_sin[:,regular] = Leg[:,regular] / y[regular]

    #Consider special case for m=1 at theta=0:
    m = 1
    for n in range(1,LL+1):
        Leg_sin[idx[n, m],asymptote[0]] = -0.5 * np.sqrt(n*(2.0*n+1.0)*(n+1.0)/2.0)
    if(np.shape(asymptote)==(2,)): #include theta == pi case
        for n in range(1,LL+1):
            Leg_sin[idx[n,m],asymptote[1]] = (-1) ** (n+1) * Leg_sin[idx[n,m],asymptote[0]]

    #Evaluate derivatives of normalized Associated Legendre polynomials using recurrence relation 
    Leg_deriv = np.zeros((size, NPTS), np.float64)
    for n in range(1, LL):
        for m in range(0, n):
            Leg_deriv[idx[n, m], :] = -(n+1) * x[:] * Leg_sin[idx[n, m],:] + \
                                      np.sqrt((2*n+1)*(n+m+1)*(n-m+1)/(2*n+3)) * \
                                      Leg_sin[idx[n+1, m],:]

    #Compute for a single pointing if present
    if(source_theta is not None):

        Leg_deriv1 = np.zeros((size), dtype=np.float64)
        Leg_sin1 = np.zeros((size), dtype=np.float64)

        #check if source_theta is a part of existing array:
        if source_theta in theta:

            #find the position of theta corresponding to source_theta (assuming all values of theta are unique)
            pos = np.where(theta == source_theta)

            Leg_deriv1[:] = Leg_deriv[:, pos].T
            Leg_sin1[:] = Leg_sin[:, pos].T

        else: #compute legendre polynomial and sin term for this unique position

            P = np.zeros((size), dtype=np.float64)
            Leg_deriv1 = np.zeros((size), np.float64)
            Leg_pol1 = computeP(LL, idx, A, B, P, np.cos(source_theta))
            Leg_sin1 = Leg_pol1 / np.sin(source_theta)
            for n in range(0, LL):
                for m in range(0, n):
                    Leg_deriv1[idx[n, m]] = -(n+1) * np.cos(source_theta) * \
                                            Leg_sin1[idx[n, m]] + \
                                            np.sqrt((2*n+1)*(n+m+1)*(n-m+1)/(2*n+3)) * \
                                            Leg_sin1[idx[n+1, m]]

        return Leg_deriv, Leg_sin, idx, x, y, Leg_deriv1, Leg_sin1

    else:

        return Leg_deriv, Leg_sin, idx, x, y

#spherical modes used to re-construct electric fields
def extract_modes(h5f, Vcplx, freq):

    """
    h5f - .h5 file containing shperical modes
    Vcplx - complex beamformer coefficients
    freq - single frequency point
    Aim: extracts modal coefficients from h5 file and
    sums them up, scaling by beamformer coefficients.
    They are used to reconstruct electric fields from FEKO simulation
    N,M are arrays containing all combinations of indexes required to reconstruct
    spherical vector harmonics
    """

    beam_modes = {}
    N_ANT = CONST.N_ANT
    DEG2RAD = CONST.DEG2RAD
    POL = CONST.POL

    # finding maximum length of modes for this frequency

    max_length = 0  # initialize
    for ant_i in range(0, N_ANT):

        # select spherical wave table
        name = '%s%s_%s' % (POL, ant_i + 1, freq)

        # find maximum length
        if h5f[name].shape[1] // 2 > max_length:
            max_length = h5f[name].shape[1] // 2

    # accumulating spherical harmonics coefficients for the array
    Q1_accum = np.zeros(max_length, dtype=np.complex128)
    Q2_accum = np.zeros(max_length, dtype=np.complex128)

    # Read in modes
    Q_modes_all = h5f['modes'][()].T
    Nmax = 0
    M_accum = None
    N_accum = None

    for ant_i in range(0, N_ANT):

        # re-initialise Q1 and Q2 for every antenna
        Q1 = np.zeros(max_length, dtype=np.complex128)
        Q2 = np.zeros(max_length, dtype=np.complex128)

        # select spherical wave table
        name = '%s%s_%s' % (POL, ant_i + 1, freq)
        Q_all = h5f[name][()].T

        # current length
        my_len = np.max(Q_all.shape)
        my_len_half = my_len // 2

        Q_modes = Q_modes_all[0:my_len, :]  # Get modes for this antenna
        # convert Qall to M, N, Q1, Q2 vectors for processing

        # find s=1 and s=2 indices for this antenna
        s1 = Q_modes[0:my_len, 0] <= 1 #TE mode
        s2 = Q_modes[0:my_len, 0] > 1 #TM mode

        # grab m,n vectors
        M = Q_modes[s1, 1]
        N = Q_modes[s1, 2]

        # update to the larger M and N
        if np.max(N) > Nmax:
            M_accum = M
            N_accum = N
            Nmax = np.max(N_accum)

        # grab Q1mn and Q2mn and make them complex
        Q1[0:my_len_half] = Q_all[s1, 0] * np.exp(1j * Q_all[s1, 1] * DEG2RAD)
        Q2[0:my_len_half] = Q_all[s2, 0] * np.exp(1j * Q_all[s2, 1] * DEG2RAD)

        # accumulate Q1 and Q2, scaled by excitation voltage
        Q1_accum += Q1 * Vcplx[ant_i]
        Q2_accum += Q2 * Vcplx[ant_i]

    beam_modes = {'Q1': Q1_accum, 'Q2': Q2_accum, 'M': M_accum, 'N': N_accum}

    return beam_modes

def construct_FF(phi, theta, idx, leg_deriv, leg_sin, beam_modes):

    """
    This routine generates E(theta,phi)(hat(theta)+hat(phi)) for all
    angles theta and phi (arrays)
    idx - index to Associated Legendre values of n,m
    beam_modes - spherical modal coefficients
    leg_deriv - d(P(cos(theta)))/d(theta) [Normalised]
    leg_sin - P(cos(theta))/sin(theta) [Normalised]
    """

    Phi_max = np.size(phi)
    Theta_max = np.size(theta)

    target = beam_modes
    dim = len(target['N']) #dimension of summation

    Q1 = target['Q1'] #extract i-th Q1 modal coefficient
    Q2 = target['Q2'] #extract i-th Q2 modal coefficient
    M = target['M'].astype(np.int64)
    N = target['N'].astype(np.int64)

    Factor = np.zeros((Phi_max, dim), dtype=np.complex128)
    Ld = np.zeros((Theta_max, dim), dtype=np.float64) #Associated Legendre Polynomials
    Ls = np.zeros((Theta_max, dim), dtype=np.float64) #Associated Legendre over sin term

    #Arrays that will be used to search the index on Legendre polynomials
    Ma = np.abs(M)

    #sign_factor = (-M/|M|)^M. Note that at M=0, the limit -> 1, but the result is undefined if
    #computed normally. Therefore, for M>0 invert sign for all odd values of M.
    #For M<=0 the sign remains the same. 
    sign_factor = np.ones(dim, dtype=np.float64)
    sign_factor[(M > 0) & (M % 2 != 0)] = -1.0

    #outer product is required to evaluate exp(jm(phi)). 
    phase_factor = np.outer(phi, M) #size[phi_max,dim]

    #this is 1/sqrt(n*(n+1)) * (1j)^(N) * {(-M/|M|)^(M)} * exp(jm(phi))
    phi_factor = np.power(1j, N) * sign_factor / np.sqrt(N*(N+1.0)) * np.exp(1j * phase_factor)

    #Map functions to correct values
    for i in range(0, dim):
        MT = Ma[i]
        NT = N[i]
        index = idx[NT, MT]
        Ld[:, i] = leg_deriv[index, :]
        Ls[:, i] = leg_sin[index, :]

    e_th = (Ld*Q2-M*Q1*Ls)
    e_ph = (M*Q2*Ls-Ld*Q1)*1j

    E_Theta = np.inner(phi_factor, e_th) * CONST.SQRT_FAC
    E_Phi = np.inner(phi_factor, e_ph) * CONST.SQRT_FAC

    return E_Theta, E_Phi

# ================================================================================================================
   
def construct_FF1(phi1, theta1, idx, leg_deriv, leg_sin, beam_modes):

    """
    This routine generates E(theta,phi)(hat(theta)+hat(phi)) for a
    single pointing on the sky (phi1, theta1).
    idx - index to Associated Legendre values of n,m
    beam_modes - spherical modal coefficients
    leg_deriv - d(P(cos(theta)))/d(theta) [Normalised]
    leg_sin - P(cos(theta))/sin(theta) [Normalised]
    """

    target = beam_modes

    dim = len(target['N']) #dimension of summation

    Ld = np.zeros((dim), dtype=np.float64) #Associated Legendre Polynomials
    Ls = np.zeros((dim), dtype=np.float64) #Associated Legendre over sin term

    M = target['M'].astype(np.int64)
    N = target['N'].astype(np.int64)
    Q1 = target['Q1'] #extract i-th Q1 modal coefficient
    Q2 = target['Q2'] #extract i-th Q2 modal coefficient

    Ma = np.abs(M)

    #sign_factor = (-M/|M|)^M. Note that at M=0, the limit -> 1, but the result is undefined if
    #computed normally. Therefore, for M>0 invert sign for all odd values of M.
    #For M<=0 the sign remains the same. 
    sign_factor = np.ones(dim, dtype=np.float64)
    sign_factor[(M > 0) & (M % 2 != 0)] = -1.0

    #this is 1/sqrt(n*(n+1)) * (1j)^(N) * {(-M/|M|)^(M)} * exp(jm(phi))
    phi_factor = np.power(1j, N) * sign_factor / np.sqrt(N*(N+1.0)) * np.exp(1j * phi1 * M)
   
    #Map functions to correct values
    for i in range(0, dim):
        MT = Ma[i]
        NT = N[i]
        index = idx[NT, MT]
        Ld[i] = leg_deriv[index]
        Ls[i] = leg_sin[index]

    e_th = (Ld*Q2-M*Q1*Ls)
    e_ph = (M*Q2*Ls-Ld*Q1)*1j

    E_Theta = np.sum(e_th*phi_factor)*CONST.SQRT_FAC
    E_Phi = np.sum(e_ph*phi_factor)*CONST.SQRT_FAC

    return E_Theta, E_Phi
