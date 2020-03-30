import numpy as np
from input_data import CONST

#For more details regarding computation of fully normalized Associated Legendre polynomials and 
#Spherical harmonics, refer to the article below:
#Limpanuparb, T. and Milthorpe, J., 2014. 
#Associated Legendre Polynomials and Spherical Harmonics Computation for Chemistry Applications. 
#arXiv preprint arXiv:1410.1748.
def computeP(L, A, B, x, y):
    '''

    Parameters
    ----------
    L : integer
        highest order of associated legendre polynomials.
    A : float64 (array)
        used in recurrence relations of legendre polynomials.
    B : float64 (array)
        used in recurrence relations of legendre polynomials.
    x : float64 (scalar)
        cosine of theta angle.
    y : float64 (scalar)
        sin of theta angle.

    Returns
    -------
    P : normalized associated polynomials of size [L + 1, L + 1]
        the values of normalized associated legendre of 
        degree lmax = L and order |mmax| = L for a particular angle theta.

    '''
    P = np.zeros((L + 1, L + 1), dtype=np.float64)

    temp = 0.7071067811865475244008443621 #1/sqrt(2)
    P[0, 0] = temp
    if(L>0):
        SQRT3DIV2 = 1.224744871391589049098642037353 #sqrt(3)/sqrt(2)
        P[1, 0] = x*SQRT3DIV2
        temp = -SQRT3DIV2*y*temp
        P[1, 1]=temp
        for l in range(2, L+1):
            for m in range(0, l-1):
                P[l, m] = A[l, m] * (x * P[l-1, m] + B[l, m] * P[l-2, m])
            P[l, l-1] = x * np.sqrt(2.0 * (l-1) + 3.0) * temp
            temp = -np.sqrt(1.0 + 0.5 / l) * y * temp
            P[l, l]=temp
    return P

#For more information regarding P/sin term, read
#Li, P. and Jiang, L.J., 2012. 
#The far field transformation for the antenna modeling based on spherical electric field measurements. 
#Progress In Electromagnetics Research, 123, pp.243-261.
def LegendreP(theta, N_max, source_theta=None):
    '''

    Parameters
    ----------
    theta : float64 (array)
        array of theta angles (radians) in spherical coordinate system.
    N_max : integer
        maximum order of associated legendre polynomials and derivatives.
    source_theta : float64, optional
        a single theta angle (radian) for which associated
        legendre polynomials should be computed. the default is none.

    Returns
    -------
    LegendreD[0 : N_max + 2, 0 : N_max + 2, 0 : len(theta) + 1] : float64 (array)
        derivatives of normalized associated legendre polynomials for
        all degrees and orders up to N_max (including) computed for all angles of theta.
    LegendreS[0 : N_max + 2, 0 : N_max + 2, 0 : len(theta) + 1] : float64 (array) 
        legendre / sin(theta) term for all degrees and orders up to
        N_max (including) computed for all angles of theta.
    x[0 : len(theta) + 1] : float64 (array)
        cos(theta) computed for all angles of theta.
    y[0 : len(theta) + 1] : float64 (array)
        sin(theta) computed for all angles of theta.
    LegendreD1[0 : N_max + 2, 0 : N_max + 2] : float64 (array)
        derivatives of normalized associated legendre polynomials for all
        degrees and orders up to N_max (including) computed for a particular angle source_theta.
    LegendreS1[0 : N_max + 2, 0 : N_max + 2] : float64 (array)
        legendre / sin(theta) term for all degrees and orders 
        up to N_max (including) computed for a particular angle source_theta.

    '''
    NPTS = len(theta)
    LL = N_max + 1 

    x = np.cos(theta)
    y = np.sin(theta)

    #Temporary arrays used in recursion
    A = np.zeros((LL + 1, LL + 1), dtype=np.float64) 
    B = np.zeros((LL + 1, LL + 1), dtype=np.float64)

    #Compute temporary arrays 
    for l in range(2, LL+1):
        ls = l**2
        lm1s = (l-1)**2
        for m in range(0, l-1):
            ms = m**2
            A[l, m] = np.sqrt((4.0 * ls - 1.0) / (ls - ms))
            B[l, m] = -np.sqrt((lm1s - ms) / (4.0 * lm1s - 1.0))

    #Allocate space for Legendre polynomials
    LegendreP = np.zeros((LL + 1, LL + 1, NPTS), dtype=np.float64) # [size=(L,M),Theta]

    #Compute all Associated legendre functions on x-th grid
    for i in range(0, NPTS):
        LegendreP[:, :, i] = computeP(LL, A, B, x[i], y[i])

    #Compute P/sin functions, general case:
    LegendreS = np.zeros((LL + 1, LL + 1, NPTS), np.float64)

    #consider asymptotic cases when theta = 0 or theta = pi
    eps = 1e-14
    mask1 = (abs(theta - 0.0) < eps)
    mask2 = (abs(theta - np.pi) < eps)

    #combine masks
    mask = np.ma.mask_or(mask1, mask2)

    #find location of asymptotes
    asymptote = np.where(mask)[0]

    #find location of regular points 
    regular = np.where(~mask)[0]

    #evaluate function at regular points
    LegendreS[:, :, regular] = LegendreP[:, :, regular] / y[regular]

    #Consider special case for m=1 at theta=0:
    m = 1
    for n in range(1, LL + 1):
        LegendreS[n, m ,asymptote[0]] = -0.5 * np.sqrt(n * (2.0 * n + 1.0) * (n + 1.0) / 2.0)

    if(np.shape(asymptote)==(2,)): #include theta == pi case
        for n in range(1, LL + 1):
            LegendreS[n, m, asymptote[1]] = (-1) ** (n + 1) * LegendreS[n, m, asymptote[0]]

    #Evaluate derivatives of normalized Associated Legendre polynomials using recurrence relation 
    LegendreD = np.zeros((LL, LL, NPTS), np.float64)

    #Leg_deriv = 0 when m = 0, n = 0
    for n in range(1, LL):
        for m in range(0, n + 1): #include the case when n = m
            LegendreD[n, m, :] = -(n + 1.0) * x[:] * LegendreS[n, m, :] + \
                                 np.sqrt((2.0 * n + 1.0) * (n + m + 1.0) * \
                                 (n - m + 1.0) / (2.0 * n + 3.0)) * LegendreS[n + 1, m, :]

    #Compute for a single pointing if present
    if(source_theta is not None):

        LegendreD1 = np.zeros((LL, LL), dtype=np.float64)
        LegendreS1 = np.zeros((LL + 1, LL + 1), dtype=np.float64)

        #check if source_theta is a part of existing array:
        if source_theta in theta:

            #find the position of theta corresponding to source_theta (assuming all values of theta are unique)
            pos = np.where(abs(theta - source_theta) < eps)

            LegendreD1[:, :] = np.squeeze(LegendreD[:, :, pos])
            LegendreS1[:, :] = np.squeeze(LegendreS[:, :, pos])

        else: #compute legendre polynomial and sin term for this unique position

            LegendreD1 = np.zeros((LL, LL), np.float64)

            xs = np.cos(source_theta)
            ys = np.sin(source_theta)

            LegendreP1 = computeP(LL, A, B, xs, ys)
            LegendreS1 = LegendreP1 / ys

            for n in range(1, LL):
                for m in range(0, n + 1):
                    LegendreD1[n, m] = -(n + 1) * xs * LegendreS1[n, m] + \
                                       np.sqrt((2.0 * n + 1.0) * (n + m + 1.0) * \
                                       (n - m + 1.0) / (2.0 * n + 3.0)) * LegendreS1[n + 1, m]

        return LegendreD, LegendreS[0 : LL + 1, 0 : LL + 1], x, y, LegendreD1, LegendreS1

    else:

        return LegendreD, LegendreS[0 : LL + 1, 0 : LL + 1], x, y

#spherical modes used to re-construct electric fields
def extract_modes(h5f, Vcplx, freq):
    '''

    Parameters
    ----------
    h5f : dictionary of float and integer arrays
        file containing spherical modal coefficients and indexes of spherical
        vector harmonics for a particular polarisation and frequency
    Vcplx : complex128 (array)
        array of complex beamformer coefficients corresponding to a particular
        grid point specified in the input_data.py.
    freq : float64 (singe number)
        a single frequency point given in Hz.

    Returns
    -------
    beam_modes : dictionary of complex128 and integer arrays
        contains accumulated spherical modal coefficients scaled by
        the values of complex beamformer coefficients.
    N, M - integer arrays
        combination of indexes required for reconstruction of spherical vector
        harmonics
    '''

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

def construct_FF(phi, theta, leg_deriv, leg_sin, beam_modes):
    '''

    Parameters
    ----------
    phi : float64 (array)
        array of phi angles in spherical coordinate system.
    theta : float64 (array)
        array of theta angles in spherical coordinate system.
    leg_deriv : float64 (array)
        derivatives of normalised associated legendre polynomials.
    leg_sin : float64 (array)
        normalised associated legendre divided by sin(theta).
    beam_modes : dictionary of complex128 and integer arrays
        dictionary containing spherical modal coefficients scaled by
        complex beamformer weights and indexes of spherical harmonics.

    Returns
    -------
    E_Theta : float64 (array)
        array of electric fields as a function of phi and theta in theta hat 
        direction.
    E_Phi : float64 (array)
        array of electric fields as a function of phi and theta in phi hat 
        direction.

    '''

    Theta_max = np.size(theta)

    target = beam_modes
    dim = len(target['N']) #dimension of summation

    Q1 = target['Q1'] #extract i-th Q1 modal coefficient
    Q2 = target['Q2'] #extract i-th Q2 modal coefficient
    M = target['M'].astype(np.int64)
    N = target['N'].astype(np.int64)

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
        ni = N[i]
        mi = Ma[i]
        Ld[:, i]  = leg_deriv[ni, mi, :]
        Ls[:, i] = leg_sin[ni, mi, :]

    e_th = (Ld*Q2-M*Q1*Ls)
    e_ph = (M*Q2*Ls-Ld*Q1)*1j

    E_Theta = np.inner(phi_factor, e_th) * CONST.SQRT_FAC
    E_Phi = np.inner(phi_factor, e_ph) * CONST.SQRT_FAC

    return E_Theta, E_Phi

# ================================================================================================================
   
def construct_FF1(phi1, theta1, leg_deriv, leg_sin, beam_modes):
    '''

    Parameters
    ----------
    phi1 : float64
        a single phi pointing angle in spherical coordinate system.
    theta1 : float64
        a single theta pointing angle in spherical coordinate system.
    leg_deriv : float64 (array)
        derivatives of normalised associated legendre polynomials.
    leg_sin : float64 (array)
        normalised associated legendre divided by sin(theta).
    beam_modes : dictionary of complex128 and integer arrays
        dictionary containing spherical modal coefficients scaled by
        complex beamformer weights and indexes of spherical harmonics.

    Returns
    -------
    E_Theta : float64
        value of the electric field corresponding 
        to (phi1, theta1) in theta hat direction.
    E_Phi : float64
        value of the electric field corresponding 
        to (phi1, theta1) in phi hat direction.

    '''

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
        ni = N[i]
        mi = Ma[i]
        Ld[i]  = leg_deriv[ni, mi]
        Ls[i] = leg_sin[ni, mi]

    e_th = (Ld*Q2-M*Q1*Ls)
    e_ph = (M*Q2*Ls-Ld*Q1)*1j

    E_Theta = np.sum(e_th*phi_factor)*CONST.SQRT_FAC
    E_Phi = np.sum(e_ph*phi_factor)*CONST.SQRT_FAC

    return E_Theta, E_Phi
