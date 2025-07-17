import numpy as np


def delta_lorentzian(x, xc, eta):

    return (eta / np.pi) / ( (x-xc) ** 2 + eta ** 2)

#def delta_gaussian(x, xc, eta):
#
#    return 1./(eta*(2*np.pi)**0.5) * np.exp(-0.5*((x-xc)/eta)**2)

def delta_gaussian(x, xc, eta):

    return 1./(eta*(np.pi)**0.5) * np.exp(-((x-xc)/eta)**2)


def fermi_function(beta, omega, mu):

    return  1./(np.exp(beta*(omega-mu)) + 1)


def fermi_function_bar(beta, omega):
    """
    1- fermi_function

    """

    return np.exp(beta*omega)*fermi_function(beta, omega)

def epsilon(i,j,k):
    """
    anti-symmetric tensor epsilon_ijk

    """
    if i == j or j == k or i == k:
      return 0

def fourier_transform_30(f, t, omega):

    """
    Use numpy.fft to compute the fourier transform of f(t)
    with zero padding.
    Returns: f(omega), omega
    """

    domega = omega[1] - omega[0]
    tmax = (2 * np.pi) / domega

    nt = len(t)
    dt = t[1] - t[0]
    t0 = t[0]

    nt = int((tmax - t[0]) / dt)

    omega = np.fft.fftfreq(nt, dt) * (2 * np.pi)
    f_omega = np.fft.fft(f, n=nt) * dt

    # Sort in order of ascending frequencies
    omega = - omega  # Sign convention
    ii = np.argsort(omega)
    omega = omega[ii]
    f_omega = f_omega[ii]

    # Apply time shift
    f_omega *= np.exp(1j * omega * t0)

    return f_omega, omega

def hilbert_transform(Im, omega_in, omega_out, eta=0.01):
    """
    Compute the Hilbert transform of a causal, frequency-dependent quantity as

        Re(omega_out) =
            (1/pi) \int d_omega_in Im(x) / (omega_out - omega_in + i eta)

    Arguments
    ---------

    Im[N]:
        The imaginary part of the object.
    omega_in[N]:
        The frequencies at which the self-energy is provided.
        Must be a linearly spaced grid.
    omega_out[M]:
        The frequencies at which to compute the Hilbert transform.

    Returns
    -------

    Re[M]:
        The real part of the object.

    """
    Re = np.zeros(omega_out.shape, dtype=np.float64)

    di = omega_in[1] - omega_in[0]
    oo = omega_out
    eta = np.real(eta) * 1j

    for im, oi in zip(Im, omega_in):

        Re += np.real( im / (oo - oi + eta) ) * di / np.pi

    return Re

#def fourier_interpolation():
#
#    """
#    Given quantity on an UPM grid, interpolate it onto arbitrary point
#
#    """
#
#    return



