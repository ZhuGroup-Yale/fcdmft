import numpy as np

from fcdmft.ac.interface import AC_Method
from pyscf.lib import logger
from scipy.optimize import least_squares

class TwoPoleAC(AC_Method):
    """Two-pole analytic continuation method."""

    _data_keys = {'coeff', 'omega', 'orbs', 'nocc'}
    _method = 'twopole'

    def __init__(self, orbs, nocc, **options):
        """Constructor for TwoPoleAC.

        Parameters
        ----------
        orbs : list[int]
            indices of active orbitals
        nocc : int
            number of occupied orbitals
        """
        super().__init__(**options)
        self.coeff = None
        self.omega = None
        self.orbs = orbs
        self.nocc = nocc

    def ac_fit(self, data, omega, axis=-1):
        """Compute two-pole AC coefficients for the given data and omega.

        Parameters
        ----------
        data : np.ndarray
            Data to be fit, e.g. self energy
        omega : np.ndarray[np.double]
            1D imaginary frequency grid
        axis : int, optional
            Indicates which axis of the data array corresponds to the frequency axis, by default -1.
            Example: data.shape is (nmo, nmo, nw), call ac_fit(data, omega, axis=-1)
                     data.shape is (nw, nmo, nmo), call ac_fit(data, omega, axis=0)
        """
        self.omega = np.asarray(omega).copy()
        nw = self.omega.size
        data = np.asarray(data)

        assert omega.ndim == 1
        assert data.shape[axis] == nw

        # Move the frequency axis to the last axis
        data_transpose = np.moveaxis(data, axis, -1)

        # Shape of the data array, excluding the frequency axis
        self.shape = data_transpose.shape[:-1]

        coeff = np.zeros((10, *self.shape))

        for idx in np.ndindex(self.shape):
            # randomly generated initial guess
            p = self.orbs[idx[0]] # orbital index
            if p < self.nocc:
                x0 = np.array([0, 1, 1, 1, -1, 0, 0, 0, -1.0, -0.5])
            else:
                x0 = np.array([0, 1, 1, 1, -1, 0, 0, 0, 1.0, 0.5])
            # TODO: analytic gradient
            xopt = least_squares(
                two_pole_fit,
                x0,
                jac='3-point',
                method='trf',
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=1000,
                verbose=0,
                args=(omega, data_transpose[idx]),
            )
            if not xopt.success:
                log = logger.Logger()
                log.warn('2P-Fit Orb %d not converged, cost function %e' % (p, xopt.cost))
            coeff[(slice(None, None, None), *idx)] = xopt.x.copy()
        self.coeff = coeff

    def ac_eval(self, freqs, axis=-1):
        """Evaluate two-pole AC at arbitrary complex frequencies.

        Parameters
        ----------
        freqs : np.ndarray[np.complex128]
            1D array of complex frequencies
        axis : int, optional
            Indicates which axis of the output array should correspond to the frequency axis, by default -1.
            Example: if you want (nmo, nmo, nfreq), call ac_eval(freqs, axis=-1)
                     if you want (nfreq, nmo, nmo), call ac_eval(data, freqs, axis=0)

        Returns
        -------
        np.ndarray
            Pade-Thiele AC evaluated at `freqs`
        """
        if self.coeff is None or self.shape is None:
            raise ValueError('two-pole coefficients not set. Call ac_fit first.')
        is_scalar = np.isscalar(freqs)
        freqs = np.asarray(freqs)

        if self.coeff is None:
            raise ValueError("Pade coefficients not set. Call ac_fit first.")
        freqs = np.asarray(freqs)

        # Handle scalar freqs
        if np.ndim(freqs) == 0:
            freqs = np.array([freqs])

        assert freqs.ndim == 1

        if freqs.dtype != np.complex128:
            freqs = freqs.astype(np.complex128)
        cf = self.coeff[:5] + 1j * self.coeff[5:]

        freqs_broadcasted = np.expand_dims(freqs, axis=tuple(range(1, len(self.shape) + 1)))
        acvals = cf[0] + cf[1] / (freqs_broadcasted + cf[3]) + cf[2] / (freqs_broadcasted + cf[4])
        if not is_scalar:
            return np.moveaxis(acvals, 0, axis)
        else:
            return acvals[0]

    def __getitem__(self, multi_idx):
        assert self.coeff is not None
        new_obj = self.__class__(self.orbs, self.nocc)
        new_obj.coeff = self.coeff[(slice(None, None, None), *np.index_exp[multi_idx])]
        new_obj.omega = self.omega
        new_obj.shape = new_obj.coeff.shape[1:]
        return new_obj

    def diagonal(self, axis1=0, axis2=1):
        if self.coeff is None:
            raise ValueError("Two pole coefficients not set. Call ac_fit first.")
        if len(self.shape) < 2:
            raise ValueError("Two pole coefficients are not >=2D. Cannot get diagonal.")

        new_obj = self.__class__(self.orbs, self.nocc)
        new_obj.coeff = np.diagonal(self.coeff, axis1=axis1+1, axis2=axis2+1)
        new_obj.omega = self.omega
        new_obj.shape = new_obj.coeff.shape[1:]
        return new_obj

def two_pole_fit(coeff, omega, sigma):
    cf = coeff[:5] + 1j * coeff[5:]
    f = cf[0] + cf[1] / (omega + cf[3]) + cf[2] / (omega + cf[4]) - sigma
    f[0] = f[0] / 0.01
    return np.array([f.real, f.imag]).reshape(-1)


def two_pole(freqs, coeff):
    cf = coeff[:5] + 1j * coeff[5:]
    return cf[0] + cf[1] / (freqs + cf[3]) + cf[2] / (freqs + cf[4])


# For compatibility with old code:

def AC_twopole_diag(sigma, omega, orbs, nocc):
    """
    Analytic continuation to real axis using a two-pole model

    Args:
        sigma : 2D array (norbs, nomega)
        omega : 1D array (nomega)
        orbs : list
        nocc : integer

    Returns:
        coeff: 2D array (ncoeff, norbs)
    """
    acobj = TwoPoleAC(orbs, nocc)
    acobj.ac_fit(sigma, omega, axis=-1)
    return acobj.coeff


def AC_twopole_full(sigma, omega, orbs, nocc):
    """
    Analytic continuation to real axis using a two-pole model

    Args:
        sigma : 3D array (norbs, norbs, nomega)
        omega : 1D array (nomega)
        orbs : list
        nocc : integer

    Returns:
        coeff: 3D array (ncoeff, norbs, norbs)
    """
    acobj = TwoPoleAC(orbs, nocc)
    acobj.ac_fit(sigma, omega, axis=-1)
    return acobj.coeff

