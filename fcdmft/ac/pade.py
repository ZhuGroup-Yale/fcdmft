import numpy as np
import warnings

from fcdmft.ac.interface import AC_Method


def _get_ac_idx(nw, npts=18, step_ratio=2.0 / 3.0, idx_start=1):
    """Get an array of indices, with stepsize decreasing.

    Parameters
    ----------
    nw : int
        number of frequency points
    npts : int, optional
        final number of selected points, by default 18
    step_ratio : float, optional
        final stepsize / initial stepsize, by default 2.0/3.0
    idx_start : int, optional
        first index of final array, by default 1

    Returns
    -------
    np.ndarray
        an array for indexing frequency and omega.
    """
    if nw <= npts:
        raise ValueError('nw (%s) should be larger than npts (%s)' % (nw, npts))
    steps = np.linspace(1.0, step_ratio, npts)
    steps /= np.sum(steps)
    steps = np.cumsum(steps * nw)
    steps += idx_start - steps[0]
    steps = np.round(steps).astype(int)
    return steps


def pade_thiele_ndarray(freqs, zn, coeff):
    """NDarray-friendly analytic continuation using Pade-Thiele method.

    Parameters
    ----------
    freqs : np.ndarray, shape (nfreqs,), complex
        Points in the complex plane at which to evaluate the analytic continuation.
    zn : np.ndarray, shape (ncoeff,), complex
        interpolation points
    coeff : np.ndarray, shape (ncoeff, M1, M2, ...,), complex
        Pade-Thiele coefficients

    Returns
    -------
    np.ndarray, shape (nfreqs, M1, M2, ...,), complex
        Pade-Thiele analytic continuation evaluated at `freqs`
    """
    ncoeff = len(coeff)

    if freqs.ndim != 1 or zn.ndim != 1:
        raise ValueError('freqs and zn must be 1D arrays')
    if ncoeff != len(zn):
        raise ValueError('coeff and zn must have the same length')
    freqs_broadcasted = np.expand_dims(freqs, tuple(range(1, coeff.ndim)))

    X = coeff[-1] * (freqs_broadcasted - zn[-2])
    # X has shape (nfreqs, M1, M2, ...)

    for i in range(ncoeff - 1):
        idx = ncoeff - i - 1
        X = coeff[idx] * (freqs_broadcasted - zn[idx - 1]) / (1.0 + X)
    X = coeff[0] / (1.0 + X)
    return X


def thiele_ndarray(fn, zn):
    """Iterative Thiele algorithm to compute coefficients of Pade approximant

    Parameters
    ----------
    fn : np.ndarray, shape (nw, N1, N2, ...,), complex
        Function values at the points zn
    zn : np.ndarray, shape(nw,), complex
        Points in the complex plane used to compute fn

    Returns
    -------
    np.ndarray, shape(nw, N1, N2, ...), complex
        Coefficients of Pade approximant
    """
    nw = len(zn)
    # No need to allocate coeffs since g = coeffs at the end.
    g = fn.copy()
    # g has shape (nw, N1, N2, ...)

    zn_broadcasted = np.expand_dims(zn, tuple(range(1, g.ndim)))
    # zn_broadcasted has shape (nw, 1, 1, ..., 1)

    for i in range(1, nw):
        # At this stage, coeffs[i-1] is already computed.
        # coeffs[i-1] = g[i-1]

        g[i:] = (g[i - 1] - g[i:]) / ((zn_broadcasted[i:] - zn_broadcasted[i - 1]) * g[i:])
    return g


class PadeAC(AC_Method):
    """
    Analytic continuation to real axis using a Pade approximation
    from Thiele's reciprocal difference method
    Reference: J. Low Temp. Phys. 29, 179 (1977)
    """

    _data_keys = {'npts', 'step_ratio', 'coeff', 'omega', 'idx'}
    _method = 'pade'

    def __init__(self, *args, npts=18, step_ratio=2.0 / 3.0, **options):
        """Constructor for PadeAC class

        Parameters
        ----------
        npts : int, optional
            Number of frequency points to use for AC, by default 18
        step_ratio : float, optional
            Frequency grid step ratio, by default 2.0/3.0
        """
        super().__init__(*args, **options)
        self.step_ratio = step_ratio
        if npts % 2 != 0:
            warnings.warn(f'Pade AC: npts should be even, but {npts} was given. Using {npts-1} instead.')
            npts -= 1
            assert npts > 0
        self.npts = npts
        self.coeff = None  # Pade fitting coefficient
        self.omega = None  # input frequency grids
        self.idx = None  # idx of frequency grids used for fitting

    @property
    def omega_fit(self):
        """Return the frequency grid used for fitting."""
        if self.omega is None or self.idx is None:
            return None
        return self.omega[self.idx]

    def ac_fit(self, data, omega, axis=-1):
        """Compute Pade-Thiele AC coefficients for the given data and omega.

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
        
        if self.idx is None:
            self.idx = _get_ac_idx(nw, npts=self.npts, step_ratio=self.step_ratio)

        sub_omega = self.omega[self.idx]
        sub_data = np.moveaxis(data, axis, 0)[self.idx]
        self.coeff = thiele_ndarray(sub_data, sub_omega)
        self.shape = self.coeff.shape[1:]

    def ac_eval(self, freqs, axis=-1):
        """Evaluate Pade AC at arbitrary complex frequencies.

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
        if self.coeff is None or self.omega is None:
            raise ValueError('Pade coefficients not set. Call ac_fit first.')
        is_scalar = np.isscalar(freqs)
        freqs = np.asarray(freqs)

        # Handle scalar freqs
        if np.ndim(freqs) == 0:
            freqs = np.array([freqs])

        assert freqs.ndim == 1
        if freqs.dtype != np.complex128:
            freqs = freqs.astype(np.complex128)
        X = pade_thiele_ndarray(freqs, self.omega[self.idx], self.coeff)
        if not is_scalar:
            return np.moveaxis(X, 0, axis)
        else:
            return X[0]

    def __getitem__(self, multi_idx):
        assert self.coeff is not None
        new_obj = self.__class__(npts=self.npts, step_ratio=self.step_ratio)
        new_obj.coeff = self.coeff[(slice(None, None, None), *np.index_exp[multi_idx])]
        new_obj.omega = self.omega
        new_obj.idx = self.idx
        new_obj.shape = new_obj.coeff.shape[1:]

        return new_obj

    def diagonal(self, axis1=0, axis2=1):
        assert self.coeff is not None
        assert len(self.shape) >= 2
        new_obj = self.__class__(npts=self.npts, step_ratio=self.step_ratio)
        new_obj.coeff = np.diagonal(self.coeff, axis1=axis1 + 1, axis2=axis2 + 1)
        new_obj.omega = self.omega
        new_obj.idx = self.idx
        new_obj.shape = new_obj.coeff.shape[1:]
        return new_obj


def AC_pade_thiele_diag(sigma, omega, npts=18, step_ratio=2.0 / 3.0):
    """Pade fitting for diagonal elements for a matrix.

    Parameters
    ----------
    sigma : complex ndarray
        matrix to fit, (norbs, nomega)
    omega : complex array
        frequency of the matrix sigma (nomega)
    npts : int, optional
        number of selected points, by default 18
    step_ratio : _type_, optional
        step ratio to select points, by default 2.0/3.0

    Returns
    -------
    acobj.coeff : complex ndarray
        fitting coefficient
    acobj.omega[acobj.idx] : complex ndarray
        selected frequency points for fitting
    """
    acobj = PadeAC(npts=npts, step_ratio=step_ratio)
    acobj.ac_fit(sigma, omega)
    return acobj.coeff, acobj.omega[acobj.idx]


def AC_pade_thiele_full(sigma, omega, npts=18, step_ratio=2.0 / 3.0):
    """Pade fitting for full matrix.

    Parameters
    ----------
    sigma : complex ndarray
        matrix to fit, (norbs, nomega)
    omega : complex array
        frequency of the matrix sigma (nomega)
    npts : int, optional
        number of selected points, by default 18
    step_ratio : _type_, optional
        step ratio to select points, by default 2.0/3.0

    Returns
    -------
    acobj.coeff : complex ndarray
        fitting coefficient
    acobj.omega[acobj.idx] : complex ndarray
        selected frequency points for fitting
    """
    acobj = PadeAC(npts=npts, step_ratio=step_ratio)
    acobj.ac_fit(sigma, omega)
    return acobj.coeff, acobj.omega[acobj.idx]


# Old functions


def thiele(fn, zn):
    nfit = len(zn)
    g = np.zeros((nfit, nfit), dtype=np.complex128)
    g[:, 0] = fn.copy()
    for i in range(1, nfit):
        g[i:, i] = (g[i - 1, i - 1] - g[i:, i - 1]) / ((zn[i:] - zn[i - 1]) * g[i:, i - 1])
    a = g.diagonal()
    return a


def pade_thiele(freqs, zn, coeff):
    nfit = len(coeff)
    X = coeff[-1] * (freqs - zn[-2])
    for i in range(nfit - 1):
        idx = nfit - i - 1
        X = coeff[idx] * (freqs - zn[idx - 1]) / (1.0 + X)
    X = coeff[0] / (1.0 + X)
    return X


if __name__ == '__main__':
    # Test PadeAC
    import numpy as np
    from fcdmft.ac.grids import _get_scaled_legendre_roots

    # Generate some test data
    npts = 18
    step_ratio = 2.0 / 3.0
    omega, wts = _get_scaled_legendre_roots(npts, x0=0.5)
    vals = np.random.random((npts, 100, 100)) + 1j * np.random.random((npts, 100, 100))

    coeffs_good = np.zeros((npts, 100, 100), dtype=np.complex128)
    for i in range(100):
        for j in range(100):
            coeffs_good[:, i, j] = thiele(vals[:, i, j], omega)
    coeffs2 = thiele_ndarray(vals, omega)
    assert np.allclose(coeffs_good, coeffs2)

    freqs = np.linspace(0.0, 1, 100)
    ac_good = np.zeros((100, 100, 100), dtype=np.complex128)
    for i in range(100):
        for j in range(100):
            ac_good[:, i, j] = pade_thiele(freqs, omega, coeffs_good[:, i, j])
    ac2 = pade_thiele_ndarray(freqs, omega, coeffs_good)
    assert np.allclose(ac_good, ac2)
