from typing import Tuple, Union
import numpy as np
from abc import ABC, abstractmethod
import warnings

from pyscf.lib import chkfile

class AC_Method(ABC):
    """
    Base class for AC methods

    Attributes
    ----------
    shape : tuple
        Shape of the data array that was passed to ac_fit, excluding the frequency axis.
    """

    _data_keys : set
    _method : str

    def __init__(self, *args, **options):
        if args:
            warnings.warn(f"{len(args)} unused positional arguments passed to the AC_method constructor.")
        if options:
            warnings.warn(f"Unused keyword arguments passed to the AC_method constructor: {options}")
        self.shape = ()

    @abstractmethod
    def ac_fit(self, data: np.ndarray, omega: np.ndarray, axis: int = -1):
        """The kernel of the AC method. This function should be implemented in the derived class.

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
        raise NotImplementedError

    @abstractmethod
    def ac_eval(self, freqs: np.ndarray, axis: int = -1):
        """After you call ac_fit, you can call this function to evaluate the AC at
           arbitrary complex frequencies.

        Parameters
        ----------
        freqs : np.ndarray[np.complex128] | complex
            1D array of complex frequencies, or a single complex frequency
        axis : int, optional
            Indicates which axis of the output array should correspond to the frequency axis, by default -1.
            Example: if you want (nmo, nmo, nfreq), call ac_eval(freqs, axis=-1)
                     if you want (nfreq, nmo, nmo), call ac_eval(data, freqs, axis=0)
            If freqs is a scalar, the output shape will be the same as the input data shape.

        Returns
        -------
        np.ndarray
            Interpolated/approximated values at the new complex frequencies
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, multiidx: Union[Tuple[int], slice]):
        """
        Get a slice of the AC method. This function should be implemented in the derived class.
        Defining this method lets you do things like acobj[p,q] or acobj[p,:].

        Parameters
        ----------
        multiidx : Tuple[int] | slice
            Index or slice object for the AC method.

        Returns
        -------
        object
            New instance of the AC method with the selected slice.
        """
        raise NotImplementedError

    @abstractmethod
    def diagonal(self, axis1=0, axis2=1):
        """Create a new instance of the AC method with only the diagonal elements
           of the self.coeff tensor. Convenient for getting diagonal of self-energy
           after calculating the full self-energy matrix.

           This behaves more or less the same as np.diagonal.

        Parameters
        ----------
        axis1 : int, optional
            First axis, by default 0
        axis2 : int, optional
            Second axis, by default 1

        Returns
        -------
        object
            New instance of the AC method with only the diagonal elements.
        """
        raise NotImplementedError

    def save(self, chkfilename: str, dataname: str = 'ac'):
        """Save the AC object and coefficients to an HDF5 file.

        Parameters
        ----------

        """
        data_dic = { key : getattr(self, key) for key in self._data_keys }
        data_dic['method'] = self._method
        chkfile.dump(chkfilename, dataname, data_dic)
