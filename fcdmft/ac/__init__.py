from fcdmft.ac.grids import _get_scaled_legendre_roots, _get_clenshaw_curtis_roots
from fcdmft.ac.interface import AC_Method
from fcdmft.ac.two_pole import TwoPoleAC
from fcdmft.ac.pade import PadeAC

from pyscf.lib import chkfile

def load_ac(chkfilename: str, dataname: str = 'ac') -> AC_Method:
    """Load an AC object from an HDF5 file.

    Parameters
    ----------
    chkfilename : str
        Path to the HDF5 file
    dataname : str, optional
        Name of the dataset in the HDF5 file, by default 'ac'

    Returns
    -------
    AC_Method
        The loaded AC object
    """
    data_dic = chkfile.load(chkfilename, dataname)
    method = data_dic.pop('method')
    ac_class = {'twopole': TwoPoleAC, 'pade': PadeAC}[method.decode()]

    # Instantiate the AC object
    acobj = ac_class.__new__(ac_class)
    # Update the AC object with the data from the HDF5 file
    acobj.__dict__.update(data_dic)
    acobj.shape = acobj.coeff.shape[1:]
    return acobj
