'''
Density fitting
===============

This module provides the fundamental functions to handle the 3-index tensors
(including the 3-center 2-electron AO and MO integrals, the Cholesky
decomposed integrals) required by the density fitting method or the RI
(resolution of identity) approximation.

Simple usage::

    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz')
    >>> mf = dft.RKS(mol).density_fit().run()
'''

from . import incore
from . import outcore
from . import direct
from . import addons
from .addons import load, aug_etb, DEFAULT_AUXBASIS, make_auxbasis, make_auxmol
from .df import DF, GDF, DF4C, GDF4C


def density_fit(obj, *args, **kwargs):
    '''Given SCF/MCSCF or post-HF object, use density fitting technique to
    approximate the 2e integrals.'''
    return obj.density_fit(*args, **kwargs)
