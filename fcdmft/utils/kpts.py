import numpy as np


def get_kconserv_ria_efficient(cell, kpts, tol=1e-12):
    r"""Get the momentum conservation array for single excitation amplitudes
    for a set of k-points with appropriate k-shift.


    Given k-point indices (kshift, m) the array kconserv[kshift,m] returns
    the index n that satisfies momentum conservation,

        (k(m) - k(n) - k(kshift)) \dot a = 2n\pi
    """
    nkpts = kpts.shape[0]
    a = cell.lattice_vectors() / (2 * np.pi)

    kconserv = np.zeros((nkpts, nkpts), dtype=int)
    kvKM = -kpts[:, None, :] + kpts[:, :]
    for N, kvN in enumerate(kpts):
        kvKMN = np.einsum('wx,kmx->wkm', a, kvKM - kvN, optimize=True)
        # check whether (1/(2pi) k_{KLN} dot a) is an integer
        kvKMN_int = np.rint(kvKMN)
        mask = np.einsum('wkm->km', abs(kvKMN - kvKMN_int), optimize=True) < tol
        kconserv[mask] = N
    return kconserv
