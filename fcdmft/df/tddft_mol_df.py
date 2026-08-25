"""
Time-Dependent Density Functional Theory (tddft) for excitation energy.
Both restricted and unrestricted cases are supported.
tddft can be solved with (energy-specific) Davidson algorithm, Lanczos algorithms or fully diagonalization.

References, adapted from the BSE code from:
    Hillenbrand, Christopher, Jiachen Li, and Tianyu Zhu. arXiv:2410.24168 (2024).
    J. Comput. Chem. 38, 383 (2017).
    Ghosh, S. K and  Chattaraj, P. K. (Eds.). (2013).
    SIAM J. Matrix Anal. Appl. 39, 683 (2018).
"""

import time
import os

import numpy as np
import scipy
import scipy.linalg as sla
import h5py

from pyscf import lib
from pyscf.data import nist
from pyscf.tools import mo_mapping

HARTREE2EV = nist.HARTREE2EV

einsum = lib.einsum

def tddft_full_diagonalization(multi, nocc, mo_energy, Lpq, TDA=False):
    """Full diagonalization of tddft equation.
    tddft equation is defined as equation 1 in doi.org/10.1002/jcc.24688.
    Spin-adapted formalism can be found in chapter 18.3.2 in "Concepts and methods in modern theoretical chemistry.
    Electronic structure (2013, CRC) Ghosh S.K., Chattaraj P.K. (eds.)"
    The working equation is rewritten as equation 15 in doi.org/10.1063/1.477483.

    Args:
        tddft (fcdmft.gw.mol.tddft.tddft): tddft object.
        multi (str): multiplicity, 's'=singlet, 't'=triplet, 'u'=unrestricted.

    Returns:
        exci (double array): excitation energy.
        X_vec (double ndarray): X block of eigenvector (excitation).
        Y_vec (double ndarray): Y block of eigenvector (de-excitation).
    """
    nspin, _, nmo, _ = Lpq.shape

    # determine dimension
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    full_dim = dim[0] + dim[1] if nspin == 2 else dim[0]
    apb = np.zeros(shape=[full_dim, full_dim], dtype=np.double)

    # amb is not allocated if TDA is true, since B=0
    if not TDA:
        amb = np.zeros_like(apb)

    # scale Coulomb matrix
    scale = 4.0 / nspin
    if TDA:
        scale /= 2.0

    # Coulomb part
    if multi == 's' or multi == 'u':
        for i in range(nspin):
            for j in range(nspin):
                apb[i * dim[0] : i * dim[0] + dim[i], j * dim[0] : j * dim[0] + dim[j]] += (
                    einsum('Lia,Ljb->iajb', Lpq[i][:, : nocc[i], nocc[i] :], Lpq[j][:, : nocc[j], nocc[j] :]).reshape(
                        dim[i], dim[j]
                    )
                    * scale
                )

    # TODO : W becomes K contraction
    # K part
    for i in range(nspin):
        KA = -einsum('Lij,Lab->iajb', Lpq[i][:, : nocc[i], : nocc[i]], Lpq[i][:, nocc[i] :, nocc[i] :])
        KA = KA.reshape(nocc[i] * nvir[i], nocc[i] * nvir[i])
        apb[i * dim[0] : i * dim[0] + dim[i], i * dim[0] : i * dim[0] + dim[i]] += KA
        if not TDA:
            amb[i * dim[0] : i * dim[0] + dim[i], i * dim[0] : i * dim[0] + dim[i]] += KA
            KB = -einsum('Lib,Laj->iajb', Lpq[i][:, : nocc[i], nocc[i] :], Lpq[i][:, nocc[i] :, : nocc[i]])
            KB = KB.reshape(nocc[i] * nvir[i], nocc[i] * nvir[i])
            apb[i * dim[0] : i * dim[0] + dim[i], i * dim[0] : i * dim[0] + dim[i]] += KB
            amb[i * dim[0] : i * dim[0] + dim[i], i * dim[0] : i * dim[0] + dim[i]] -= KB

    # orbital energy contribution to A+B and A-B matrix
    orb_diff = []
    for i in range(nspin):
        orb_diff.append(np.asarray(mo_energy[i][None, nocc[i] :] - mo_energy[i][: nocc[i], None]).reshape(-1))
    orb_diff = np.concatenate(orb_diff, axis=0)
    if not TDA:
        np.fill_diagonal(amb, orb_diff + np.diag(amb))
    np.fill_diagonal(apb, orb_diff + np.diag(apb))

    if TDA:
        # Diagonalizing A is numerically more stable than
        # diagonalizing A^2. Solve standard hermitian eigenvalue problem

        # B = 0, so A = apb
        exci, xpy = scipy.linalg.eigh(apb)
        X_vec = xpy.T
        Y_vec = np.zeros_like(xpy)

    else:
        # equation 15 in doi/10.1063/1.477483, solved by LAPACK function dsygvd
        exci_sqr, xpy_w = scipy.linalg.eigh(apb, amb, type=3)
        exci = np.sqrt(exci_sqr)

        # dsygvd normalizes xpy_w such that
        # xpy_w @ xpy_w.T = A - B
        # Using the fact that A - B = (X+Y) @ diag(w) @ (X+Y).T,
        # we calculate X+Y = xpy_w @ diag(1/sqrt(w)).
        xpy = xpy_w / np.sqrt(exci)[None, :]

        # (A+B) |X+Y> = w |X-Y>, so
        # |X-Y> = w^-1 (A+B) |X+Y>
        xmy = (apb @ xpy) / exci[None, :]

        # Rows of X_vec and Y_vec are the eigenvectors, hence the transpose.
        X_vec = (xpy + xmy).T / 2.0
        Y_vec = (xpy - xmy).T / 2.0

    # reshape X and Y eigenvector
    if nspin == 1:
        X_vec = [X_vec.reshape(-1, nocc[0], nvir[0])]
        Y_vec = [Y_vec.reshape(-1, nocc[0], nvir[0])]
    else:
        X_vec_a, X_vec_b, Y_vec_a, Y_vec_b = [], [], [], []
        for r in range(len(exci)):
            X_vec_a.append(X_vec[r][: dim[0]].reshape(nocc[0], nvir[0]))
            X_vec_b.append(X_vec[r][dim[0] :].reshape(nocc[1], nvir[1]))
            Y_vec_a.append(Y_vec[r][: dim[0]].reshape(nocc[0], nvir[0]))
            Y_vec_b.append(Y_vec[r][dim[0] :].reshape(nocc[1], nvir[1]))
        X_vec = [np.asarray(X_vec_a), np.asarray(X_vec_b)]
        Y_vec = [np.asarray(Y_vec_a), np.asarray(Y_vec_b)]

    #tddft.exci = exci
    #tddft.X_vec = X_vec
    #tddft.Y_vec = Y_vec

    return exci, X_vec, Y_vec


def get_excitation_from_eigenvector(multi, nocc, mo_energy, Lpq, X_vec, Y_vec, TDA=False):
    """Get excitation energies from corresponding eigenvectors.
    exci = (X+Y)^T (A+B) (X+Y)
    Equation.18 in doi.org/10.1063/1.477483

    Args:
        multi (str): multiplicity, 's'=singlet, 't'=triplet, 'u'=unrestricted.
        nocc (int array): the number of occupied orbitals.
        mo_energy (double ndarray): orbital energy.
        Lpq (double ndarray): 3-center density-fitting matrix.
        X_vec (list of double ndarray): X block of eigenvector (excitation).
        Y_vec (list of double ndarray): Y block of eigenvector (de-excitation).
        TDA (bool, optional): use TDA approximation. Defaults to False.

    Returns:
        exci (double array): excitation energies.
    """
    nspin, naux, nmo, _ = Lpq.shape

    # determine dimension
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    full_dim = dim[0] + dim[1] if nspin == 2 else dim[0]


    nroot = X_vec[0].shape[0]
    if nspin == 1:
        xpy = (X_vec[0] + Y_vec[0]).reshape(nroot, -1)
    else:
        X_vec_a, X_vec_b = X_vec[0].reshape(nroot, -1), X_vec[1].reshape(nroot, -1)
        Y_vec_a, Y_vec_b = Y_vec[0].reshape(nroot, -1), Y_vec[1].reshape(nroot, -1)
        X_vec_ab = np.concatenate((X_vec_a, X_vec_b), axis=1)
        Y_vec_ab = np.concatenate((Y_vec_a, Y_vec_b), axis=1)
        xpy = X_vec_ab + Y_vec_ab

    # scale Coulomb matrix
    scale = 4.0 / nspin
    if TDA is True:
        scale /= 2.0

    apb_prod = np.zeros(shape=[nroot, full_dim], dtype=np.double)

    # contraction: V
    if multi != 't' and multi != 'T':
        for ivec in range(nroot):
            Lpq_z = np.zeros(shape=[nspin, naux], dtype=np.double)
            for s in range(nspin):
                z = xpy[ivec][s * dim[0] : s * dim[0] + dim[s]].reshape(nocc[s], nvir[s])
                Lpq_z[s] = einsum('Pjb,jb->P', Lpq[s][:, : nocc[s], nocc[s] :], z)

            for s in range(nspin):
                for t in range(nspin):
                    vz = einsum('Pia,P->ia', Lpq[s][:, : nocc[s], nocc[s] :], Lpq_z[t]).reshape(-1) * scale
                    apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += vz

    # TODO: W becomes K contraction
    # contraction: W
    for ivec in range(nroot):
        # this matrix can be large
        Lpq_z = np.zeros(shape=[nspin, naux, nmo, nmo], dtype=np.double)
        for s in range(nspin):
            z = xpy[ivec][s * dim[0] : s * dim[0] + dim[s]].reshape(nocc[s], nvir[s])
            Lpq_z[s][:, :, : nocc[s]] = einsum('Lpb,jb->Lpj', Lpq[s][:, :, nocc[s] :], z)
            waz = -einsum(
                'Lji,Laj->ia', Lpq_bar[s][:, : nocc[s], : nocc[s]], Lpq_z[s][:, nocc[s] :, : nocc[s]]
            ).reshape(-1)
            if TDA is True:
                wbz = np.zeros_like(waz)
            else:
                wbz = -einsum(
                    'Laj,Lij->ia', Lpq_bar[s][:, nocc[s] :, : nocc[s]], Lpq_z[s][:, : nocc[s], : nocc[s]]
                ).reshape(-1)
            apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += waz + wbz

    # contraction: orbital energy difference
    for ivec in range(nroot):
        for s in range(nspin):
            orb_diff = np.asarray(mo_energy[s][None, nocc[s] :] - mo_energy[s][: nocc[s], None]).reshape(-1)
            oz = orb_diff * xpy[ivec][s * dim[0] : s * dim[0] + dim[s]]
            apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += oz

    # As shown in doi.org/10.1063/1.477483
    # equation 18 (A+B)|X+Y>=w|X-Y> and orthogonality in equation 20 <X+Y|X-Y> = delta
    # so <X+Y|(A+B) = <X+Y|w|X-Y> = w
    exci = np.zeros(shape=[nroot], dtype=np.double)
    for i in range(nroot):
        exci[i] = np.matmul(xpy[i].T, apb_prod[i])

    return exci


def davidson_restart(Mp, Mm, tri_vec, nvec_pair_to_save, e_min=0.0):
    """Restart Davidson algorithm.

    Parameters
    ----------
    Mp : ndarray
        The matrix <tri_vec|A+B|tri_vec>
    Mm : ndarray or None
        The matrix <tri_vec|A-B|tri_vec>
    tri_vec : ndarray
        Trial vectors.
    nvec_pair_to_save : int
        Number of vector pairs to save.

    Returns
    -------
    int
        Number of new trial vectors returned.
    """
    # Full tddft case.
    if Mm is not None:
        full_dim = tri_vec.shape[1]
        assert (
            tri_vec.shape[0] >= nvec_pair_to_save
        ), 'Requested number of saved trial vectors is larger than the allocated space.'
        Mp_sym = (Mp + Mp.T) / 2.0
        Mm_sym = (Mm + Mm.T) / 2.0
        nprod = Mm.shape[0]
        exci_sqr, xpy_w = scipy.linalg.eigh(Mp_sym, Mm_sym, type=3)
        e_tri = np.sqrt(exci_sqr)
        emin_index = np.searchsorted(e_tri, e_min, side='left')

        if full_dim < 2 * nvec_pair_to_save:
            print('full_dim < 2*nvec_pair_to_save')
            Q, _, _ = sla.qr(tri_vec[:nprod].T, mode='economic', pivoting=True)
            tri_vec[:nprod] = Q.T
            return nprod

        # Truncate the eigenvectors and eigenvalues outside the target energy range.
        e_tri = e_tri[emin_index:]
        xpy_w = xpy_w[:, emin_index:]
        nvec_pair_to_save = min(nvec_pair_to_save, e_tri.size)

        # Calculate normalized |X+Y> and |X-Y> in subspace.
        xpy = xpy_w / np.sqrt(e_tri)[None, :]
        xmy = (Mp_sym @ xpy) / e_tri[None, :]

        # Write out the left and right vectors in the full space to a temporary file.
        # They're written to disk because they may be too large to fit in memory.
        with lib.H5TmpFile() as chkf:
            dset = chkf.create_dataset('tri_vec', shape=(2 * nvec_pair_to_save, full_dim), fillvalue=0)
            blksize = 10
            buf = np.empty((blksize, full_dim))
            for i in range(0, nvec_pair_to_save, blksize):
                if i + blksize < nvec_pair_to_save:
                    np.matmul(xpy[:, i : i + blksize].T, tri_vec[:nprod], out=buf)
                    dset.write_direct(buf, dest_sel=np.s_[2 * i : 2 * i + blksize])
                    np.matmul(xmy[:, i : i + blksize].T, tri_vec[:nprod], out=buf)
                    dset.write_direct(buf, dest_sel=np.s_[2 * i + blksize : 2 * i + 2 * blksize])
                else:
                    remaining = nvec_pair_to_save - i
                    np.matmul(xpy[:, i : i + remaining].T, tri_vec[:nprod], out=buf[:remaining])
                    dset.write_direct(buf[:remaining], dest_sel=np.s_[2 * i : 2 * i + remaining])
                    np.matmul(xmy[:, i : i + remaining].T, tri_vec[:nprod], out=buf[:remaining])
                    dset.write_direct(buf[:remaining], dest_sel=np.s_[2 * i + remaining : 2 * i + 2 * remaining])

            # Read the vectors back in and orthogonalize them.
            assert (tri_vec[: 2 * nvec_pair_to_save].T).flags.f_contiguous
            dset.read_direct(
                tri_vec, source_sel=np.s_[: 2 * nvec_pair_to_save], dest_sel=np.s_[: 2 * nvec_pair_to_save]
            )

            # In-place QR decomposition leaving orthogonal vectors Q as the rows
            # of tri_vec.
            lwork = sla.lapack.dgeqrf_lwork(2 * nvec_pair_to_save, full_dim)
            _, tau, _, _ = sla.lapack.dgeqrf(tri_vec[: 2 * nvec_pair_to_save].T, lwork=lwork, overwrite_a=1)
            sla.lapack.dorgqr(tri_vec[: 2 * nvec_pair_to_save].T, tau, overwrite_a=1)
        return 2 * nvec_pair_to_save

    # TDA case.
    else:
        full_dim = tri_vec.shape[1]
        assert (
            tri_vec.shape[0] >= nvec_pair_to_save
        ), 'Requested number of saved trial vectors is larger than the allocated space.'
        Mp_sym = (Mp + Mp.T) / 2.0
        nprod = Mp.shape[0]
        e_tri, x = scipy.linalg.eigh(Mp_sym)
        emin_index = np.searchsorted(e_tri, e_min, side='left')

        if full_dim < nvec_pair_to_save:
            print('full_dim < 2*nvec_pair_to_save')
            Q, R, _ = sla.qr(tri_vec[:nprod].T, mode='economic', pivoting=True)
            tri_vec[:nprod] = Q.T
            return nprod

        # Truncate the eigenvectors and eigenvalues outside the target energy range.
        e_tri = e_tri[emin_index:]
        x = x[:, emin_index:]
        nvec_pair_to_save = min(nvec_pair_to_save, e_tri.size)

        # Write out the left and right vectors in the full space to a temporary file.
        # They're written to disk because they may be too large to fit in memory.
        with lib.H5TmpFile() as chkf:
            dset = chkf.create_dataset('tri_vec', shape=(nvec_pair_to_save, full_dim), fillvalue=0)
            blksize = 10
            buf = np.empty((blksize, full_dim))
            for i in range(0, nvec_pair_to_save, blksize):
                if i + blksize < nvec_pair_to_save:
                    np.matmul(x[:, i : i + blksize].T, tri_vec[:nprod], out=buf)
                    dset.write_direct(buf, dest_sel=np.s_[i : i + blksize])
                else:
                    remaining = nvec_pair_to_save - i
                    np.matmul(x[:, i : i + remaining].T, tri_vec[:nprod], out=buf[:remaining])
                    dset.write_direct(buf[:remaining], dest_sel=np.s_[i : i + remaining])

            # Read the vectors back in and orthogonalize them.
            assert (tri_vec[:nvec_pair_to_save].T).flags.f_contiguous
            dset.read_direct(tri_vec, source_sel=np.s_[:nvec_pair_to_save], dest_sel=np.s_[:nvec_pair_to_save])

            # In-place QR decomposition leaving orthogonal vectors Q as the rows
            # of tri_vec.
            lwork = sla.lapack.dgeqrf_lwork(nvec_pair_to_save, full_dim)
            _, tau, _, _ = sla.lapack.dgeqrf(tri_vec[:nvec_pair_to_save].T, lwork=lwork, overwrite_a=1)
            sla.lapack.dorgqr(tri_vec[:nvec_pair_to_save].T, tau, overwrite_a=1)
        return nvec_pair_to_save


def davidson_save_checkpoint(chkfile, tri_vec, amb_prod, apb_prod, nprod):
    """Save the current state of the Davidson algorithm to a checkpoint file.

    Parameters
    ----------
    chkfile : str, pathlib.Path
        Path to the checkpoint file.
    tri_vec : ndarray
        Trial vectors.
    amb_prod : ndarray
        The vectors (A-B)|tri_vec>.
    apb_prod : ndarray
        The vectors (A+B)|tri_vec>.
    nprod : int
        The number of vectors to be written---we save the slice tri_vec[:nprod].
    """
    full_dim = tri_vec.shape[1]
    with h5py.File(chkfile, 'a') as chkf:
        if 'tri_vec' not in chkf:
            chkf.create_dataset('tri_vec', data=tri_vec, maxshape=(None, full_dim), chunks=(1, full_dim))
            chkf.create_dataset('amb_prod', data=amb_prod, maxshape=(None, full_dim), chunks=(1, full_dim))
            chkf.create_dataset('apb_prod', data=apb_prod, maxshape=(None, full_dim), chunks=(1, full_dim))
        else:
            old_ntri = chkf['tri_vec'].shape[0]
            # Discard old contents if we're overwriting.
            if nprod < old_ntri:
                old_ntri = 0
            writesel = np.s_[old_ntri:nprod]
            chkf['tri_vec'].resize((nprod, full_dim))
            chkf['amb_prod'].resize((nprod, full_dim))
            chkf['apb_prod'].resize((nprod, full_dim))
            chkf['tri_vec'].write_direct(tri_vec, source_sel=writesel, dest_sel=writesel)
            chkf['amb_prod'].write_direct(amb_prod, source_sel=writesel, dest_sel=writesel)
            chkf['apb_prod'].write_direct(apb_prod, source_sel=writesel, dest_sel=writesel)


def davidson_load_from_checkpoint(chkfile, tri_vec, amb_prod, apb_prod, nload=None):
    """Load the contents of a checkpoint file into the Davidson algorithm.

    Parameters
    ----------
    chkfile : str, pathlib.Path
        Path to the checkpoint file.
    tri_vec : ndarray
        Array to contain trial vectors.
    amb_prod : ndarray
        Array to contain the vectors (A-B)|tri_vec>.
    apb_prod : ndarray
        Array to contain the vectors (A+B)|tri_vec>.
    nload : int, optional
        Maximum number of trial vectors to load. If None, load all vectors.

    Returns
    -------
    int
        The number of trial vectors loaded.
    """
    with h5py.File(chkfile, 'r') as chkf:
        if 'tri_vec' not in chkf:
            raise ValueError('Checkpoint file does not contain tri_vec.')
        ntri = chkf['tri_vec'].shape[0]
        if nload is not None:
            ntri = min(ntri, nload)
        sel = np.s_[:ntri]
        for array in (tri_vec, amb_prod, apb_prod):
            if array.shape[0] < ntri:
                raise ValueError(f'max_vec is too small to load {ntri} vectors as requested.')
        chkf['tri_vec'].read_direct(tri_vec, source_sel=sel, dest_sel=sel)
        chkf['amb_prod'].read_direct(amb_prod, source_sel=sel, dest_sel=sel)
        chkf['apb_prod'].read_direct(apb_prod, source_sel=sel, dest_sel=sel)
    return ntri


def tddft_davidson(
    tddft,
    multi,
    e_min=0.0,
    delta=0.0,
    core_orbs=None,
    init_from_chkfile=None,
    expand_only_core=False,
    precond_exact_diag=False,
):
    """Davidson algorithm for tddft.
    The Davidson algorithm follows doi.org/10.1063/1.477483.
    tddft equation is defined as equation 1 in doi.org/10.1002/jcc.24688.
    Spin-adapted formalism can be found in chapter 18.3.2 in "Concepts and methods in modern theoretical chemistry.
    Electronic structure (2013, CRC) Ghosh S.K., Chattaraj P.K. (eds.)"

    Args:
        tddft (fcdmft.gw.mol.tddft.tddft): tddft object.
        multi (str): multiplicity, 's'=singlet, 't'=triplet, 'u'=unrestricted.
        e_min (float, optional): minimum desired excitation energy. Defaults to 0.0.
        delta (float, optional): energy shift for trial vector generation, typically <=0.0.
        core_orbs (optional) : filter function or AO labels or AO index, for generating trial vectors from core orbitals.
                    If this is provided, then e_min and delta are not used to generate trial vectors.
        init_from_chkfile (str, optional): checkpoint file to initialize the Davidson algorithm. Defaults to None.

    Returns:
        exci (double array): excitation energy.
        X_vec (double ndarray): X block of eigenvector (excitation).
        Y_vec (double ndarray): Y block of eigenvector (de-excitation).
    """
    # load matrix
    nspin = tddft.nspin
    nmo = tddft.nmo
    nocc = tddft.nocc
    mo_energy = tddft.mo_energy
    Lpq = tddft.Lpq
    # load parameter
    TDA = tddft.TDA
    max_vec = tddft.max_vec
    nroot = tddft.nroot
    max_iter = tddft.max_iter
    max_expand = tddft.max_expand
    init_ntri = max(2, tddft.init_ntri)
    residue_thresh = tddft.residue_thresh

    # determine dimension
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    full_dim = dim[0] + dim[1] if nspin == 2 else dim[0]

    # initialize trial vector
    tri_vec = np.zeros(shape=[max_vec, full_dim], dtype=np.double)
    ntri = min(init_ntri, full_dim)  # initial guess size should be larger than nroot

    if tddft.trial == 'identity':
        ntri_found, tri_vec_found = get_davidson_trial_vector(
            tddft, ntri=ntri, nocc=nocc, mo_energy=mo_energy, e_min=e_min, delta=delta, core_orbs=core_orbs
        )
    elif tddft.trial == 'subspace':
        ntri_found, tri_vec_found = get_davidson_trial_vector_diag(
            ntri, multi, nocc, mo_energy, Lpq, nocc_sub=tddft.nocc_sub, nvir_sub=tddft.nvir_sub, e_min=e_min, delta=delta,
            TDA=TDA
        )
    else:
        raise ValueError

    if ntri_found < ntri:
        lib.logger.info(tddft, f'only {ntri_found} trial vectors are generated rather than {ntri}.')
        ntri = ntri_found
    if ntri_found < init_ntri:
        raise ValueError('cannot find enough trial vectors; lower e_min or add more trial vectors')
    tri_vec[:ntri, :] = tri_vec_found
    del tri_vec_found

    # initialize Davidson matrix
    apb_prod = np.zeros_like(tri_vec)
    if not TDA:
        amb_prod = np.zeros_like(tri_vec)
    else:
        amb_prod = None

    Lia = [np.ascontiguousarray(Lpq[s][:, : nocc[s], nocc[s] :]) for s in range(nspin)]
    Laa = [np.ascontiguousarray(Lpq[s][:, nocc[s] :, nocc[s] :]) for s in range(nspin)]
    Lii = [np.ascontiguousarray(Lpq[s][:, : nocc[s] , : nocc[s] ]) for s in range(nspin)]

    if precond_exact_diag:
        assert TDA
        Laa_diag = [np.diagonal(Laa[s], axis1=1, axis2=2) for s in range(nspin)]
        Lii_diag = [np.diagonal(Lii[s], axis1=1, axis2=2) for s in range(nspin)]
        v_iaia = [
            2 / nspin * np.vecdot(Lia[s].reshape(-1, nocc[s] * nvir[s]).T, Lia[s].reshape(-1, nocc[s] * nvir[s]).T)
            for s in range(nspin)
        ]
        # TODO: add DFT kernel
        Kiiaa = [(Lii_diag[s].T @ Laa_diag[s]).reshape(nocc[s] * nvir[s]) for s in range(nspin)]
        Kiaia = [
            np.vecdot(Lia[s].reshape(-1, nocc[s] * nvir[s]).T, Lia[s].reshape(-1, nocc[s] * nvir[s]).T).reshape(
                nocc[s] * nvir[s]
            )
            for s in range(nspin)
        ]
        if TDA:
            apb_diag = [v_iaia[s] - Kiiaa[s] - Kiaia[s] for s in range(nspin)]
            amb_diag = apb_diag
        else:
            apb_diag = [2 * v_iaia[s] - Kiiaa[s] - Kiaia[s] for s in range(nspin)]
            amb_diag = [Kiaia[s] - Kiiaa[s] for s in range(nspin)]
        Laa_diag = None
        Lii_diag = None

    # We no longer need Lpq in this function.
    Lpq = None

    # Delete Lpq if it is not needed anymore.
    if tddft.delete_lpq:
        tddft.Lpq = None

    iter = 0
    nprod = 0  # the number of contracted vectors
    total_contract_work = 0
    total_linalg_work = 0

    Mm = None
    Mp = None

    if init_from_chkfile is not None:
        ntri = davidson_load_from_checkpoint(init_from_chkfile, tri_vec, amb_prod, apb_prod)
        lib.logger.info(tddft, f'Loaded {ntri} trial vectors from {init_from_chkfile}.')
        nprod = ntri

    chk_last = 0

    if tddft.ni is not None:
        if tddft.nspin == 1 and tddft.multi.lower() != 't': 
            tddft.rho0, tddft.vxc, tddft.fxc = tddft.ni.cache_xc_kernel(tddft.mol, tddft.mf.grids, tddft.mf.xc, tddft.mo_coeff[0], tddft.mo_occ[0], 0)
            tddft.rho0 = [tddft.rho0]
            tddft.vxc = [tddft.vxc]
            tddft.fxc = [tddft.fxc]
            tddft.dm0 = [tddft.dm0]
                
            
        elif tddft.nspin == 2:
            tddft.rho0, tddft.vxc, tddft.fxc = tddft.ni.cache_xc_kernel(tddft.mol, tddft.mf.grids, tddft.mf.xc, tddft.mf.mo_coeff, tddft.mf.mo_occ, 1)
            
        elif tddft.multi.lower() == 't':
            tddft.rho0, tddft.vxc, tddft.fxc = tddft.ni.cache_xc_kernel(tddft.mol, tddft.mf.grids, tddft.mf.xc, tddft.mf.mo_coeff, tddft.mf.mo_occ, 1)
            
            tddft.rho0 = [tddft.rho0]
            tddft.vxc = [tddft.vxc]
            tddft.fxc = [tddft.fxc]
            tddft.dm0 = [tddft.dm0]
                
        def numint_fn(tri_vec, dim, nocc, nvir):
            
            mem_now = lib.current_memory()[0]
            max_mem_ni = max(2000, tddft.mf.max_memory*.8-mem_now)
            
        
            if tddft.nspin == 1 :
                s = 0
                hermi = 0
                zs = tri_vec[s * dim[0] : s * dim[0] + dim[s]].reshape(nocc[s], nvir[s])
                # in AO basis    
                dm_zs  = lib.einsum('ov,pv,qo->pq', zs, tddft.mo_coeff[s][:,nocc[s]:], tddft.mo_coeff[s][:,:nocc[s]])
                if tddft.multi.lower() != 't':
                    v = tddft.ni.nr_rks_fxc(tddft.mol, tddft.mf.grids, tddft.mf.xc, tddft.dm0[s], dm_zs, 0, hermi, tddft.rho0[s], tddft.vxc[s], tddft.fxc[s], max_memory=max_mem_ni)   
                else:
                    v = 0.5*tddft.ni.nr_rks_fxc_st(tddft.mol, tddft.mf.grids, tddft.mf.xc, tddft.dm0, dm_zs, 0, False,
                                          tddft.rho0[s], tddft.vxc[s], tddft.fxc[s], max_memory=max_mem_ni)
                # return in MO basis
                v = lib.einsum('pq,qo,pv->ov', v, tddft.mo_coeff[s][:,:nocc[s]], tddft.mo_coeff[s][:,nocc[s]:]).reshape(-1)[np.newaxis, ...]
                return v
            
            else:
                hermi = 0
                dm_z = []
                for s in range(tddft.nspin):
                    zs = tri_vec[s * dim[0] : s * dim[0] + dim[s]].reshape(nocc[s], nvir[s])
                    dm_z.append(lib.einsum('ov,pv,qo->pq', zs, tddft.mo_coeff[s][:,nocc[s]:], tddft.mo_coeff[s][:,:nocc[s]]))
                
                dm_z = np.stack(dm_z)
                dm_z = (dm_z + dm_z.transpose(0,2,1))*0.5
                
                vdft_zs = tddft.ni.nr_uks_fxc(tddft.mol, tddft.mf.grids, tddft.mf.xc, tddft.dm0, dm_z, 0, hermi, tddft.rho0, tddft.vxc, tddft.fxc, max_memory=max_mem_ni)
                vdft_z = []
                for s in range(tddft.nspin):
                    vdft_z.append(lib.einsum('pq,qo,pv->ov', vdft_zs[s], tddft.mo_coeff[s][:,:nocc[s]], tddft.mo_coeff[s][:,nocc[s]:]).reshape(-1))

                return vdft_z
                            
    else:
        numint_fn = None
        

    while iter < max_iter:
        lib.logger.info(tddft, '\ntddft Davidson #%d iteration, ntri= %d , nprod= %d .', iter + 1, ntri, nprod)
        if not TDA:
            apb_prod[nprod:ntri, :], amb_prod[nprod:ntri, :], contract_work_this_iter = _tddft_contraction(
                multi=multi,
                nocc=nocc,
                mo_energy=mo_energy,
                Lia=Lia,
                Laa=Laa,
                Lii=Lii,
                
                tri_vec=tri_vec[nprod:ntri, :],
                TDA=False,
                hybrid = tddft.hybrid,
                hyb_coeff = tddft.hyb,
                ni_fn = numint_fn
            )
        else:
            apb_prod[nprod:ntri, :], _, contract_work_this_iter = _tddft_contraction(
                multi=multi,
                nocc=nocc,
                mo_energy=mo_energy,
                Lia=Lia,
                Laa=Laa,
                Lii=Lii,
                
                tri_vec=tri_vec[nprod:ntri, :],
                TDA=True,
                hybrid = tddft.hybrid,
                hyb_coeff = tddft.hyb,
                ni_fn = numint_fn
            )
        total_contract_work += contract_work_this_iter
        lib.logger.info(tddft, f'work for iter {iter+1}: {float(contract_work_this_iter):.2E}')

        Mp, Mm, mmwork = update_mp_mm(Mp, Mm, tri_vec, apb_prod, amb_prod, ntri, nprod)
        Mp_sym = (Mp + Mp.T) / 2.0
        if not TDA:
            Mm_sym = (Mm + Mm.T) / 2.0
        total_linalg_work += mmwork
        nprod_prev, nprod = nprod, ntri

        if tddft.chkfile is not None:
            if nprod - chk_last >= tddft.chk_every:
                davidson_save_checkpoint(tddft.chkfile, tri_vec, amb_prod, apb_prod, nprod)
                lib.logger.info(tddft, f'Saving progress at iteration {iter+1} to {tddft.chkfile}: {chk_last}->{nprod}.')
                chk_last = nprod

        nroot_current = min(nroot, ntri)
        # equation 15 in doi/10.1063/1.477483, solved by LAPACK function dsygvd

        # Save current NumPy error handling settings
        nperrhandling = np.geterr()['invalid']
        try:
            if not TDA:
                exci_sqr, xpy_w = scipy.linalg.eigh(Mp_sym.T, Mm_sym.T, type=3)
                np.seterr(invalid='raise')
                e_tri = np.sqrt(exci_sqr)
            else:
                np.seterr(invalid='raise')
                e_tri, xpy_w = scipy.linalg.eigh(Mp_sym.T, driver='evd')

        except (scipy.linalg.LinAlgError, FloatingPointError):
            lib.logger.warn(tddft, 'Restarting Davidson algorithm.')
            # restart Davidson algorithm
            # Throw away most recent trial vectors, since they are likely to be linearly dependent
            if tddft.restart_max_size is None:
                nvec_pair_to_save = nprod_prev
            else:
                nvec_pair_to_save = min(nprod_prev, tddft.restart_max_size)
            if not TDA:
                ntri = davidson_restart(
                    Mp[:nprod_prev, :nprod_prev],
                    Mm[:nprod_prev, :nprod_prev],
                    tri_vec,
                    nvec_pair_to_save=nvec_pair_to_save,
                    e_min=e_min,
                )
            else:
                ntri = davidson_restart(
                    Mp[:nprod_prev, :nprod_prev], None, tri_vec, nvec_pair_to_save=nvec_pair_to_save, e_min=e_min
                )

            # Set nprod to 0 to recalculate all mat-vec products.
            nprod = 0
            Mp = None
            Mm = None
            iter += 1
            continue

        finally:
            # Restore NumPy error handling settings
            np.seterr(invalid=nperrhandling)

        if not TDA:
            # dsygvd normalizes xpy_w such that
            # xpy_w @ xpy_w.T = A - B
            # Using the fact that A - B = (X+Y) @ diag(w) @ (X+Y).T,
            # we calculate X+Y = xpy_w @ diag(1/sqrt(w)).
            xpy = xpy_w / np.sqrt(e_tri)[None, :]

            # (A+B) |X+Y> = w |X-Y>, so
            # |X-Y> = w^-1 (A+B) |X+Y>
            xmy = (Mp_sym @ xpy) / e_tri[None, :]

            # Thanks to the use of the generalized eigensolver,
            # xpy and xmy already form a biorthonormal system.

        else:
            # TDA is easy
            xpy = xpy_w

        total_linalg_work += ntri**3

        found_roots = np.flatnonzero(e_tri >= e_min)
        nrootfound = min(nroot, found_roots.size)
        lib.logger.debug(tddft, 'lowest %d exci above minimum: \n%s', nrootfound, e_tri[found_roots[:nrootfound]])
        emin_index = np.searchsorted(e_tri, e_min, side='left')
        if emin_index + nroot_current > ntri:
            emin_index = ntri - nroot_current
            if ntri >= nroot:
                lib.logger.info(tddft, 'fewer than nroot exci found above e_min.')

        if core_orbs is not None and nspin == 1 and expand_only_core:
            if not hasattr(tddft, 'mol'):
                raise ValueError('mol object is required if core_orbs is given.')
            # Select those occupied orbitals with a significant contribution from given core orbitals.
            occ_we_want = np.flatnonzero(
                mo_mapping.mo_comps(core_orbs, tddft.mol, tddft.mo_coeff[0][:, : nocc[0]]) >= 0.3
            )
            core_roots = []

            for idx in range(emin_index, ntri):
                if not TDA:
                    Xvec = (0.5 * (xpy[:, idx].T + xmy[:, idx].T)) @ tri_vec[:ntri, :]
                else:
                    Xvec = xpy[:, idx].T @ tri_vec[:ntri, :]
                Xvec = Xvec.reshape(nocc[0], nvir[0])
                Xvecsqr = np.linalg.norm(Xvec, axis=1)
                X_core_component = np.linalg.norm(Xvecsqr[occ_we_want])
                if X_core_component > 0.3:
                    core_roots.append(idx)
                if len(core_roots) >= nroot_current:
                    break
            exci_candidate_indices = np.asarray(core_roots, dtype=int)
            lib.logger.debug(
                tddft,
                'lowest %d core excitations above minimum: \n%s',
                exci_candidate_indices.size,
                e_tri[exci_candidate_indices],
            )

        else:
            exci_candidate_indices = np.s_[emin_index : emin_index + nroot_current]

        ntri_old = ntri

        exci = e_tri[exci_candidate_indices]
        # get left and right eigenvector in the full space, equation 25 and 26 in doi.org/10.1063/1.477483

        right_vec_tri = xpy.T[exci_candidate_indices, :]
        right_vec = np.matmul(right_vec_tri, tri_vec[:ntri, :])
        total_linalg_work += nroot_current * ntri * full_dim

        if not TDA:
            left_vec_tri = xmy.T[exci_candidate_indices, :]
            left_vec = np.matmul(left_vec_tri, tri_vec[:ntri, :])
            total_linalg_work += nroot_current * ntri * full_dim

        if not TDA:
            right_res = -exci[:, None] * left_vec
            left_res = -exci[:, None] * right_vec
            right_res += np.matmul(right_vec_tri, apb_prod[:ntri, :])
            left_res += np.matmul(left_vec_tri, amb_prod[:ntri, :])

            # check convergence
            res_norms_left = np.linalg.norm(left_res, axis=1) ** 2
            res_norms_right = np.linalg.norm(right_res, axis=1) ** 2
            res_norms = np.maximum(res_norms_left, res_norms_right)

        else:  # TDA
            right_res = -exci[:, None] * right_vec
            right_res += np.matmul(right_vec_tri, apb_prod[:ntri, :])
            res_norms = np.linalg.norm(right_res, axis=1) ** 2

        max_res_norm = np.max(res_norms)
        conv_vec = res_norms < residue_thresh
        lib.logger.info(tddft, 'max residue norm = %.4e', max_res_norm)
        if conv_vec.size >= nroot:
            if np.all(conv_vec[:nroot]):
                conv = True
                break

        not_converged = np.flatnonzero(~conv_vec)
        errs_not_converged = res_norms[not_converged]
        assert np.max(errs_not_converged) == max_res_norm
        srt_errs = np.argsort(errs_not_converged)[::-1]
        nexpand = min(max_expand, nroot_current, not_converged.size, full_dim - ntri)
        candidates_to_expand = not_converged[srt_errs[:nexpand]]

        # Gather both left and right residues
        if not TDA:
            all_res = np.empty(shape=(2 * nexpand, full_dim), dtype=np.double)
        else:
            all_res = np.empty(shape=(nexpand, full_dim), dtype=np.double)

        # preconditioning the residues, equation 29 in doi.org/10.1063/1.477483.
        for s in range(nspin):
            q_vec = exci[candidates_to_expand, None, None] - (
                mo_energy[s][None, None, nocc[s] :] - mo_energy[s][None, : nocc[s], None]
            )
            q_vec = q_vec.reshape(-1, nocc[s] * nvir[s])
            if precond_exact_diag:
                q_vec -= apb_diag[s].reshape(-1, nocc[s] * nvir[s])
            all_res[:nexpand, s * dim[0] : s * dim[0] + dim[s]] = (
                right_res[candidates_to_expand, s * dim[0] : s * dim[0] + dim[s]] / q_vec
            )
            if not TDA:
                all_res[nexpand:, s * dim[0] : s * dim[0] + dim[s]] = (
                    left_res[candidates_to_expand, s * dim[0] : s * dim[0] + dim[s]] / q_vec
                )

        # The rows of all_res are now the preconditioned left residues
        # followed by the preconditioned right residues.

        # Orthogonalize residues against current trial vectors
        all_res -= (all_res @ tri_vec[:ntri, :].T) @ tri_vec[:ntri, :]
        # Orthogonalize residues amongst themselves
        Q, R, _ = scipy.linalg.qr(all_res.T, mode='economic', pivoting=True)

        # Don't care about the small residues
        orth_res = Q.T[np.abs(np.diag(R)) > 1e-10]
        # But we should take at least one new vector.
        if orth_res.size == 0:
            orth_res = Q.T[:1]

        # Make sure the residues are orthogonal to the trial vectors
        # and normalize them.
        orth_res -= (orth_res @ tri_vec[:ntri, :].T) @ tri_vec[:ntri, :]
        orth_res /= np.linalg.norm(orth_res, axis=1)[:, None]

        n_new_vec = min(orth_res.shape[0], full_dim - ntri)
        if n_new_vec > 0:
            if ntri + n_new_vec > tri_vec.shape[0]:
                raise ValueError('Exceeded max_vec. Davidson algorithm for tddft is not converged!')
            tri_vec[ntri : ntri + n_new_vec] = orth_res[:n_new_vec]
            ntri += n_new_vec
            lib.logger.info(tddft, 'add %d new trial vectors.', n_new_vec)
        else:
            # We need to restart.
            lib.logger.warn(tddft, 'Restarting Davidson algorithm.')
            if tddft.restart_max_size is None:
                nvec_pair_to_save = ntri
            else:
                nvec_pair_to_save = min(ntri, tddft.restart_max_size)
            ntri = davidson_restart(Mp, Mm, tri_vec, nvec_pair_to_save=nvec_pair_to_save, e_min=e_min)
        conv = False

        iter += 1
        if conv is True:
            break

    assert conv is True, 'Davidson algorithm for tddft is not converged!'

    if tddft.chkfile is not None:
        davidson_save_checkpoint(tddft.chkfile, tri_vec, amb_prod, apb_prod, nprod)
        lib.logger.info(tddft, f'Saving progress at iteration {iter+1} to {tddft.chkfile}: {chk_last}->{nprod}.')
        chk_last = nprod

    lib.logger.info(tddft, f'tddft converged in {iter} iterations, final subspace size = {nprod}')
    lib.logger.info(tddft, f'total work for contraction: {float(total_contract_work):.2E}')
    lib.logger.info(tddft, f'total work for linalg: {float(total_linalg_work):.2E}')
    lib.logger.info(tddft, f'Mp condition number: {np.linalg.cond(Mp_sym)}')
    if Mm is not None:
        lib.logger.info(tddft, f'Mm condition number: {np.linalg.cond(Mm_sym)}')

    found_roots = np.flatnonzero((exci >= e_min) & conv_vec)
    nrootfound = found_roots.size
    lib.logger.debug(tddft, 'Finished with %d converged roots: \n%s', nrootfound, exci[found_roots])

    # transfer left and right eigenvector to X and Y

    if not TDA:
        X_vec = (left_vec[found_roots] + right_vec[found_roots]) * 0.5
        Y_vec = (-left_vec[found_roots] + right_vec[found_roots]) * 0.5
    else:
        X_vec = right_vec[found_roots]
        Y_vec = np.zeros_like(X_vec)

    # reshape X and Y eigenvector
    if nspin == 1:
        X_vec = [X_vec.reshape(nrootfound, nocc[0], nvir[0])]
        Y_vec = [Y_vec.reshape(nrootfound, nocc[0], nvir[0])]
    else:
        X_vec_a, X_vec_b, Y_vec_a, Y_vec_b = [], [], [], []
        for r in range(nrootfound):
            X_vec_a.append(X_vec[r][: dim[0]].reshape(nocc[0], nvir[0]))
            X_vec_b.append(X_vec[r][dim[0] :].reshape(nocc[1], nvir[1]))
            Y_vec_a.append(Y_vec[r][: dim[0]].reshape(nocc[0], nvir[0]))
            Y_vec_b.append(Y_vec[r][dim[0] :].reshape(nocc[1], nvir[1]))
        X_vec = [np.asarray(X_vec_a), np.asarray(X_vec_b)]
        Y_vec = [np.asarray(Y_vec_a), np.asarray(Y_vec_b)]

    tddft.exci = exci[found_roots]
    tddft.X_vec = X_vec
    tddft.Y_vec = Y_vec

    return exci[found_roots], X_vec, Y_vec


def update_mp_mm(Mp, Mm, tri_vec, apb_prod, amb_prod, ntri, nprod):
    """Update Mp and Mm to reflect the new trial vectors.

    Parameters
    ----------
    Mp : ndarray
        The matrix <tri_vec|A+B|tri_vec>
    Mm : ndarray or None
        The matrix <tri_vec|A-B|tri_vec>
    tri_vec : ndarray
        Trial vectors (stored as rows).
    apb_prod : ndarray
        The vectors (A+B)|tri_vec> (stored as rows).
    amb_prod : ndarray or None
        The vectors (A-B)|tri_vec> (stored as rows).
    ntri : int
        Number of valid trial vectors in tri_vec.
    nprod : int
        Number of valid trial vectors when Mm and Mp were last updated.

    Returns
    -------
    (ndarray, ndarray, int)
        Mm, Mp, work; where work is a rough estimate of the FLOP count.
    """
    full_dim = tri_vec.shape[1]
    work = 0
    if Mp is None or Mm is None:
        # A+B and A-B in subspace, step 3 in doi.org/10.1063/1.477483
        if apb_prod is not None:
            Mp = np.matmul(tri_vec[:ntri, :], apb_prod[:ntri, :].T)
            work += ntri**2 * full_dim

        if amb_prod is not None:
            Mm = np.matmul(tri_vec[:ntri, :], amb_prod[:ntri, :].T)
            work += ntri**2 * full_dim

    else:
        if apb_prod is not None:
            Mp_new = np.zeros(shape=[ntri, ntri], dtype=np.double)
            Mp_new[:nprod, :nprod] = Mp[:nprod, :nprod]
            Mp_new[nprod:ntri, :ntri] = tri_vec[nprod:ntri, :] @ apb_prod[:ntri, :].T
            Mp_new[:ntri, nprod:ntri] = Mp_new[nprod:ntri, :ntri].T
            Mp_new[nprod:ntri, nprod:ntri] = tri_vec[nprod:ntri, :] @ apb_prod[nprod:ntri, :].T
            Mp = Mp_new
            work += (ntri**2 - nprod**2) * full_dim

        if amb_prod is not None:
            Mm_new = np.zeros(shape=[ntri, ntri], dtype=np.double)
            Mm_new[:nprod, :nprod] = Mm[:nprod, :nprod]
            Mm_new[nprod:ntri, :ntri] = tri_vec[nprod:ntri, :] @ amb_prod[:ntri, :].T
            Mm_new[:ntri, nprod:ntri] = Mm_new[nprod:ntri, :ntri].T
            Mm_new[nprod:ntri, nprod:ntri] = tri_vec[nprod:ntri, :] @ amb_prod[nprod:ntri, :].T
            Mm = Mm_new
            work += (ntri**2 - nprod**2) * full_dim

    return Mp, Mm, work


def tddft_lanczos(tddft, multi, u1=None, core_orbs=None, nsteps=100):
    """Lanczos algorithm for tddft.
    Follows 10.1137/16M1102641.

    Args:
        tddft (fcdmft.gw.mol.tddft.tddft): tddft object.
        multi (str): multiplicity, 's'=singlet, 't'=triplet, 'u'=unrestricted.
        u1 (double ndarray, optional): initial state for Lanczos algorithm. Defaults to None.
        nsteps (int, optional): the number of Lanczos steps. Defaults to 100.

    Returns:
        exci (double array): excitation energy.
        X_vec (double ndarray): X block of eigenvector (excitation).
        Y_vec (double ndarray): Y block of eigenvector (de-excitation).
    """
    # load matrix
    nspin = tddft.nspin
    nmo = tddft.nmo
    nocc = tddft.nocc
    mo_energy = tddft.mo_energy
    # load parameter
    TDA = tddft.TDA

    # determine dimension
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    full_dim = dim[0] + dim[1] if nspin == 2 else dim[0]

    Lii = [np.ascontiguousarray(tddft.Lpq[s][:, : nocc[s] , : nocc[s] ]) for s in range(nspin)]
    Laa = [np.ascontiguousarray(tddft.Lpq[s][:, nocc[s] :, nocc[s] :]) for s in range(nspin)]
    Lia = [np.ascontiguousarray(tddft.Lpq[s][:, : nocc[s], nocc[s] :]) for s in range(nspin)]

    prev_vecs = np.zeros((nsteps + 1, full_dim))

    if core_orbs is not None:
        assert u1 is None, 'u1 and core_orbs cannot be used together'
        u1 = np.zeros(full_dim)
        occ_to_take = [
            np.flatnonzero(mo_mapping.mo_comps(core_orbs, tddft.mol, tddft.mo_coeff[s]) >= 0.5) for s in range(nspin)
        ]
        for s in range(nspin):
            vir_to_take = np.arange(nocc[s], nmo, dtype=int)
            for o in occ_to_take[s]:
                u1[s * dim[s] + o * nvir[s] + vir_to_take] = 1.0
        u1 = u1 / np.linalg.norm(u1)

    elif u1 is None:
        eia = []
        for s in range(nspin):
            eia.append(np.asarray(mo_energy[s][None, nocc[s] :] - mo_energy[s][: nocc[s], None]).reshape(-1))
        eia = np.concatenate(eia, axis=0)
        u1 = np.random.random(full_dim) - 0.5
        u1 = u1 / np.linalg.norm(u1)

    apb_u1, _, _ = _tddft_contraction(
        multi=multi,
        nocc=nocc,
        mo_energy=mo_energy,
        Lia=Lia,
        Laa=Laa,
        Lii=Lii,
        
        tri_vec=u1[None, :],
        TDA=TDA,
    )

    apb_u1 = apb_u1.reshape(-1)

    betas = np.zeros(nsteps)
    alphas = np.zeros(nsteps)

    if TDA is False:
        u1_apbnorm = np.dot(u1, apb_u1)
        u = u1 / np.sqrt(u1_apbnorm)
        v = apb_u1 / np.sqrt(u1_apbnorm)
    else:
        u = u1 / np.linalg.norm(u1)
        v = u

    u_last = np.zeros_like(u)
    v_last = np.zeros_like(v)
    beta_last = 0.0

    prev_vecs[0] = v
    nprev = 1

    for step in range(nsteps):
        lib.logger.debug(tddft, 'tddft Lanczos #%d iteration', step + 1)
        if TDA is False:
            # x = (A - B) v_j - beta_{j-1} u_{j-1}
            _, amb_v, _ = _tddft_contraction(
                multi=multi,
                nocc=nocc,
                mo_energy=mo_energy,
                Lia=Lia,
                Laa=Laa,
                Lii=Lii,
                
                tri_vec=v.reshape((1, -1)),
                TDA=TDA,
            )
            amb_v = amb_v.reshape(-1)
            sla.blas.daxpy(u_last, amb_v, a=-beta_last)
            x = amb_v
            # alpha = v_j^T x
            alphas[step] = np.dot(x, v)
            # x = x - alpha u_j
            sla.blas.daxpy(u, x, a=-alphas[step])
            # y = (A + B) x
            y, _, _ = _tddft_contraction(
                multi=multi,
                nocc=nocc,
                mo_energy=mo_energy,
                Lia=Lia,
                Laa=Laa,
                Lii=Lii,
                
                tri_vec=x.reshape((1, -1)),
                TDA=TDA,
            )
            y = y.reshape(-1)
            # beta_j = sqrt(x^T y)
            betas[step] = np.sqrt(np.dot(x, y))
            u_last = u
            v_last = v
            # u_{j+1} = x / beta_j
            # v_{j+1} = y / beta_j
            sla.blas.dscal(1.0 / betas[step], x)
            sla.blas.dscal(1.0 / betas[step], y)
            u = x
            v = y
        else:
            # TDA approximation
            # v = A u_j - beta_{j-1} u_{j-1}
            v, _, _ = _tddft_contraction(
                multi=multi,
                nocc=nocc,
                mo_energy=mo_energy,
                Lia=Lia,
                Laa=Laa,
                Lii=Lii,
                
                tri_vec=u.reshape((1, -1)),
                TDA=TDA,
            )
            v = v.reshape(-1)
            sla.blas.daxpy(u_last, v, a=-beta_last)
            # alpha_j = u_j^T v
            alphas[step] = np.dot(u, v)
            # v = v - alpha u_j
            sla.blas.daxpy(u, v, a=-alphas[step])

            # orthogonalize against previous vectors
            hs = prev_vecs[:nprev] @ v
            v -= prev_vecs[:nprev].T @ hs

            # beta_j = ||v||
            betas[step] = np.linalg.norm(v)
            # u_{j+1} = v / beta_j
            sla.blas.dscal(1.0 / betas[step], v)
            u_last = u
            u = v
            prev_vecs[nprev] = v
            nprev += 1
        beta_last = betas[step]
    return alphas, betas


def lanczos_roots_magnitudes(alphas, betas, TDA=False):
    """Estimate the excitation spectrum density from the results of the Lanczos algorithm.

    Parameters
    ----------
    alphas (double array): coefficients from the Lanczos algorithm.
        diagonal elements of the tridiagonal matrix.
    betas (double array): coefficients from the Lanczos algorithm.
        off-diagonal elements of the tridiagonal matrix.
    e_range (tuple): energy range (e_min, e_max).
    eta (float): broadening factor.
    nw (int): number of frequency points.
    TDA (bool, optional): Lanczos used TDA approximation. Defaults to False.

    Returns
    -------
    freqs (double array): frequency points at which to compute density estimate.
    density (double array): excitation spectrum density estimate.
    """
    Tk_diag = np.concatenate([alphas, alphas[-2::-1]], axis=0)
    Tk_offdiag = np.concatenate([betas, betas[-3::-1]], axis=0)
    roots, S = scipy.linalg.eigh_tridiagonal(Tk_diag, Tk_offdiag, lapack_driver='stebz')
    roots_pos = roots[roots > 0]
    if TDA is False:
        roots_pos = np.sqrt(roots_pos)
    magnitudes = S[0, roots > 0] ** 2

    if TDA:
        return roots_pos, magnitudes
    else:
        return roots_pos, magnitudes / roots_pos


def lanczos_estimate_spectrum(alphas, betas, e_range, eta, nw, TDA=False):
    """Estimate the excitation spectrum density from the results of the Lanczos algorithm.

    Parameters
    ----------
    alphas (double array): coefficients from the Lanczos algorithm.
        diagonal elements of the tridiagonal matrix.
    betas (double array): coefficients from the Lanczos algorithm.
        off-diagonal elements of the tridiagonal matrix.
    e_range (tuple): energy range (e_min, e_max).
    eta (float): broadening factor.
    nw (int): number of frequency points.
    TDA (bool, optional): Lanczos used TDA approximation. Defaults to False.

    Returns
    -------
    freqs (double array): frequency points at which to compute density estimate.
    density (double array): excitation spectrum density estimate.
    """

    roots_pos, magnitudes = lanczos_roots_magnitudes(alphas, betas, TDA=TDA)

    freqs = np.linspace(e_range[0], e_range[1], nw)

    def gauss_broad(omega, eta, roots):
        normalization = 1.0 / np.sqrt(2 * np.pi * eta**2)
        return normalization * (
            np.exp(-((omega[:, None] - roots[None, :]) ** 2) / (2 * eta**2))
            - np.exp(-((omega[:, None] + roots[None, :]) ** 2) / (2 * eta**2))
        )

    density = gauss_broad(freqs, eta, roots_pos) @ magnitudes
    return freqs, density


def get_davidson_trial_vector(tddft, ntri, nocc, mo_energy, e_min=0.0, delta=0.0, core_orbs=None):
    """Generate initial trial vectors for particle-hole excitations.
    The order is determined by the occ-vir pair orbital energy difference.
    The initial trial vectors are diagonal. They are generated by taking
    occ-vir pairs with an energy difference of >= e_min + delta.

    Args:
        ntri (int): the number of desired initial trial vectors.
        nocc (int array): the number of occupied orbitals.
        mo_energy (double ndarray): orbital energy.
        e_min (float, optional): minimum desired excitation energy. Defaults to 0.0.
        delta (float, optional): energy shift for trial vector generation, typically <=0.0.

    Returns:
        ntri, int: the number of actual trial vectors generated
        tri_vec, double ndarray: initial trial vectors
    """
    nspin, nmo = mo_energy.shape
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    full_dim = dim[0] + dim[1] if nspin == 2 else dim[0]

    if core_orbs is not None:
        if not hasattr(tddft, 'mol'):
            raise ValueError('mol object is required for generating trial vectors for core excitations.')
        # Select those occupied orbitals with a significant contribution from given core orbitals.
        occ_to_take = [
            np.flatnonzero(mo_mapping.mo_comps(core_orbs, tddft.mol, tddft.mo_coeff[s]) >= 0.3) for s in range(nspin)
        ]
    else:
        occ_to_take = [np.arange(nocc[s], dtype=int) for s in range(nspin)]

    e_diffs = []
    e_diffs_shp = []

    for s in range(nspin):
        # The shape of e_diffs_s is (nocc[s], nvir[s])
        # e_diffs_s[i, a] = mo_energy[s][a] - mo_energy[s][i]
        e_diffs_s = mo_energy[s][None, nocc[s] :] - mo_energy[s][occ_to_take[s], None]
        e_diffs_shp.append(e_diffs_s.shape)
        # Flatten e_diffs[s] into a 1D array.
        e_diffs_s = e_diffs_s.reshape(-1)
        e_diffs.append(e_diffs_s)

    # At this point, the structure of e_diffs is as follows:
    # e_diffs[spin, ia] = mo_energy[spin][a] - mo_energy[spin][i]
    # where ia = a + nvir[spin] * i

    # Glue the e_diffs together into a 1D array.
    all_ediffs = np.concatenate(e_diffs, axis=0)

    # Compute the sizes of the occ-vir blocks for each spin.
    e_diffs_sizes = [0] + [nocc[s] * nvir[s] for s in range(nspin)]
    # Compute the starting index of each spin's occ-vir block.
    # This indicates where e_diffs[s] resides in all_ediffs, for each s.
    e_diffs_starts = np.cumsum(e_diffs_sizes)

    # Find the indices which sort all_ediffs.
    sort_index = np.argsort(all_ediffs)

    # Take the lowest ntri pairs with energy difference greater than e_min + delta.
    e_min_index = np.searchsorted(all_ediffs, e_min + delta, side='left', sorter=sort_index)
    if e_min_index + ntri > all_ediffs.size:
        # cannot find enough pairs for trial vectors; lower e_min
        ntri = all_ediffs.size - e_min_index
    exci_to_take = sort_index[e_min_index : e_min_index + ntri]

    # exci_to_take is an index into all_ediffs.
    # We need to convert it back to orbital indices.

    tri_vec = np.zeros(shape=[ntri, full_dim], dtype=np.double)

    cur_trivec = 0
    for s in range(nspin):
        # Figure out which excitation indices are in this spin block.
        exci_this_spin = np.extract(
            (exci_to_take >= e_diffs_starts[s]) & (exci_to_take < e_diffs_starts[s + 1]), exci_to_take
        )
        # Subtract the starting index of this spin's occ-vir block.
        # They are now in the form ia = i * nvir[s] + a.
        # That is, they are indices into e_diffs[s].reshape(-1).
        exci_this_spin -= e_diffs_starts[s]
        # Convert the indices from 1D form (i * nvir[s] + a) to 2D form (i, a).
        ex_occ, ex_vir = np.unravel_index(exci_this_spin, e_diffs_shp[s])
        ex_occ = occ_to_take[s][ex_occ]
        n_exci = exci_this_spin.size

        # The following is shorthand for
        # for i, a in zip(ex_occ, ex_vir):
        #     tri_vec[cur_trivec, s * dim[s] + i * nvir[s] + a] = 1.
        #     cur_trivec += 1
        tri_vec[range(cur_trivec, cur_trivec + n_exci), s * dim[s] + ex_occ * nvir[s] + ex_vir] = 1.0
        cur_trivec += n_exci

    return ntri, tri_vec


def get_davidson_trial_vector_diag(
    ntri, multi, nocc, mo_energy, Lpq, nocc_sub=50, nvir_sub=150, e_min=0.0, delta=0.0, TDA=False
):
    """Get trial vectors from subspace diagnoalization.

    Parameters
    ----------
    ntri : int
        number of trial vectors
    multi : str
        multiplicity
    nocc : list
        number of occupied orbitals
    mo_energy : ndarray
        orbital energy
    Lpq : ndarray
        three-center density-fitting matrix
    nocc_sub : int, optional
        number of subspace occupied orbitals, by default 50
    nvir_sub : int, optional
        number of subspace virtual orbitals, by default 150
    e_min : float, optional
        minimum desired excitation energy, by default 0.0
    delta : float, optional
        energy shift for trial vector generation, typically <=0.0, by default 0.0
    TDA : bool, optional
        use Tamm-Dancoff approximation, by default False

    Returns
    -------
    ntri : int
        the number of actual trial vectors generated
    tri_vec : double ndarray
        initial trial vectors
    """
    nspin, nmo = mo_energy.shape
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]

    # adjust active space if necessary
    nocc_sub = int(min(nocc[0], nocc_sub))
    nvir_sub = int(min(nvir[0], nvir_sub))

    if nspin == 1:
        nocc_sub = [nocc_sub]
        nvir_sub = [nvir_sub]
    else:
        # numbers of beta orbitals are determined by alpha
        spin = nocc[0] - nocc[1]
        nocc_sub = [nocc_sub, nocc_sub - spin]
        nvir_sub = [nvir_sub, nvir_sub + spin]

    # get active-space tddft input
    start = [(nocc[s] - nocc_sub[s]) for s in range(nspin)]
    end = [(nocc[s] + nvir_sub[s]) for s in range(nspin)]
    mo_energy_sub = np.asarray([mo_energy[s, start[s] : end[s]] for s in range(nspin)])
    Lpq_sub = np.asarray([Lpq[s, :, start[s] : end[s], start[s] : end[s]] for s in range(nspin)])

    exci, X_vec, Y_vec = tddft_full_diagonalization(
        multi=multi, nocc=nocc_sub, mo_energy=mo_energy_sub, Lpq=Lpq_sub, TDA=TDA
    )

    for i in range(len(exci)):
        if exci[i] > (e_min + delta):
            first_state = i
            break

    ntri = min(ntri, len(exci) - first_state)
    tri_vec = []
    for s in range(nspin):
        tri_vec.append(np.zeros(shape=[ntri, nocc[s], nvir[s]], dtype=np.double))
        X_vec_tri = X_vec[s][first_state : first_state + ntri].reshape(ntri, nocc_sub[s], nvir_sub[s])
        tri_vec[s][:, nocc[s] - nocc_sub[s] :, :nvir_sub[s]] = X_vec_tri
        tri_vec[s] = tri_vec[s].reshape(ntri, dim[s])
    tri_vec = np.concatenate(tri_vec, axis=1)

    return ntri, tri_vec


def _tddft_contraction(multi, nocc, mo_energy, Lia, Laa, Lii, tri_vec, TDA=False, ni_fn = None,
                      hyb_coeff = 1.0, hybrid = True):
    """Contraction for TDDFT matrix and trial vectors.
    W part is as equation 25 and 26 in doi.org/10.1002/jcc.24688.

    Args:
        multi (str): multiplicity, 's'=singlet, 't'=triplet, 'u'=unrestricted.
        nocc (int array): the number of occupied orbitals.
        mo_energy (double ndarray): orbital energy.
        Lia (double ndarray): 3-center density-fitting matrix, ov block.
        Laa (double ndarray): 3-center density-fitting matrix, vv block.
        Lii (double ndarray): 3-center density-fitting matrix, oo block.
        tri_vec (double ndarray): trial vector.
        TDA (bool, optional): use TDA approximation. Defaults to False. If True, only apb_prod is returned.

    Returns:
        apb_prod, double ndarray: A+B matrix and trial vector contracted vectors.
        amb_prod, double ndarray: A-B matrix and trial vector contracted vectors.
    """
    nspin = len(Lia)
    naux, _, _ = Lia[0].shape
    nmo = Lia[0].shape[2] + Lia[0].shape[1]
    ntri = tri_vec.shape[0]

    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    full_dim = dim[0] + dim[1] if nspin == 2 else dim[0]

    work_done = 0

    scale = 4.0 / nspin
    if TDA is True:
        scale /= 2.0

    apb_prod = np.zeros(shape=[ntri, full_dim], dtype=np.double)
    if TDA:
        amb_prod = None
    else:
        amb_prod = np.zeros(shape=[ntri, full_dim], dtype=np.double)

    # contraction: V
    if multi != 't' and multi != 'T':
        for ivec in range(ntri):
            Lpq_z = np.empty(shape=[nspin, naux], dtype=np.double)
            for s in range(nspin):
                z = tri_vec[ivec][s * dim[0] : s * dim[0] + dim[s]].reshape(nocc[s], nvir[s])

                # The following code is exactly equivalent to
                # Lpq_z[s] = einsum('Pjb,jb->P', Lia[s], z)
                Lpq_z[s] = Lia[s].reshape(naux, -1) @ z.reshape(-1)
                work_done += naux * nvir[s] * nocc[s]

            for s in range(nspin):
                for t in range(nspin):
                    # vz = einsum('Pia,P->ia', Lia[s], Lpq_z[t]).reshape(-1) * scale
                    vz = (Lia[s].reshape(naux, -1).T @ Lpq_z[t]).reshape(-1) * scale
                    work_done += naux * nvir[s] * nocc[s]
                    apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += vz
                    # No need to compute this for TDA
                    # if TDA is True and return_amb:
                    #     amb_prod[ivec][s * dim[0]: s * dim[0] + dim[s]] += vz

    # contraction: W becomes K
    if hybrid:
        for ivec in range(ntri):
            # continue
            # this matrix can be large
            # Lpq_z = np.zeros(shape=[nspin, naux, nmo, nmo], dtype=np.double)
            for s in range(nspin):
                z = tri_vec[ivec][s * dim[0] : s * dim[0] + dim[s]].reshape(nocc[s], nvir[s])
    
                # TODO: refact
                
                Laj_zs = einsum('Lab,jb->Laj', Laa[s], z)
                kaz = -einsum('Lji,Laj->ia', Lii[s], Laj_zs).reshape(-1) * hyb_coeff
                work_done += naux * nocc[s] * nocc[s] * nvir[s]
                if not TDA:
                    Lij_zs = einsum('Lib,jb->Lij', Lia[s], z)
                    kbz = -einsum('Lja,Lij->ia', Lia[s], Lij_zs).reshape(-1) * hyb_coeff
                    
                    apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += kaz + kbz
                    amb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += kaz - kbz
                    work_done += naux * nocc[s] * nocc[s] * nvir[s]
                else:
                    apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += kaz
                    

            
    # contraction: orbital energy difference
    for s in range(nspin):
        orb_diff = np.asarray(mo_energy[s][None, nocc[s] :] - mo_energy[s][: nocc[s], None]).reshape(-1)
        for ivec in range(ntri):
            oz = orb_diff * tri_vec[ivec][s * dim[0] : s * dim[0] + dim[s]]
            apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += oz
            if not TDA:
                amb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += oz
            work_done += 2 * oz.size

    # TDDFT fxc                 
    if ni_fn is not None:
        for ivec in range(ntri):
            vdft_z = ni_fn(tri_vec[ivec], dim, nocc, nvir)
            for s in range(nspin):
                apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += vdft_z[s] * scale

    return apb_prod, amb_prod, work_done

def _get_oscillator_strength(multi, exci, X_vec, Y_vec, mo_coeff, nocc, mol):
    """Get transition dipoles and oscillator strengths.

    Args:
        multi (char): multiplicity. "s"=singlet, "t"=triplet, "u"=unrestricted.
        exci (double array): excitation energy.
        X_vec (double ndarray): X block of eigenvector (excitation).
        Y_vec (double ndarray): Y block of eigenvector (de-excitation).
        mo_coeff (double ndarray): coefficient from AO to MO.
        nocc (int array): number of occupied orbitals.
        mol (pyscf.gto.mole.Mole): Mole object for generating dipole matrix.

    Returns:
        dipole (double ndarray): transition dipoles of all excitations.
        oscillator_strength (double array): oscillator strengths of all excitations.
    """
    nspin, nao, nmo = mo_coeff.shape
    nroot = X_vec[0].shape[0]

    dipole = np.zeros(shape=[3, nroot], dtype=np.double, order='F')
    oscillator_strength = np.zeros(shape=[nroot], dtype=np.double)

    # tddft is blind to triplet oscillator strength
    if multi == 't':
        return dipole, oscillator_strength

    with mol.with_common_orig((0, 0, 0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)

    # Transform AO dipole integrals to MO basis
    mo_dip = [mo_coeff[s][:, : nocc[s]].T @ ao_dip @ mo_coeff[s][:, nocc[s] :] for s in range(nspin)]

    for j in range(nroot):
        for s in range(nspin):
            dipole[:, j] += np.einsum('ia,xia->x', X_vec[s][j], mo_dip[s]) + np.einsum(
                'ia,xia->x', Y_vec[s][j], mo_dip[s]
            )

    if nspin == 1:
        dipole *= np.sqrt(2)

    oscillator_strength = (2 / 3) * exci * np.sum(dipole**2, axis=0)

    return dipole, oscillator_strength


def _get_spin_square(nocc, X_vec, Y_vec, mo_coeff, ovlp):
    """Get <S2> expectation value.

    Args:
        nocc (int array): number of occupied orbitals.
        X_vec (double ndarray): X block of eigenvector (excitation).
        Y_vec (double ndarray): Y block of eigenvector (de-excitation).
        mo_coeff (double ndarray): coefficient from AO to MO.
        ovlp (double ndarray): overlap matrix.

    Returns:
        s2 (double array): <S2> expectation value of excitations.
    """
    nroot = X_vec[0].shape[0]
    ab_ovlp = mo_coeff[0].T @ ovlp @ mo_coeff[1]
    s2 = np.zeros(shape=[nroot], dtype=np.double)
    s2[:] = nocc[0] - (nocc[0] - nocc[1]) / 2.0 + ((nocc[0] - nocc[1]) / 2.0) ** 2
    for iroot in range(nroot):
        # alpha excitation ket
        # a alpha and j beta exchange: alpha excitation bra
        s2[iroot] -= einsum(
            'ia,ib,aj,bj->',
            X_vec[0][iroot] + Y_vec[0][iroot],
            X_vec[0][iroot] - Y_vec[0][iroot],
            ab_ovlp[nocc[0] :, : nocc[1]],
            ab_ovlp[nocc[0] :, : nocc[1]],
        )
        # a alpha and j beta exchange: beta excitation bra
        s2[iroot] -= einsum(
            'ia,jb,ij,ab->',
            X_vec[0][iroot] + Y_vec[0][iroot],
            X_vec[1][iroot] - Y_vec[1][iroot],
            ab_ovlp[: nocc[0], : nocc[1]],
            ab_ovlp[nocc[0] :, nocc[1] :],
        )
        # i alpha and j beta exchange: same alpha excitation bra
        s2[iroot] -= einsum(
            'ia,ia,jk->',
            X_vec[0][iroot] + Y_vec[0][iroot],
            X_vec[0][iroot] - Y_vec[0][iroot],
            ab_ovlp[: nocc[0], : nocc[1]] ** 2,
        )
        s2[iroot] += einsum(
            'ia,ia,ik->',
            X_vec[0][iroot] + Y_vec[0][iroot],
            X_vec[0][iroot] - Y_vec[0][iroot],
            ab_ovlp[: nocc[0], : nocc[1]] ** 2,
        )
        # beta excitation ket
        # i alpha and b beta exchange: beta excitation bra
        s2[iroot] -= einsum(
            'ia,ib,ja,jb->',
            X_vec[1][iroot] + Y_vec[1][iroot],
            X_vec[1][iroot] - Y_vec[1][iroot],
            ab_ovlp[: nocc[0], nocc[1] :],
            ab_ovlp[: nocc[0], nocc[1] :],
        )
        # i alpha and b beta exchange: alpha excitation bra
        s2[iroot] -= einsum(
            'ia,jb,ji,ba->',
            X_vec[1][iroot] + Y_vec[1][iroot],
            X_vec[0][iroot] - Y_vec[0][iroot],
            ab_ovlp[: nocc[0], : nocc[1]],
            ab_ovlp[nocc[0] :, nocc[1] :],
        )
        # i alpha and j beta exchange: same alpha excitation bra
        s2[iroot] -= einsum(
            'ia,ia,jk->',
            X_vec[1][iroot] + Y_vec[1][iroot],
            X_vec[1][iroot] - Y_vec[1][iroot],
            ab_ovlp[: nocc[0], : nocc[1]] ** 2,
        )
        s2[iroot] += einsum(
            'ia,ia,ji->',
            X_vec[1][iroot] + Y_vec[1][iroot],
            X_vec[1][iroot] - Y_vec[1][iroot],
            ab_ovlp[: nocc[0], : nocc[1]] ** 2,
        )

    return s2


class tddft(lib.StreamObject):
    def __init__(
        self,
        # initialize with a GW object
        mf=None,
        # initialize with nocc, mo_energy and Lpq
        nocc=None,
        mo_energy=None,
        Lpq=None,
        verbose=5,
        # options
        TDA=False,
        nroot=10,
        max_vec=None,
        max_iter=100,
        max_expand=None,
        init_ntri=None,
        residue_thresh=1.0e-8,
        delete_lpq=False,
        chkfile=None,
        chk_every=10,
        restart_max_size=None
    ):
        """Initialize tddft object.
        The tddft object can be initialized by restricted or unrestricted GW object.

        Args:
            gw (fcdmft.gw.mol, optional): GW object.
            nocc (int/int array, optional): number of occupied orbitals.
            mo_energy(double ndarray, optional): quasiparticle energy.
            Lpq (double ndarray, optional): three-center density-fitting matrix in MO.
            verbose (int, optional): print level.
            TDA (bool, optional): use TDA approximation to ignore B matrix. Defaults to False.
            nroot (int, optional): the number of desired roots. Defaults to 10.
            max_vec (int, optional): max allowed subspace size. Defaults to 200.
            max_iter (int, optional): max Davidson iteration. Defaults to 100.
            max_expand(int, optional): max number of trial vectors to expand. Defaults to nroot.
            init_ntri (int, optional): initial number of trial vectors. Default is 4 * nroot.
            residue_thresh (double, optional): threshold if the residue needs to be added as a new trial vector.
            Defaults to 1.0e-8.
            delete_lpq (bool, optional): delete Lpq during the calculation to save memory. Defaults to False.
            chkfile (str, optional): checkpoint file. Defaults to None.
            chk_every (int, optional): checkpoint frequency. Defaults to 10 (trial vectors).
        """
        # initialize matrix
        if mf is not None:
            self.nspin = 1 if np.asarray(mf.mo_energy).ndim == 1 else 2
            self.verbose = mf.mol.verbose
            self.mol = mf.mol
            self.mf = mf
            self.nocc = [mf.mol.nelectron // 2] if self.nspin == 1 else mf.nelec
            
            self.mo_energy = np.asarray(mf.mo_energy)
            if self.mo_energy.ndim == 1:
                self.mo_energy = self.mo_energy[np.newaxis, ...]
            self.mo_coeff = mf.mo_coeff
            if self.mo_coeff.ndim == 2:
                self.mo_coeff = self.mo_coeff[np.newaxis, ...]
                self.mo_occ = self.mf.mo_occ[np.newaxis, ...]
            self.nmo = self.mo_energy.shape[-1]
            # initialize density-fitting matrix
            if self.nspin == 2:
                mf.nmo = [self.nmo, self.nmo]
            self.Lpq = Lpq
            if Lpq is None:
                from fcdmft.df.incore import cholesky_eri_mo
                self.Lpq = np.stack([cholesky_eri_mo(self.mol, (self.mo_coeff[s], self.mo_coeff[s]), auxbasis = f'{self.mol.basis}-ri') for s in range(self.nspin)])
                self.Lpq = self.Lpq.reshape((self.Lpq.shape[0], self.Lpq.shape[1], self.nmo, self.nmo))
        elif mf is None and (nocc is not None and mo_energy is not None and Lpq is not None):
            self.verbose = verbose
            if np.asarray(mo_energy).ndim == 1:
                self.nspin = 1
                self.nocc = np.asarray(nocc)[np.newaxis, ...]
                self.mo_energy = np.asarray(mo_energy)[np.newaxis, ...]
                self.Lpq = np.asarray(Lpq)[np.newaxis, ...]
                self.nmo = mo_energy.shape[0]
            else:
                self.nspin = 2
                self.nocc = np.asarray(nocc)
                self.mo_energy = np.asarray(mo_energy)
                self.Lpq = np.asarray(Lpq)
                self.nmo = [mo_energy.shape[1], mo_energy.shape[1]]
        else:
            raise NotImplementedError('tddft object is not properly initialized!')

        # options
        self.TDA = TDA  # use TDA approximation to ignore B matrix
        self.delete_lpq = delete_lpq  # delete Lpq after calculation
        self.chkfile = chkfile  # checkpoint file
        self.chk_every = chk_every  # checkpoint frequency

        # Davidson algorithm
        self.multi = None  # multiplicity
        self.nroot = nroot  # the number of desired roots
        self.trial = 'identity'  # mode to initialize trial vector
        self.nocc_sub = 50  # number of occpuied orbitals in the trial vector subspace
        self.nvir_sub = 150  # number of virtual orbitals in the trial vector subspace
        self.max_vec = 12 * nroot if max_vec is None else max_vec  # max allowed subspace size
        self.max_iter = max_iter  # max Davidson iteration
        # max number of trial vectors to expand per iteration
        self.max_expand = min(100, nroot) if max_expand is None else max_expand
        self.residue_thresh = residue_thresh  # threshold if the residue needs to be added as a new trial vector
        self.init_ntri = min(100, nroot) if init_ntri is None else init_ntri
        self.restart_max_size = restart_max_size  # max number of trial vectors to keep during a restart

        # results
        self.exci = None  # excitation energy
        self.X_vec = None  # X block of eigenvector (excitation)
        self.Y_vec = None  # Y block of eigenvector (de-excitation)

        # TDDFT part, for fxc
        if mf is None or self.mf.xc == 'hf':
            self.ni = None
            self.omega, self.alpha, self.hyb = 0, 1.0, 1.0
            self.hybrid = True
        else:
            self.ni = self.mf._numint
            self.ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
            if mf.do_nlc():
                log.warn(mf, 'NLC functional found in DFT object.  Its second '
                            'derivative is not available. Its contribution is '
                            'not included in the response function.')
            self.omega, self.alpha, self.hyb = self.ni.rsh_and_hybrid_coeff(self.mf.xc, self.mol.spin)
            self.hybrid = self.ni.libxc.is_hybrid_xc(self.mf.xc)
            self.dm0 = self.mf.make_rdm1()
            
                
        if self.omega > 0:
            raise NotImplementedError
            


    def dump_flags(self):
        log = lib.logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        nvir = [(self.nmo - self.nocc[i]) for i in range(self.nspin)]
        dim = [(self.nocc[i] * nvir[i]) for i in range(self.nspin)]
        log.info('multiplicity = %s', self.multi)
        log.info('nmo = %s', self.nmo)
        log.info('nocc = %s', self.nocc[0] if self.nspin == 1 else self.nocc)
        log.info('nvir = %s', nvir[0] if self.nspin == 1 else nvir)
        log.info('occ-vir dimension = %s', dim[0] if self.nspin == 1 else dim)
        if self.nspin == 2:
            log.info('tddft full dimension = %s', dim[0] + dim[1])
        log.info('Tamm-Dancoff approximation = %s', self.TDA)
        log.info('number of roots = %d', self.nroot)
        log.info('trial vector = %s', self.trial)
        if self.trial == 'subspace':
            log.info('subspace nocc = %d nvir = %d', self.nocc_sub, self.nvir_sub)
        log.info('max subspace size = %d', self.max_vec)
        log.info('max iteration = %s', self.max_iter)
        log.info('convergence tolerance = %s', self.residue_thresh)
        log.info('')
        return

    def check_memory(self):
        """Check memory needed for the tddft calculation."""
        nvir = [(self.nmo - self.nocc[i]) for i in range(self.nspin)]
        dim = [(self.nocc[i] * nvir[i]) for i in range(self.nspin)]
        full_dim = dim[0] + dim[1] if self.nspin == 2 else dim[0]
        naux = self.Lpq.shape[1]

        # Lpq; trial vector, A+B/A-B matrix with trial vector product
        mem = (naux * self.nmo * self.nmo + self.max_vec * full_dim * 3) * 8
        lib.logger.info(self, 'tddft needs at least %.1f GB memory.', mem / 1.0e9)

        return

    def kernel(self, multi, e_min=0.0, delta=0.0, **kwargs):
        # check spin and multiplicity
        assert isinstance(multi, str)
        multi = multi[0].lower()
        assert (self.nspin == 1 and (multi == 's' or multi == 't')) or (self.nspin == 2 and multi == 'u')
        self.multi = multi

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.check_memory()
        exci, X, Y = tddft_davidson(tddft=self, multi=multi, e_min=e_min, delta=delta, **kwargs)
        lib.logger.timer(self, 'tddft', *cput0)
        return exci, X, Y

    def full_diagonalization(self, multi):
        cput0 = (time.process_time(), time.perf_counter())
        lib.logger.info(self, '\ntddft full diagonalization: %s', multi)
        self.multi = multi

        # set nroot as full dimension for analysis
        nvir = [(self.nmo - self.nocc[i]) for i in range(self.nspin)]
        dim = [(self.nocc[i] * nvir[i]) for i in range(self.nspin)]
        self.nroot = dim[0] + dim[1] if self.nspin == 2 else dim[0]

        # A+B, A-B, X+Y, X-Y
        mem = (self.nroot * self.nroot * 4) * 8
        lib.logger.info(self, 'tddft needs at least %.1f GB memory.', mem / 1.0e9)

        self.exci, self.X_vec, self.Y_vec = tddft_full_diagonalization(
            multi=multi, nocc=self.nocc, mo_energy=self.mo_energy, Lpq=self.Lpq, TDA=self.TDA
        )
        lib.logger.timer(self, 'tddft full diagonalization', *cput0)
        return self.exci, self.X_vec, self.Y_vec

    def analyze(self, thresh=0.1, oscillator=True, s2=True, e_min=0.0):
        """Analyze excitations.

        Args:
            thresh (float, optional): threshold to print dominant component. Defaults to 0.1.
            oscillator (bool, optional): calculate oscillator strength. Defaults to True.
            s2 (bool, optional): calculate <S2> expectation value. Defaults to True.
            e_min (float, optional): minimum excitation energy to analyze. Defaults to 0.0.

        Returns:
            all_data (list): list of dictionaries containing results of analysis.
        """
        multi = self.multi
        nspin = self.nspin
        nmo = self.nmo
        nocc = self.nocc

        emin_index = np.searchsorted(self.exci, e_min, side='left')
        exci = self.exci[emin_index:]

        X_vec = [X_vec_s[emin_index:] for X_vec_s in self.X_vec]
        Y_vec = [Y_vec_s[emin_index:] for Y_vec_s in self.Y_vec]
        nvir = [(nmo - nocc[i]) for i in range(nspin)]

        if oscillator is True:
            dipole, oscillator_strength = _get_oscillator_strength(
                multi=multi, exci=exci, X_vec=X_vec, Y_vec=Y_vec, mo_coeff=self.mo_coeff, nocc=nocc, mol=self.mol
            )

        if s2 is True and nspin == 2:
            s2 = _get_spin_square(nocc=nocc, X_vec=X_vec, Y_vec=Y_vec, mo_coeff=self.mo_coeff, ovlp=self.mf.get_ovlp())

        all_data = []

        print('-' * 55)
        if multi == 's':
            print('restricted singlet tddft')
        elif multi == 't':
            print('restricted triplet tddft')
        elif multi == 'u':
            print('unrestricted tddft')
        for r in range(exci.size):
            this_datum = {
                'excited_state': r + 1,
                'excitation_energy': float(exci[r]),
                'excitation_energy_ev': float(exci[r] * HARTREE2EV),
            }
            print('-' * 55)
            print('excited state: %-d' % (r + 1))
            print('excitation energy:   %15.8f   AU   %15.8f   eV' % (exci[r], exci[r] * HARTREE2EV))
            if multi == 's':
                if oscillator is True:
                    print('spin allowed, oscillator strength:   %15.8f   AU' % oscillator_strength[r])
                    print(
                        'transition dipole: x =  %15.6f  , y =  %15.6f  , z =  %15.6f'
                        % (dipole[0][r], dipole[1][r], dipole[2][r])
                    )
                    this_datum['oscillator_strength'] = float(oscillator_strength[r])
                    this_datum['transition_dipole'] = (float(dipole[0][r]), float(dipole[1][r]), float(dipole[2][r]))
            elif multi == 't':
                if oscillator is True:
                    print('spin forbidden, oscillator strength and transition dipoles are not defined')
            elif multi == 'u':
                if s2 is True:
                    print('<S^2> =    %.6f', s2[r])
                if oscillator is True:
                    print('oscillator strength:   %15.8f   AU' % oscillator_strength[r])
                    print(
                        'transition dipole: x =  %15.6f  , y =  %15.6f  , z =  %15.6f'
                        % (dipole[0][r], dipole[1][r], dipole[2][r])
                    )
                    this_datum['s2'] = s2[r]
                    this_datum['oscillator_strength'] = float(oscillator_strength[r])
                    this_datum['transition_dipole'] = (float(dipole[0][r]), float(dipole[1][r]), float(dipole[2][r]))

            def print_component(comp, with_spin=False):
                if not with_spin:
                    print(f"{comp['i']:5} -> {comp['a']:5}, {comp['weight']:15f}, {comp['type']}")
                else:
                    print(
                        f"{comp['i']:5} -> {comp['a']:5}, spin {comp['spin']}, {comp['weight']:15f}, {comp['type']}"
                    )

            this_datum_components = []
            print('dominant component')
            if nspin == 1:
                for i in range(nocc[0]):
                    for a in range(nvir[0]):
                        if abs(X_vec[0][r][i][a]) > thresh:
                            comp = {
                                    'i': i + 1,
                                    'a': int(a + nocc[0] + 1),
                                    'spin': 0,
                                    'weight': float(X_vec[0][r][i][a]),
                                    'type': 'X',
                            }
                            this_datum_components.append(comp)
                            print_component(comp)
                        if abs(Y_vec[0][r][i][a]) > thresh:
                            comp = {
                                    'i': i + 1,
                                    'a': int(a + nocc[0] + 1),
                                    'spin': 0,
                                    'weight': float(Y_vec[0][r][i][a]),
                                    'type': 'Y',
                            }
                            this_datum_components.append(comp)
                            print_component(comp)
            else:
                for s in range(nspin):
                    spin = 'a' if s == 0 else 'b'
                    for i in range(nocc[s]):
                        for a in range(nvir[s]):
                            if abs(X_vec[s][r][i][a]) > thresh:
                                comp = {
                                        'i': i + 1,
                                        'a': int(a + nocc[s] + 1),
                                        'spin': s,
                                        'weight': float(X_vec[s][r][i][a]),
                                        'type': 'X',
                                }
                                this_datum_components.append(comp)
                                print_component(comp, with_spin=True)
                            if abs(Y_vec[s][r][i][a]) > thresh:
                                comp = {
                                        'i': i + 1,
                                        'a': int(a + nocc[s] + 1),
                                        'spin': s,
                                        'weight': float(X_vec[s][r][i][a]),
                                        'type': 'Y',
                                }
                                this_datum_components.append(comp)
                                print_component(comp, with_spin=True)
            this_datum['components'] = this_datum_components
            all_data.append(this_datum)
        return all_data

    def get_oscillator_strength(self):
        """Get transition dipoles and oscillator strengths.

        Returns:
            dipole (double ndarray): transition dipoles.
            oscillator_strength (double array): oscillator strengths.
        """
        assert self.exci is not None and self.X_vec is not None and self.Y_vec is not None
        assert self.mo_coeff is not None and self.mol is not None
        dipole, oscillator_strength = _get_oscillator_strength(
            multi=self.multi,
            exci=self.exci,
            X_vec=self.X_vec,
            Y_vec=self.Y_vec,
            mo_coeff=self.mo_coeff,
            nocc=self.nocc,
            mol=self.mol,
        )

        return dipole, oscillator_strength


if __name__ == '__main__':
    from pyscf import gto, dft, scf
    print('---MAEs in eV---')
    mol = gto.Mole()
    mol.atom = '''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161'''
    mol.basis = 'ccpvdz'
    mol.build()
    
    mf = scf.RKS(mol)
    mf.xc = 'hf'
    mf.kernel()

    mytd = tddft(mf = mf, nroot = 10, max_vec = 500, residue_thresh = 1.0e-11)
    mytd.TDA = True
    exci_new = mytd.kernel(multi = 's')[0]
    pyscf_ref = np.array([0.33869256, 0.4039096 , 0.43547223, 0.50126752, 0.55287743,
       0.67425701, 0.84642711, 0.91891412, 0.97352696, 1.01628373])

    print('TDHF+TDA+s mae vs PySCF: ',HARTREE2EV*np.mean(np.abs(exci_new-pyscf_ref)))

    mol = gto.Mole()
    mol.atom = '''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161'''
    mol.basis = 'ccpvdz'
    mol.build()
    
    mf = scf.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    mytd = tddft(mf = mf, nroot = 10, max_vec = 500, residue_thresh = 1.0e-11)
    mytd.TDA = True
    exci_new = mytd.kernel(multi = 's')[0]
    pyscf_ref = np.array([0.29364738, 0.36209534, 0.38268895, 0.45526952, 0.52756327,
       0.64107749, 0.77519937, 0.84452901, 0.89875, 0.92742015])

    print('TDDFT(PBE0)+TDA+s mae vs PySCF: ',HARTREE2EV*np.mean(np.abs(exci_new-pyscf_ref)))
    
    mol = gto.Mole()
    mol.atom = '''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161'''
    mol.basis = 'ccpvdz'
    mol.spin = 1
    mol.charge = 1
    mol.build()
    
    mf = scf.UKS(mol)
    mf.xc = 'hf'
    mf.kernel()

    mytd = tddft(mf = mf, nroot = 10, max_vec = 500, residue_thresh = 1.0e-11)
    mytd.TDA = True
    exci_new = mytd.kernel(multi = 'u')[0]
    pyscf_ref = np.array([0.09541135, 0.25669874, 0.53930442, 0.59081323, 0.60198381,
       0.60325613, 0.64665233, 0.67640226, 0.67702624, 0.69584741])
    print('UTDHF+TDA+u mae vs PySCF: ',HARTREE2EV*np.mean(np.abs(exci_new-pyscf_ref)))

    mol = gto.Mole()
    mol.atom = '''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161'''
    mol.basis = 'ccpvdz'
    mol.spin = 1
    mol.charge = 1
    mol.build()
    
    mf = scf.UKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    mytd = tddft(mf = mf, nroot = 10, max_vec = 500, residue_thresh = 1.0e-11)
    mytd.TDA = True
    exci_new = mytd.kernel(multi = 'u')[0]


    pyscf_ref = np.array([0.09546915, 0.24289117, 0.49772066, 0.53623871, 0.54371346,
       0.56107213, 0.61338201, 0.61917995, 0.6305623 , 0.68065354])
    print('UTDDFT(PBE0)+TDA+u mae vs PySCF: ',HARTREE2EV*np.mean(np.abs(exci_new-pyscf_ref)))
    
    mol = gto.Mole()
    mol.atom = '''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161'''
    mol.basis = 'ccpvdz'
    mol.build()
    
    mf = scf.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    mytd = tddft(mf = mf, nroot = 10, max_vec = 500, residue_thresh = 1.0e-11)
    mytd.TDA = False
    exci_new = mytd.kernel(multi = 's')[0]
    pyscf_ref = np.array([0.29264235, 0.36180771, 0.38054299, 0.45332977, 0.52616813,
       0.63307338, 0.77500306, 0.84265298, 0.89468566, 0.92630214])

    print('TDDFT(PBE0)+s mae vs PySCF: ',HARTREE2EV*np.mean(np.abs(exci_new-pyscf_ref)))


    mol = gto.Mole()
    mol.atom = '''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161'''
    mol.basis = 'ccpvdz'
    mol.build()
    
    mf = scf.RKS(mol)
    mf.xc = 'hf'
    mf.kernel()

    mytd = tddft(mf = mf, nroot = 10, max_vec = 500, residue_thresh = 1.0e-11)
    mytd.TDA = False
    exci_new = mytd.kernel(multi = 't')[0]
    pyscf_ref = np.array([0.29971562, 0.3739557 , 0.37695375, 0.43262397, 0.49848254,
       0.54407927, 0.82889214, 0.86905081, 0.87743233, 0.89580129])

    print('TDHF+t mae vs PySCF: ',HARTREE2EV*np.mean(np.abs(exci_new-pyscf_ref)))


    
    mol = gto.Mole()
    mol.atom = '''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161'''
    mol.basis = 'ccpvdz'
    mol.build()
    
    mf = scf.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    mytd = tddft(mf = mf, nroot = 10, max_vec = 500, residue_thresh = 1.0e-11)
    mytd.TDA = False
    exci_new = np.sort(mytd.kernel(multi = 't')[0])
    pyscf_ref = np.array([0.26551226, 0.34223127, 0.34261803, 0.41170624, 0.48096384,
       0.53925444, 0.75944308, 0.80724378, 0.81411531, 0.86072436])

    print('TDDFT(PBE0)+t mae vs PySCF: ',HARTREE2EV*np.mean(np.abs(exci_new-pyscf_ref)))





