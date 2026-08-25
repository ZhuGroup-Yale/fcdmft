'''
Periodic spin-restricted G0W0 method based on the analytic continuation scheme.
This implementation has N^4 scaling,
and is faster than GW-CD (N^4~N^5) and fully analytic GW (N^6) methods.
GW-AC is recommended for valence states only, and is inaccurate for core states.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14 053020 (2012)
'''

from functools import reduce
import h5py
import numpy as np
import os
import scipy
import time

import scipy.linalg

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.lib import einsum, logger, temporary_env
from pyscf.pbc import df, dft, scf
from pyscf.pbc.mp.kmp2 import get_frozen_mask
from pyscf.pbc.mpitools.mpi_helper import safeAllreduceInPlace

from fcdmft.ac.grids import _get_scaled_legendre_roots
from fcdmft.ac.two_pole import TwoPoleAC
from fcdmft.ac.pade import PadeAC
from fcdmft.utils.arraymath import mkslice
from fcdmft.utils import arraymath

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD


def kernel(gw):
    mf = gw._scf
    nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts

    # set frozen orbitals
    gw.set_frozen_orbs()
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz
    kptlist = gw.kptlist
    if kptlist is None:
        gw.kptlist = kptlist = range(gw.nkpts)
    mo_energy_frz = _mo_energy_frozen(gw, gw.mo_energy)
    mo_coeff_frz = _mo_frozen(gw, gw.mo_coeff)

    # v_xc
    with temporary_env(mf, verbose=0), temporary_env(mf.mol, verbose=0), temporary_env(mf.with_df, verbose=0):
        dm = mf.make_rdm1()
        v_mf_ao = mf.get_veff() - mf.get_j(dm_kpts=dm)
    v_mf = np.zeros(shape=[nkpts, nmo, nmo], dtype=np.complex128)
    for k in range(nkpts):
        v_mf[k] = reduce(np.matmul, (mo_coeff_frz[k].T.conj(), v_mf_ao[k], mo_coeff_frz[k]))
    gw.vxc = v_mf

    # v_hf from DFT/HF density
    vk = gw.get_sigma_exchange()

    # finite size correction for exchange self-energy
    if gw.fc:
        vk_corr = -2.0 / np.pi * (6.0 * np.pi**2 / gw.mol.vol / nkpts) ** (1.0 / 3.0)
        for k in range(nkpts):
            for i in range(nocc):
                vk[k][i, i] = vk[k][i, i] + vk_corr
    gw.vk = vk

    # set up Fermi level
    gw.ef = ef = get_ef(kmf=mf, mo_energy=mf.mo_energy)

    # grids for integration on imaginary axis
    gw.freqs, gw.wts = freqs, wts = _get_scaled_legendre_roots(gw.nw)

    # calculate self-energy on imaginary axis
    if gw.outcore:
        sigmaI, omega = get_sigma_outcore(
            gw, freqs, wts, ef=ef, mo_energy=mo_energy_frz, orbs=orbs_frz, kptlist=kptlist, iw_cutoff=gw.ac_iw_cutoff,
            fullsigma=gw.fullsigma,
        )
    else:
        sigmaI, omega = get_sigma(
            gw, freqs, wts, ef=ef, mo_energy=mo_energy_frz, orbs=orbs_frz, kptlist=kptlist, iw_cutoff=gw.ac_iw_cutoff,
            fullsigma=gw.fullsigma,
        )

    # analytic continuation
    if gw.ac == 'twopole':
        acobj = TwoPoleAC(list(range(nmo)), nocc)
    elif gw.ac == 'pade':
        acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
    elif gw.ac == 'pes':
        raise NotImplementedError
    else:
        raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))

    acobj.ac_fit(sigmaI, omega, axis=-1)
    acobj.coeff = comm.bcast(acobj.coeff, root=0)

    if gw.fullsigma:
        diag_acobj = acobj.diagonal(axis1=1, axis2=2)
    else:
        diag_acobj = acobj

    mo_energy = np.zeros_like(mf.mo_energy)
    for ik, k in enumerate(kptlist):
        for ip, p in enumerate(orbs_frz):
            if gw.qpe_linearized:
                # linearized G0W0
                de = 1e-6
                ep = mf.mo_energy[k][orbs[ip]]
                sigmaR = diag_acobj[ik, ip].ac_eval(ep).real
                dsigma = diag_acobj[ik, ip].ac_eval(ep + de).real - sigmaR.real
                zn = 1.0 / (1.0 - dsigma / de)
                if gw.qpe_linearized_range is not None:
                    zn = 1.0 if zn < gw.qpe_linearized_range[0] or zn > gw.qpe_linearized_range[1] else zn
                mo_energy[k, orbs[ip]] = ep + zn * (sigmaR + vk[k, p, p] - v_mf[k, p, p]).real
            else:
                # self-consistently solve QP equation
                def quasiparticle(omega):
                    sigmaR = diag_acobj[ik, ip].ac_eval(omega)
                    return omega - mf.mo_energy[k][orbs[ip]] - (sigmaR + vk[k, p, p] - v_mf[k, p, p]).real

                try:
                    mo_energy[k, orbs[ip]] = scipy.optimize.newton(
                        quasiparticle, mf.mo_energy[k][orbs[ip]], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter
                    )
                except RuntimeError:
                    if rank == 0:
                        logger.warn(gw, 'QPE for k=%d orbital=%d not converged!', k, orbs[ip])

    # save GW results
    gw.mo_energy = mo_energy
    gw.acobj = acobj

    if rank == 0 and gw.verbose >= logger.DEBUG:
        with np.printoptions(threshold=len(mf.mo_energy[0])):
            for k in range(nkpts):
                logger.debug(gw, '  GW mo_energy @ k%d =\n%s', k, mo_energy[k])
        logger.warn(gw, 'GW QP energies may not be sorted from min to max')

    if rank == 0 and gw.writefile > 0:
        fn = 'vxc.h5'
        feri = h5py.File(fn, 'w')
        feri['vk'] = np.asarray(vk)
        feri['v_mf'] = np.asarray(v_mf)
        feri.close()

        fn = 'sigma_imag.h5'
        feri = h5py.File(fn, 'w')
        feri['sigmaI'] = np.asarray(sigmaI)
        feri['omega'] = np.asarray(omega)
        if gw.sigmaI is not None:
            feri['sigmaI_full'] = np.asarray(gw.sigmaI)
        feri.close()

        acobj.save('ac_coeff.h5')

    return


def get_rho_response(omega, mo_energy, Lia, kidx):
    """Get Pi=PV.
    P is density-density response function.
    V is two-electron integral.
    See equation 24 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    omega : double
        real position of imaginary frequency
    mo_energy : double 2d array
        orbital energy
    Lia : complex 4d ndarray
        occupied-virtual block of three-center density-fitting matrix in MO
    kidx : list
        momentum-conserved k-point list kj=kidx[ki]

    Returns
    -------
    Pi : complex ndarray
        Pi in auxiliary basis at freq iw
    """
    nkpts, naux, nocc, nvir = Lia.shape

    # Compute Pi for kL
    Pi = np.zeros(shape=[naux, naux], dtype=np.complex128)
    for i in range(nkpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        eia = mo_energy[i, :nocc, None] - mo_energy[a, None, nocc:]
        Lia_i = Lia[i]
        try:
            import numexpr as ne

            eia = ne.evaluate('eia / (omega**2 + eia**2)')
            Pia = ne.evaluate('Lia_i * eia')
        except ImportError:
            eia = eia / (omega**2 + eia**2)
            Pia = Lia_i * eia
        # Response from both spin-up and spin-down density
        # Pi += (4./nkpts) * einsum('Pia,Qia->PQ', Pia, Lov.conj())
        scipy.linalg.blas.zgemm(
            alpha=4.0 / nkpts,
            a=Lia_i.reshape(naux, nocc * nvir).T,
            b=Pia.reshape(naux, nocc * nvir).T,
            c=Pi.T,
            trans_a=2,
            trans_b=0,
            beta=1.0,
            overwrite_c=True,
        )
        Pia = Lia_i = None

    return Pi


def get_rho_response_metal(omega, mo_energy, mo_occ, Lpq, kidx):
    """Get Pi=PV for metallic systems.
    P is density-density response function.
    V is two-electron integral.
    See equation 24 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    omega : double
        real position of imaginary frequency
    mo_energy : double ndarray
        orbital energy
    mo_occ : double ndarray
        occupation number
    Lpq : complex ndarray
        three-center density-fitting matrix in MO
    kidx : list
        momentum-conserved k-point list kj=kidx[ki]

    Returns
    -------
    Pi : complex ndarray
        Pi in auxiliary basis at freq iw
    """
    nkpts, naux, nmo, _ = Lpq.shape
    mo_occ = [x / 2.0 for x in mo_occ]

    # Compute Pi for kL
    Pi = np.zeros(shape=[naux, naux], dtype=np.complex128)
    for i in range(nkpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        eia = mo_energy[i, :, None] - mo_energy[a, None, :]
        fia = mo_occ[i][:, None] - mo_occ[a][None, :]
        Lia = np.ascontiguousarray(Lpq[i])
        try:
            import numexpr as ne

            eia = ne.evaluate('eia * fia / (omega**2 + eia**2)')
            Pia = ne.evaluate('Lia * eia')
        except ImportError:
            eia = eia * fia / (omega**2 + eia**2)
            Pia = Lia * eia
        # Response from both spin-up and spin-down density
        # both ia and ai are included, this gives a factor of 2.0
        # Pi += (2./nkpts) * einsum('Pia,Qia->PQ', Pia, Lpq_i.conj())
        scipy.linalg.blas.zgemm(
            alpha=2.0 / nkpts,
            a=Lia.reshape(naux, nmo * nmo).T,
            b=Pia.reshape(naux, nmo * nmo).T,
            c=Pi.T,
            trans_a=2,
            trans_b=0,
            beta=1.0,
            overwrite_c=True,
        )
        Pia = Lia = None

    return Pi


def get_rho_response_head(omega, mo_energy, qij):
    """Compute head (G=0, G'=0) density response function in auxiliary basis at freq iw.
    equation 48 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    omega : double
        frequency point
    mo_energy : double ndarray
        orbital energy
    qij : complex ndarray
        pair density matrix defined as equation 51 in 10.1021/acs.jctc.0c00704

    Returns
    -------
    Pi_00 : complex
        head response function
    """
    nkpts, nocc = qij.shape[:2]

    Pi_00 = 0j
    for k in range(nkpts):
        eia = mo_energy[k, :nocc, None] - mo_energy[k, None, nocc:]
        try:
            import numexpr as ne

            qij_k = qij[k]
            eia = ne.evaluate('eia / (omega**2 + eia**2)')
            sum_ia = ne.evaluate('sum(eia * conj(qij_k) * qij_k)')
            Pi_00 += ne.evaluate('4.0 / nkpts * sum_ia')
        except ImportError:
            eia = eia / (omega**2 + eia**2)
            Pi_00 += 4.0 / nkpts * einsum('ia,ia->', eia, qij[k].conj() * qij[k])
    return Pi_00


def get_rho_response_wing(omega, mo_energy, Lia, qij):
    """Compute wing (G=P, G'=0) density response function in auxiliary basis at freq iw.
    equation 48 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    omega : double
        frequency point
    mo_energy : double 2d array
        orbital energy
    Lia : complex 4d array
        occupied-virtual block of three-center density fitting matrix in MO
    qij : complex ndarray
        pair density matrix defined as equation 51 in 10.1021/acs.jctc.0c00704

    Returns
    -------
    Pi : complex ndarray
        wing response function
    """
    nkpts, naux, nocc, nvir = Lia.shape

    Pi = np.zeros(shape=[naux], dtype=np.complex128)
    for k in range(nkpts):
        eia = mo_energy[k, :nocc, None] - mo_energy[k, None, nocc:]
        try:
            import numexpr as ne

            qij_k = qij[k]
            eia = ne.evaluate('eia / (omega**2 + eia**2)')
            eia_q = ne.evaluate('eia * conj(qij_k)')
        except ImportError:
            eia = eia / (omega**2 + eia**2)
            eia_q = eia * qij[k].conj()
        Pi += 4.0 / nkpts * np.matmul(Lia[k].reshape(naux, nocc * nvir), eia_q.reshape(nocc * nvir))
    return Pi


def get_qij(gw, q, mo_energy, mo_coeff, uniform_grids=False):
    """Compute pair density matrix in the long-wavelength limit through kp perturbation theory
    qij = 1/Omega * |< psi_{ik} | e^{iqr} | psi_{ak-q} >|^2
    equation 51 in 10.1021/acs.jctc.0c00704
    Ref: Phys. Rev. B 83, 245122 (2011)

    Parameters
    ----------
    gw : KRGWAC
        gw object, provides attributes: nocc, nmo, kpts, mol
    q : double
        q grid
    mo_energy : double ndarray
        orbital energy
    mo_coeff : complex ndarray
        coefficient from AO to MO
    uniform_grids : bool, optional
        use uniform grids, by default False

    Returns
    -------
    qij : complex ndarray
        pair density matrix in the long-wavelength limit
    """
    nocc = gw.nocc
    nmo = gw.nmo
    nvir = nmo - nocc
    kpts = gw.kpts
    nkpts = len(kpts)
    cell = gw.mol

    if uniform_grids:
        with temporary_env(cell, verbose=0):
            mydf = df.FFTDF(cell, kpts=kpts)
            coords = cell.gen_uniform_grids(mydf.mesh)
    else:
        with temporary_env(cell, verbose=0):
            coords, weights = dft.gen_grid.get_becke_grids(cell, level=4)
    ngrid = len(coords)

    qij = np.zeros(shape=[nkpts, nocc, nvir], dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        ao_p = dft.numint.eval_ao(cell, coords, kpt=kpti, deriv=1)
        ao = ao_p[0]
        ao_grad = ao_p[1:4]
        if uniform_grids:
            ao_ao_grad = einsum('mg,xgn->xmn', ao.T.conj(), ao_grad) * cell.vol / ngrid
        else:
            ao_ao_grad = einsum('g,mg,xgn->xmn', weights, ao.T.conj(), ao_grad)
        q_ao_ao_grad = -1j * einsum('x,xmn->mn', q, ao_ao_grad)
        q_mo_mo_grad = reduce(np.matmul, (mo_coeff[i][:, :nocc].T.conj(), q_ao_ao_grad, mo_coeff[i][:, nocc:]))
        enm = 1.0 / (mo_energy[i][nocc:, None] - mo_energy[i][None, :nocc])
        dens = enm.T * q_mo_mo_grad
        qij[i] = dens / np.sqrt(cell.vol)

    return qij


def get_sigma(
    gw, freqs, wts, ef, mo_energy, orbs=None, kptlist=None, mo_coeff=None, mo_occ=None, iw_cutoff=None, fullsigma=False
):
    """Get GW self-energy.
    See equation 27 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    gw : KRGWAC
        GW objects, provides attributes: _scf, mol, frozen, nmo, nocc, kpts, nkpts, mo_coeff, mo_occ, fc, fc_grid, with_df
    freqs : double array
        position of imaginary frequency
    wts : double array
        weight of frequency points
    ef : double
        Fermi level
    mo_energy : double ndarray
        non-frozen orbital energy
    orbs : list, optional
        orbital index in non-frozen nmo to calculate self-energy, by default None
    kptlist : list, optional
        k-point index to calculate self-energy, by default None
    mo_coeff : complex ndarray, optional
        coefficient from AO to non-frozen MO, by default None
    mo_occ : double ndarray, optional
        non-frozen occupation number, by default None
    iw_cutoff : complex, optional
        imaginary grid cutoff for fitting, by default None
    fullsigma : bool, optional
        calculate off-diagonal elements, by default False

    Returns
    -------
    sigma: complex ndarray
        self-energy on the imaginary axis
    omega: complex ndarray
        imaginary frequency grids of self-energy
    """
    nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts

    if orbs is None:
        orbs = list(range(nmo))
    if kptlist is None:
        kptlist = list(range(nkpts))
    norbs = len(orbs)
    nklist = len(kptlist)
    nw = len(freqs)

    if mo_coeff is None:
        mo_coeff = _mo_frozen(gw, gw.mo_coeff)
    if mo_occ is None:
        mo_occ = _mo_occ_frozen(gw, gw.mo_occ)
    nao = mo_coeff[0].shape[0]

    # possible kpts shift center
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    # Integration on numerical grids
    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = sum(iw < iw_cutoff for iw in freqs) + 1
    else:
        nw_sigma = nw + 1

    omega = np.zeros(shape=[nw_sigma], dtype=np.complex128)
    omega[1:] = 1j * freqs[: (nw_sigma - 1)] + ef
    emo = omega[None, None, :] - mo_energy[:, :, None]

    if fullsigma is False:
        sigma = np.zeros(shape=[nklist, norbs, nw_sigma], dtype=np.complex128)
    else:
        sigma = np.zeros(shape=[nklist, norbs, norbs, nw_sigma], dtype=np.complex128)
    if gw.fc:
        # Set up q mesh for q->0 finite size correction
        if not gw.fc_grid:
            q_pts = np.array([1e-3, 0, 0], dtype=np.double).reshape(1, 3)
        else:
            Nq = 3
            q_pts = np.zeros(shape=[Nq**3 - 1, 3], dtype=np.double)
            for i in range(Nq):
                for j in range(Nq):
                    for k in range(Nq):
                        if i == 0 and j == 0 and k == 0:
                            continue
                        else:
                            q_pts[i * Nq**2 + j * Nq + k - 1, 0] = k * 5e-4
                            q_pts[i * Nq**2 + j * Nq + k - 1, 1] = j * 5e-4
                            q_pts[i * Nq**2 + j * Nq + k - 1, 2] = i * 5e-4
        nq_pts = len(q_pts)
        q_abs = gw.mol.get_abs_kpts(q_pts)

        # Get qij = 1/sqrt(Omega) * < psi_{ik} | e^{iqr} | psi_{ak-q} > at q: (nkpts, nocc, nvir)
        qij = np.zeros(shape=[nq_pts, nkpts, nocc, nmo - nocc], dtype=np.complex128)

        if not gw.fc_grid:
            for k in range(nq_pts):
                qij[k] = get_qij(gw, q_abs[k], mo_energy, mo_coeff)
        else:
            segsize = nq_pts // size
            if rank >= size - (nq_pts - segsize * size):
                start = rank * segsize + rank - (size - (nq_pts - segsize * size))
                stop = min(nq_pts, start + segsize + 1)
            else:
                start = rank * segsize
                stop = min(nq_pts, start + segsize)
            for k in range(start, stop):
                qij[k] = get_qij(gw, q_abs[k], mo_energy, mo_coeff)
            comm.Barrier()
            qij_gather = comm.gather(qij)
            if rank == 0:
                for i in range(1, size):
                    qij += qij_gather[i]
            comm.Barrier()
            qij = comm.bcast(qij, root=0)

    segsize = nkpts // size
    if rank >= size - (nkpts - segsize * size):
        start = rank * segsize + rank - (size - (nkpts - segsize * size))
        stop = min(nkpts, start + segsize + 1)
    else:
        start = rank * segsize
        stop = min(nkpts, start + segsize)

    cderiarr = gw.with_df.cderi_array()
    for kL in range(start, stop):
        # Lij: (ki, L, i, j) for looping every kL
        Lij = []
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros(shape=[nkpts], dtype=np.int64)
        kidx_r = np.zeros(shape=[nkpts], dtype=np.int64)
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i
                    logger.debug(gw, 'Read Lpq (kL: %s / %s, ki: %s, kj: %s @ Rank %d)' % (kL + 1, nkpts, i, j, rank))
                    Lij_out = None
                    # Read (L|pq) and ao2mo transform to (L|ij)
                    # support unequal naux on different k points
                    Lpq = cderiarr.load(kpti, kptj)
                    if Lpq.shape[-1] == (nao*(nao+1))//2:
                        Lpq = lib.unpack_tril(Lpq).reshape(-1,nao**2)
                    else:
                        Lpq = Lpq.reshape(-1,nao**2)
                    Lpq = Lpq.astype(np.complex128)

                    moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]
                    Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao=[], ao_loc=None, out=Lij_out)
                    Lij.append(Lij_out.reshape(-1, nmo, nmo))
        Lij = np.ascontiguousarray(Lij)
        naux = Lij.shape[1]

        if hasattr(gw._scf, 'sigma') is False:
            Lia = np.ascontiguousarray(Lij[:, :, :nocc, nocc:])

        # allocate intermediates
        naux_ones = np.ones(shape=[1, naux], dtype=np.complex128)
        mnQ = np.zeros(shape=[nmo * norbs, naux], dtype=np.complex128)
        if fullsigma is False:
            Qmn = np.zeros(shape=[naux, nmo * norbs], dtype=np.complex128)
            Wmn = np.zeros(shape=[nmo, norbs], dtype=np.complex128)
        else:
            Wmn = np.zeros(shape=[nmo, norbs, norbs], dtype=np.complex128)
            Lij_kmQn = np.ascontiguousarray(Lij.transpose(0, 2, 1, 3))

        for w in range(nw):
            if hasattr(gw._scf, 'sigma'):
                Pi = get_rho_response_metal(freqs[w], mo_energy, mo_occ, Lij, kidx)
            else:
                Pi = get_rho_response(freqs[w], mo_energy, Lia, kidx)
            Pi_inv = np.linalg.inv(np.eye(naux) - Pi)

            if gw.fc and kL == 0:
                eps_inv_00 = 0j
                eps_inv_P0 = np.zeros(shape=[naux], dtype=np.complex128)
                for iq in range(nq_pts):
                    # head dielectric matrix eps_00, equation 47 in 10.1021/acs.jctc.0c00704
                    Pi_00 = get_rho_response_head(freqs[w], mo_energy, qij[iq])
                    eps_00 = 1.0 - 4.0 * np.pi / np.linalg.norm(q_abs[iq]) ** 2.0 * Pi_00

                    # wings dielectric matrix eps_P0, equation 48 in 10.1021/acs.jctc.0c00704
                    Pi_P0 = get_rho_response_wing(freqs[w], mo_energy, Lia, qij[iq])
                    eps_P0 = -np.sqrt(4.0 * np.pi) / np.linalg.norm(q_abs[iq]) * Pi_P0

                    # inverse dielectric matrix
                    # equation 53 in 10.1021/acs.jctc.0c00704
                    eps_inv_00 += 1.0 / nq_pts * 1.0 / (eps_00 - reduce(np.matmul, (eps_P0.conj(), Pi_inv, eps_P0)))
                    # equation 54 in 10.1021/acs.jctc.0c00704
                    eps_inv_P0 += 1.0 / nq_pts * (-eps_inv_00) * np.matmul(Pi_inv, eps_P0)

                # head correction, equation 43 in 10.1021/acs.jctc.0c00704
                Del_00 = 2.0 / np.pi * (6.0 * np.pi**2 / gw.mol.vol / nkpts) ** (1.0 / 3.0) * (eps_inv_00 - 1.0)

            Pi_inv -= np.eye(naux)
            try:
                import numexpr as ne

                wts_w = wts[w]
                freqs_w = freqs[w]
                g0 = ne.evaluate('wts_w * emo / (emo**2 + freqs_w ** 2)')
            except ImportError:
                g0 = wts[w] * emo / (emo**2 + freqs[w] ** 2)
            for k in range(nklist):
                kn = kptlist[k]
                # Find km that conserves with kn and kL (-km+kn+kL=G)
                km = kidx_r[kn]


                if len(orbs) == nmo:
                    l_slice = np.ascontiguousarray(Lij[km].reshape(naux, -1))
                    if fullsigma:
                        l_slice_mQn = np.ascontiguousarray(Lij_kmQn[km])
                else:
                    l_slice = np.ascontiguousarray(Lij[km, :, :, mkslice(orbs)].reshape(naux, -1))
                    if fullsigma:
                        l_slice_mQn = np.ascontiguousarray(Lij_kmQn[km, :, :, mkslice(orbs)])

                # Qmn = einsum('Pmn,PQ->Qmn', Lij[km][:, :, orbs].conj(), Pi_inv)
                scipy.linalg.blas.zgemm(alpha=1.0, a=Pi_inv.T, b=l_slice.T, c=mnQ.T, overwrite_c=1, trans_b=2)

                if fullsigma is False:
                    # Wmn = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn,Lij[km][:,:,orbs])
                    try:
                        import numexpr as ne

                        Qmn[:] = mnQ.T
                        ne.evaluate('Qmn * l_slice', out=Qmn)
                    except ImportError:
                        Qmn[:] = mnQ.T * l_slice
                    np.matmul(naux_ones, Qmn, out=Wmn.reshape(1, nmo * norbs))
                    arraymath.array_scale(Wmn, 1.0 / nkpts / np.pi)

                    # sigma[k] += -einsum('mn,mw->nw',Wmn,g0[km]) / np.pi
                    # 1 / np.pi is included in Wmn above
                    sigma[k] -= np.matmul(Wmn.reshape(nmo, norbs).T, g0[km])
                else:
                    # for orbm in range(nmo):
                    #     Wmn[orbm] = 1./nkpts * np.dot(Qmn[:,orbm,:].transpose(),Lij[km][:,orbm,orbs])
                    #for m in range(nmo):
                    #    np.matmul(Qmn[:, m, :].T, np.ascontiguousarray(Lij[km, :, m, mkslice(orbs)]), out=Wmn[m])
                    np.matmul(mnQ.reshape(nmo, norbs, naux), l_slice_mQn, out=Wmn)
                    arraymath.array_scale(Wmn, 1.0 / nkpts / np.pi)

                    #Wmn = Wmn.reshape(nmo, norbs * norbs).T
                    # sigma[k] += -einsum('mnl,mw->nlw',Wmn,g0[km])/np.pi
                    # 1 / np.pi is included in Wmn above
                    sigma[k] -= np.matmul(Wmn.reshape(nmo, norbs * norbs).T, g0[km]).reshape(norbs, norbs, nw_sigma)

                if gw.fc and kL == 0:
                    # Find km that conserves with kn and kL (-km+kn+kL=G)
                    assert kn == km
                    if fullsigma is False:
                        # head correction
                        sigma[k] += -Del_00 * g0[kn][orbs] / np.pi

                        # wing correction
                        Wn_P0 = einsum('Pnn,P->n', Lij[kn], eps_inv_P0)
                        Wn_P0 = Wn_P0[orbs].real * 2.0
                        Del_P0 = np.sqrt(gw.mol.vol/4/np.pi**3) * (6*np.pi**2/gw.mol.vol/nkpts) ** (2/3) * Wn_P0
                        sigma[k] += -einsum('n,nw->nw', Del_P0, g0[kn][orbs]) / np.pi
                    else:
                        # head correction
                        tmp = -Del_00 * g0[kn][orbs] / np.pi
                        sigma[k, np.arange(norbs), np.arange(norbs), :] += tmp

                        # wing correction
                        Wn_P0 = einsum('Pnn,P->n', Lij[kn], eps_inv_P0)
                        Wn_P0 = Wn_P0[orbs].real * 2.0
                        Del_P0 = np.sqrt(gw.mol.vol/4/np.pi**3) * (6*np.pi**2/gw.mol.vol/nkpts) ** (2/3) * Wn_P0
                        tmp = -einsum('n,nw->nw', Del_P0, g0[kn][orbs]) / np.pi
                        sigma[k, np.arange(norbs), np.arange(norbs), :] += tmp

    safeAllreduceInPlace(comm, sigma)

    if gw.rdm:
        gw.sigmaI = sigma

    return sigma, omega


def get_sigma_outcore(
    gw, freqs, wts, ef, mo_energy, orbs=None, kptlist=None, mo_coeff=None, mo_occ=None, iw_cutoff=None, fullsigma=False
):
    """Low-memory routine to get GW self-energy.
    See equation 27 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    gw : KRGWAC
        GW objects, provides attributes: _scf, mol, frozen, nmo, nocc, kpts, nkpts, mo_coeff, mo_occ, fc, fc_grid, with_df
    freqs : double array
        position of imaginary frequency
    wts : double array
        weight of frequency points
    ef : double
        Fermi level
    mo_energy : double ndarray
        non-frozen orbital energy
    orbs : list, optional
        orbital index in non-frozen nmo to calculate self-energy, by default None
    kptlist : list, optional
        k-point index to calculate self-energy, by default None
    mo_coeff : complex ndarray, optional
        coefficient from AO to non-frozen MO, by default None
    mo_occ : double ndarray, optional
        non-frozen occupation number, by default None
    iw_cutoff : complex, optional
        imaginary grid cutoff for fitting, by default None
    fullsigma : bool, optional
        calculate off-diagonal elements, by default False

    Returns
    -------
    sigma: complex ndarray
        self-energy on the imaginary axis
    omega: complex ndarray
        imaginary frequency grids of self-energy
    """
    assert gw.fc is False, "finite-size correction is not implemented in get_sigma_outcore"
    nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts

    if orbs is None:
        orbs = list(range(nmo))
    if kptlist is None:
        kptlist = list(range(nkpts))
    norbs = len(orbs)
    nklist = len(kptlist)
    nw = len(freqs)

    if mo_coeff is None:
        mo_coeff = _mo_frozen(gw, gw.mo_coeff)
    if mo_occ is None:
        mo_occ = _mo_occ_frozen(gw, gw.mo_occ)
    nao = mo_coeff[0].shape[0]

    # possible kpts shift center
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    # Integration on numerical grids
    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = sum(iw < iw_cutoff for iw in freqs) + 1
    else:
        nw_sigma = nw + 1

    omega = np.zeros(shape=[nw_sigma], dtype=np.complex128)
    omega[1:] = 1j * freqs[: (nw_sigma - 1)] + ef
    emo = omega[None, None, :] - mo_energy[:, :, None]

    if fullsigma is False:
        sigma = np.zeros(shape=[nklist, norbs, nw_sigma], dtype=np.complex128)
    else:
        sigma = np.zeros(shape=[nklist, norbs, norbs, nw_sigma], dtype=np.complex128)

    segsize = nkpts // size
    if rank >= size - (nkpts - segsize * size):
        start = rank * segsize + rank - (size - (nkpts - segsize * size))
        stop = min(nkpts, start + segsize + 1)
    else:
        start = rank * segsize
        stop = min(nkpts, start + segsize)

    cput0 = (time.process_time(), time.perf_counter())
    cderiarr = gw.with_df.cderi_array()
    for kL in range(start, stop):
        cput3 = (time.process_time(), time.perf_counter())
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros(shape=[nkpts], dtype=np.int64)
        kidx_r = np.zeros(shape=[nkpts], dtype=np.int64)

        for i in range(nkpts):
            for j in range(nkpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i

        # TODO: more efficient way to find naux without loading the whole array
        Lpq_ao = cderiarr.load(kpts[0], kpts[kidx[0]])
        assert len(Lpq_ao.shape) == 2
        naux = Lpq_ao.shape[0]

        Pi = np.zeros(shape=[nw, naux, naux], dtype=np.complex128)
        cput1 = (time.process_time(), time.perf_counter())
        for i in range(nkpts):
            a = kidx[i]
            logger.debug(gw, 'Pi (kL: %s / %s, ki: %s, kj: %s @ Rank %d)' % (kL + 1, nkpts, a, kidx_r[a], rank))
            Lpq_ao = cderiarr.load(kpts[i], kpts[a])
            if Lpq_ao.shape[-1] == (nao * (nao + 1)) // 2:
                Lpq_ao = lib.unpack_tril(Lpq_ao).reshape(-1, nao**2)
            else:
                Lpq_ao = Lpq_ao.reshape(-1, nao**2)
            Lpq_ao = Lpq_ao.astype(np.complex128)

            moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[a])[2:]
            Lpq = None
            Lpq = _ao2mo.r_e2(Lpq_ao, moij, ijslice, tao=[], ao_loc=None, out=Lpq)
            del Lpq_ao
            Lpq = np.ascontiguousarray(Lpq.reshape(-1, nmo, nmo))

            if hasattr(gw._scf, 'sigma'):
                eia = mo_energy[i, :, None] - mo_energy[a, None, :]
                fia = (mo_occ[i][:, None] - mo_occ[a][None, :]) / 2.0
                Lia = Lpq
                for w in range(nw):
                    freqs_w = freqs[w]
                    try:
                        import numexpr as ne

                        eia_w = ne.evaluate('eia * fia / (freqs_w**2 + eia**2)')
                        Pia = ne.evaluate('Lia * eia_w')
                    except ImportError:
                        eia_w = eia * fia / (freqs_w**2 + eia**2)
                        Pia = Lia * eia_w
                    # Response from both spin-up and spin-down density
                    # both ia and ai are included, this gives a factor of 2.0
                    # Pi += (2./nkpts) * einsum('Pia,Qia->PQ', Pia, Lpq_i.conj())
                    scipy.linalg.blas.zgemm(
                        alpha=2.0 / nkpts,
                        a=Lia.reshape(naux, nmo * nmo).T,
                        b=Pia.reshape(naux, nmo * nmo).T,
                        c=Pi[w].T,
                        trans_a=2,
                        trans_b=0,
                        beta=1.0,
                        overwrite_c=True,
                    )
                    del eia_w, Pia
                del eia, fia
            else:
                eia = mo_energy[i, :nocc, None] - mo_energy[a, None, nocc:]
                Lia = np.ascontiguousarray(Lpq[:, :nocc, nocc:])
                nvir = Lia.shape[-1]
                for w in range(nw):
                    freqs_w = freqs[w]
                    try:
                        import numexpr as ne

                        eia_w = ne.evaluate('eia / (freqs_w**2 + eia**2)')
                        Pia = ne.evaluate('Lia * eia_w')
                    except ImportError:
                        eia_w = eia / (freqs_w**2 + eia**2)
                        Pia = Lia * eia_w
                    # Response from both spin-up and spin-down density
                    # Pi += (4./nkpts) * einsum('Pia,Qia->PQ', Pia, Lov.conj())
                    scipy.linalg.blas.zgemm(
                        alpha=4.0 / nkpts,
                        a=Lia.reshape(naux, nocc * nvir).T,
                        b=Pia.reshape(naux, nocc * nvir).T,
                        c=Pi[w].T,
                        trans_a=2,
                        trans_b=0,
                        beta=1.0,
                        overwrite_c=True,
                    )
                    del eia_w, Pia
                del eia
            del Lpq, Lia

        logger.timer(gw, 'Calculate Pi for kL: %s / %s on rank@%d' % (kL + 1, nkpts, rank), *cput1)

        for w in range(nw):
            Pi[w] = np.linalg.inv(np.eye(naux) - Pi[w])
            Pi[w] -= np.eye(naux)
        Pi_inv = Pi

        # allocate intermediates
        naux_ones = np.ones(shape=[1, naux], dtype=np.complex128)
        mnQ = np.zeros(shape=[nmo * norbs, naux], dtype=np.complex128)
        if fullsigma is False:
            Qmn = np.zeros(shape=[naux, nmo * norbs], dtype=np.complex128)
            Wmn = np.zeros(shape=[nmo, norbs], dtype=np.complex128)
        else:
            Wmn = np.zeros(shape=[nmo, norbs, norbs], dtype=np.complex128)
            #Lij_kmQn = np.ascontiguousarray(Lij.transpose(0, 2, 1, 3))

        for kn in range(nklist):
            # Find km that conserves with kn and kL (-km+kn+kL=G)
            km = kidx_r[kn]

            cput2 = (time.process_time(), time.perf_counter())
            logger.debug(gw, 'sigma (kL: %s / %s, ki: %s, kj: %s @ Rank %d)' % (kL + 1, nkpts, km, kn, rank))
            Lpq_ao = cderiarr.load(kpts[km], kpts[kn])
            if Lpq_ao.shape[-1] == (nao * (nao + 1)) // 2:
                Lpq_ao = lib.unpack_tril(Lpq_ao).reshape(-1,nao**2)
            else:
                Lpq_ao = Lpq_ao.reshape(-1,nao**2)
            Lpq_ao = Lpq_ao.astype(np.complex128)

            Lpq = None
            moij, ijslice = _conc_mos(mo_coeff[km], mo_coeff[kn])[2:]
            Lpq = _ao2mo.r_e2(Lpq_ao, moij, ijslice, tao=[], ao_loc=None, out=Lpq)
            Lpq = np.ascontiguousarray(Lpq.reshape(-1, nmo, nmo))

            if len(orbs) == nmo:
                l_slice = np.ascontiguousarray(Lpq.reshape(naux, -1))
                if fullsigma:
                    l_slice_mQn = np.ascontiguousarray(Lpq.transpose(1, 0, 2))
            else:
                l_slice = np.ascontiguousarray(Lpq[:, :, mkslice(orbs)].reshape(naux, -1))
                if fullsigma:
                    l_slice_mQn = np.ascontiguousarray(Lpq[:, :, mkslice(orbs)].transpose(1, 0, 2))

            for w in range(nw):
                try:
                    import numexpr as ne

                    wts_w = wts[w]
                    freqs_w = freqs[w]
                    emo_km = emo[km]
                    g0 = ne.evaluate('wts_w * emo_km / (emo_km**2 + freqs_w ** 2)')
                except ImportError:
                    g0 = wts[w] * emo[km] / (emo[km]**2 + freqs[w] ** 2)

                # Qmn = einsum('Pmn,PQ->Qmn', Lij[km][:, :, orbs].conj(), Pi_inv)
                scipy.linalg.blas.zgemm(alpha=1.0, a=Pi_inv[w].T, b=l_slice.T, c=mnQ.T, overwrite_c=1, trans_b=2)

                if fullsigma is False:
                    # Wmn = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn,Lij[km][:,:,orbs])
                    try:
                        import numexpr as ne

                        Qmn[:] = mnQ.T
                        ne.evaluate('Qmn * l_slice', out=Qmn)
                    except ImportError:
                        Qmn[:] = mnQ.T * l_slice
                    np.matmul(naux_ones, Qmn, out=Wmn.reshape(1, nmo * norbs))
                    arraymath.array_scale(Wmn, 1.0 / nkpts / np.pi)

                    # sigma[kn] += -einsum('mn,mw->nw',Wmn,g0[km]) / np.pi
                    # 1 / np.pi is included in Wmn above
                    sigma[kn] -= np.matmul(Wmn.reshape(nmo, norbs).T, g0)
                else:
                    # for orbm in range(nmo):
                    #     Wmn[orbm] = 1./nkpts * np.dot(Qmn[:,orbm,:].transpose(),Lij[km][:,orbm,orbs])
                    #for m in range(nmo):
                    #    np.matmul(Qmn[:, m, :].T, np.ascontiguousarray(Lij[km, :, m, mkslice(orbs)]), out=Wmn[m])
                    np.matmul(mnQ.reshape(nmo, norbs, naux), l_slice_mQn, out=Wmn)
                    arraymath.array_scale(Wmn, 1.0 / nkpts / np.pi)

                    #Wmn = Wmn.reshape(nmo, norbs * norbs).T
                    # sigma[kn] += -einsum('mnl,mw->nlw',Wmn,g0[km])/np.pi
                    # 1 / np.pi is included in Wmn above
                    sigma[kn] -= np.matmul(Wmn.reshape(nmo, norbs * norbs).T, g0).reshape(norbs, norbs, nw_sigma)

            del Lpq, l_slice
            if fullsigma:
                del l_slice_mQn
            logger.timer(gw, 'GW correlation self-energy for kL: %s / %s kn: %d on rank@%d' % (kL + 1, nkpts, kn, rank), *cput2)

        del Pi, Pi_inv, mnQ, Wmn
        if fullsigma is False:
            del Qmn
        logger.timer(gw, 'GW correlation self-energy for kL: %s / %s on rank@%d' % (kL + 1, nkpts, rank), *cput3)
    safeAllreduceInPlace(comm, sigma)

    if gw.rdm:
        gw.sigmaI = sigma

    if rank == 0:
        logger.timer(gw, 'GW correlation self-energy', *cput0)

    return sigma, omega


def get_sigma_exchange(gw, mo_coeff_full=None, mo_occ_full=None):
    """Get exchange self-energy (EXX).

    Parameters
    ----------
    gw : KRGWAC
        gw object
    mo_coeff : complex ndarray, optional
        orbital coefficient, by default None
    mo_occ : double ndarray, optional
        occupation number, by default None

    Returns
    -------
    vk : complex ndarray
        exchange self-energy
    """
    nmo = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts

    if mo_coeff_full is None:
        mo_coeff_full = gw.mo_coeff
    if mo_occ_full is None:
        mo_occ_full = gw.mo_occ
    nao = mo_coeff_full[0].shape[0]
    nmo_full = nao
    nocc_full = int(np.sum(gw._scf.mo_occ[0])) // 2

    # possible kpts shift center
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    vk = np.zeros(shape=[nkpts, nmo_full, nmo_full], dtype=np.complex128)

    segsize = nkpts // size
    if rank >= size - (nkpts - segsize * size):
        start = rank * segsize + rank - (size - (nkpts - segsize * size))
        stop = min(nkpts, start + segsize + 1)
    else:
        start = rank * segsize
        stop = min(nkpts, start + segsize)

    cderiarr = gw.with_df.cderi_array()
    for kL in range(start, stop):
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros(shape=[nkpts], dtype=np.int64)
        kidx_r = np.zeros(shape=[nkpts], dtype=np.int64)
        for i in range(nkpts):
            for j in range(nkpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i

        for kn in range(nkpts):
            # kn is i
            # Find km that conserves with kn and kL (-km+kn+kL=G)
            km = kidx_r[kn] # km is j

            # logger.debug(gw, 'Read Lpq (kL: %s / %s, ki: %s, kj: %s @ Rank %d)' % (kL + 1, nkpts, i, j, rank))

            # Read (L|pq) and ao2mo transform to (L|ij)
            # support unequal naux on different k points
            Lpq_ao = cderiarr.load(kpts[km], kpts[kn])
            if Lpq_ao.shape[-1] == (nao * (nao + 1)) // 2:
                Lpq_ao = lib.unpack_tril(Lpq_ao).reshape(-1, nao**2)
            else:
                Lpq_ao = Lpq_ao.reshape(-1, nao**2)
            Lpq_ao = Lpq_ao.astype(np.complex128)

            Lij = None
            if hasattr(gw._scf, 'sigma'):
                moij, ijslice = _conc_mos(mo_coeff_full[km], mo_coeff_full[kn])[2:]
                Lij = _ao2mo.r_e2(Lpq_ao, moij, ijslice, tao=[], ao_loc=None, out=Lij)
                Lij = Lij.reshape(-1, nmo_full, nmo_full)
            else:
                moij, ijslice = _conc_mos(mo_coeff_full[km][:, :nocc_full], mo_coeff_full[kn])[2:]
                Lij = _ao2mo.r_e2(Lpq_ao, moij, ijslice, tao=[], ao_loc=None, out=Lij)
                Lij = Lij.reshape(-1, nocc_full, nmo_full)

            if hasattr(gw._scf, 'sigma'):
                # vk[k] -= 1.0 / nkpts * einsum('i,Lip,Liq->pq', mo_occ[km], Lij[km].conj(), Lij[km]) * 0.5
                Lij_occ = Lij * mo_occ_full[km][None, :, None]
                scipy.linalg.blas.zgemm(
                    alpha=-0.5 / nkpts,
                    a=Lij_occ.reshape(-1, nmo_full).T,
                    b=Lij.reshape(-1, nmo_full).T,
                    c=vk[kn].T,
                    trans_a=0,
                    trans_b=2,
                    beta=1.0,
                    overwrite_c=True,
                )
            else:
                # vk[k] -= 1.0 / nkpts * einsum('Lip,Liq->pq', Lij[km].conj(), Lij[km])
                scipy.linalg.blas.zgemm(
                    alpha=-1.0 / nkpts,
                    a=Lij.reshape(-1, nmo_full).T,
                    b=Lij.reshape(-1, nmo_full).T,
                    c=vk[kn].T,
                    trans_a=0,
                    trans_b=2,
                    beta=1.0,
                    overwrite_c=True,
                )

    safeAllreduceInPlace(comm, vk)

    if nmo != nmo_full:
        frozen_mask = get_frozen_mask(gw)
        identity = np.eye(nmo_full, dtype=np.complex128)
        vk_frz = np.zeros(shape=[nkpts, nmo, nmo], dtype=np.complex128)
        for k in range(nkpts):
            vk_frz[k] = identity[frozen_mask[k], :] @ vk[k] @ identity[:, frozen_mask[k]]
        vk = vk_frz

    return vk


def get_ef(kmf, mo_energy):
    """Get Fermi level.
    For gapped systems, Fermi level is computed as the average between HOMO and LUMO.
    For metallic systems, Fermi level is optmized according to mo_energy.

    Parameters
    ----------
    kmf : pyscf.pbc.scf.rhf.RHF/pyscf.pbc.dft.rks.RKS
        mean-field object, provides attributes: kpts, sigma, smearing_method
    mo_energy : double array
        orbital energy

    Returns
    -------
    ef : double
        Fermi level
    """
    if hasattr(kmf, "sigma"):
        from pyscf.scf import addons as mol_addons

        if kmf.smearing_method.lower() == "fermi":
            f_occ = mol_addons._fermi_smearing_occ
        else:
            f_occ = mol_addons._gaussian_smearing_occ
        mo_energy_stack = np.hstack(np.asarray(mo_energy))
        nelectron = kmf.mol.tot_electrons(len(kmf.kpts))
        ef = mol_addons._smearing_optimize(f_occ, mo_energy_stack, (nelectron + 1) // 2, kmf.sigma)[0]
    else:
        nocc = int(kmf.cell.nelectron // 2)
        homo = -99.0
        lumo = 99.0
        for k in range(len(kmf.kpts)):
            if homo < mo_energy[k][nocc - 1]:
                homo = mo_energy[k][nocc - 1]
            if lumo > mo_energy[k][nocc]:
                lumo = mo_energy[k][nocc]
        ef = (homo + lumo) / 2.0
    return ef


def get_g0_k(omega, mo_energy, eta):
    """Get non-interacting Green's function.

    Parameters
    ----------
    omega : double or complex ndarray
        frequency grids
    mo_energy : double ndarray
        orbital energy
    eta : double
        broadening parameter

    Returns
    -------
    gf0 : complex ndarray
        non-interacting Green's function
    """
    nkpts = len(mo_energy)
    nmo = len(mo_energy[0])
    nw = len(omega)
    gf0 = np.zeros(shape=[nkpts, nmo, nmo, nw], dtype=np.complex128)
    for k in range(nkpts):
        for iw in range(nw):
            gf0[k, :, :, iw] = np.diag(1.0 / (omega[iw] + 1j * eta - mo_energy[k]))
    return gf0


def make_gf(gw, omega, eta):
    """Get dynamical Green's function and self-energy.

    Parameters
    ----------
    gw : KRGWAC
        GW object, provides attributes: orbs, kptlist, ef, ac_coeff, omega_fit, vk, vxc, _scf.mo_energy
    omega : double or complex array
        frequency grids
    eta : double
        broadening parameter

    Returns
    -------
    gf : complex ndarray
        GW Green's function
    gf0 : complex ndarray
        mean-field Green's function
    sigma : complex ndarray
        GW correlation self-energy
    """
    assert gw.frozen is None or gw.frozen == 0

    if eta is None:
        eta = gw.eta

    nomega = len(omega)
    sigma = np.zeros(shape=[gw.nkpts, gw.nmo, gw.nmo, nomega], dtype=np.complex128)
    if gw.fullsigma:
        for ik, k in enumerate(gw.kptlist):
            for ip, p in enumerate(gw.orbs_frz):
                for iq, q in enumerate(gw.orbs_frz):
                    sigma[k, p, q] = gw.acobj[ik, ip, iq].ac_eval(omega + 1j * eta) + gw.vk[k, p, q] - gw.vxc[k, p, q]
    else:
        for ik, k in enumerate(gw.kptlist):
            for ip, p in enumerate(gw.orbs_frz):
                sigma[k, p, p] = gw.acobj[ik, ip].ac_eval(omega + 1j * eta) + gw.vk[k, p, p] - gw.vxc[k, p, p]

    gf0 = get_g0_k(omega, gw._scf.mo_energy, eta)
    gf = np.zeros_like(gf0)
    for k in range(gw.nkpts):
        for iw in range(nomega):
            gf[k, :, :, iw] = np.linalg.inv(np.linalg.inv(gf0[k, :, :, iw]) - sigma[k, :, :, iw])

    return gf, gf0, sigma


def make_rdm1_linear(gw, ao_repr=False):
    """Get GW density matrix from Green's function G(it=0).
    G is from linear Dyson equation, which conserves particle number
    G = G0 + G0 Sigma G0
    See equation 16 in 10.1021/acs.jctc.0c01264

    Parameters
    ----------
    gw : KRGWAC
        GW object, provides attributes: sigmaI, mol, _scf, freqs, wts, frozen, orbs, fc
    ao_repr : bool, optional
        return density matrix in AO, by default False

    Returns
    -------
    rdm1 : double ndarray
        density matrix
    """
    assert gw.sigmaI is not None
    assert gw.rdm is True and gw.fullsigma is True
    assert gw.frozen is None or gw.frozen == 0
    sigmaI = gw.sigmaI[:, :, :, 1:]
    freqs = 1j * gw.freqs
    wts = gw.wts
    nmo = gw.nmo
    nkpts = gw.nkpts
    if len(gw.orbs) != nmo:
        sigma = np.zeros(shape=[nkpts, nmo, nmo, len(freqs)], dtype=sigmaI.dtype)
        for k in range(nkpts):
            for ia, a in enumerate(gw.orbs):
                for ib, b in enumerate(gw.orbs):
                    sigma[k, a, b, :] = sigmaI[k, ia, ib, :]
    else:
        sigma = sigmaI

    for iw in range(len(freqs)):
        sigma[:, :, :, iw] += gw.vk - gw.vxc
    gf0 = get_g0_k(freqs, np.array(gw._scf.mo_energy) - gw.ef, eta=0)
    gf = np.array(gf0, copy=True)
    for k in range(nkpts):
        for iw in range(len(freqs)):
            gf[k, :, :, iw] = reduce(np.matmul, (gf0[k, :, :, iw], sigma[k, :, :, iw], gf0[k, :, :, iw]))

    # GW density matrix
    rdm1 = np.zeros(shape=[nkpts, nmo, nmo], dtype=np.double)
    for k in range(nkpts):
        rdm1[k] = 2.0 / np.pi * einsum('ijw,w->ij', gf[k], wts).real + np.eye(nmo)
        if rank == 0:
            logger.info(gw, 'GW particle number @ k%d = %s', k, np.trace(rdm1[k]))

    # Symmetrize density matrix
    for k in range(nkpts):
        rdm1[k] = 0.5 * (rdm1[k] + rdm1[k].T)

    if ao_repr is True:
        ovlp = gw._scf.get_ovlp()
        for k in range(nkpts):
            CS = np.matmul(ovlp, gw._scf.mo_coeff[k])
            rdm1[k] = reduce(np.matmul, (CS, rdm1[k], CS.conj().T))

    return rdm1


def _mo_energy_frozen(gw, mo_energy):
    """Get non-frozen orbital energy.

    Parameters
    ----------
    gw : KRGWAC
        GW object, provides attributes: frozen, nmo, nkpt
    mo_energy : double ndarray
        full orbital energy

    Returns
    -------
    mo_energy_frozen : double ndarray
        non-frozen orbital energy
    """
    frozen_mask = get_frozen_mask(gw)
    nmo = gw.nmo
    nkpts = gw.nkpts
    mo_energy_frozen = np.zeros(shape=[nkpts, nmo], dtype=np.double)
    for k in range(nkpts):
        mo_energy_frozen[k] = mo_energy[k][frozen_mask[k]]
    return mo_energy_frozen


def _mo_frozen(gw, mo):
    """Get non-frozen orbital coefficient.

    Parameters
    ----------
    gw : KRGWAC
        GW object, provides attributes: frozen, nmo, nkpt
    mo : complex ndarray
        full orbital coefficient

    Returns
    -------
    mo_frozen : complex ndarray
        non-frozen orbital coefficient
    """
    frozen_mask = get_frozen_mask(gw)
    nmo = gw.nmo
    nkpts = gw.nkpts
    nao = mo[0].shape[0]
    mo_frozen = np.zeros(shape=[nkpts, nao, nmo], dtype=np.complex128)
    for k in range(nkpts):
        mo_frozen[k] = mo[k][:, frozen_mask[k]]
    return mo_frozen


def _mo_occ_frozen(gw, mo_occ):
    """Get non-frozen occupation number.

    Parameters
    ----------
    gw : KRGWAC
        GW object, provides attributes: frozen, nmo, nkpt
    mo_occ : double ndarray
        full occupation number

    Returns
    -------
    mo_occ_frozen : double ndarray
        non-frozen occupation number
    """
    frozen_mask = get_frozen_mask(gw)
    nmo = gw.nmo
    nkpts = gw.nkpts
    mo_occ_frozen = np.zeros(shape=[nkpts, nmo], dtype=np.double)
    for k in range(nkpts):
        mo_occ_frozen[k] = mo_occ[k][frozen_mask[k]]
    return mo_occ_frozen


def set_frozen_orbs(gw):
    """Set .frozen attribute from frozen mask.

    Parameters
    ----------
    gw : KRGWAC
        unrestricted GW object
    """
    if gw.frozen is not None:
        if gw.orbs is not None:
            if isinstance(gw.frozen, (int, np.int64)):
                # frozen core
                gw.orbs_frz = [x - gw.frozen for x in gw.orbs]
            else:
                # frozen list
                assert isinstance(gw.frozen[0], (int, np.int64))
                gw.orbs_frz = []
                for orbi in gw.orbs:
                    count = len([p for p in gw.frozen if p <= orbi])
                    gw.orbs_frz.append(orbi - count)
            if any(np.array(gw.orbs_frz) < 0):
                raise RuntimeError('GW orbs must be larger than frozen core!')
        else:
            gw.orbs_frz = range(gw.nmo)
            gw.orbs = range(len(gw._scf.mo_energy[0]))
            if isinstance(gw.frozen, (int, np.int64)):
                gw.orbs = list(set(gw.orbs) - set(range(gw.frozen)))
            else:
                assert isinstance(gw.frozen[0], (int, np.int64))
                gw.orbs = list(set(gw.orbs) - set(gw.frozen))
    else:
        if gw.orbs is None:
            gw.orbs = range(len(gw._scf.mo_energy[0]))
        gw.orbs_frz = gw.orbs
    return


class KRGWAC(lib.StreamObject):
    def __init__(self, mf, frozen=None):
        self.mol = mf.mol  # mol object
        self._scf = mf  # mean-field object
        self.verbose = self.mol.verbose  # verbose level
        self.stdout = self.mol.stdout  # standard output
        self.max_memory = mf.max_memory  # max memory in MB

        # options
        self.frozen = frozen  # frozen orbital option
        self.orbs = None  # list of orbital index in full nmo
        self.orbs_frz = None  # list of orbital index in non-frozen nmo
        self.kptlist = None  # list of k-points to evaluate
        self.fullsigma = False  # calculate off-diagonal self-energy
        self.rdm = False  # calculate GW density matrix
        self.vhf_df = False  # use density-fitting for exchange self-energy
        self.fc = True  # finite-size correction to self-energy
        self.fc_grid = False  # grids for finite-size correction to self-energy
        self.outcore = False  # low-memory routine to calculate self-energy
        self.eta = 5.0e-3  # broadening parameter
        self.nw = 100  # number of grids for integration
        self.ac = 'pade'  # analytical continuation method
        self.ac_iw_cutoff = 5.0  # imaginary frequency cutting for fitting self-energy
        self.ac_pade_npts = 18  # number of selected points for Pade approximation
        self.ac_pade_step_ratio = 2.0 / 3.0  # final/initial step size for Pade approximation
        self.qpe_max_iter = 100  # max iteration in iteratively solving quasiparticle equation
        self.qpe_tol = 1.0e-6  # tolerance in Newton method for iteratively quasiparticle equation
        self.qpe_linearized = False  # use linearized quasiparticle equation
        self.qpe_linearized_range = [0.5, 1.5]  # Z-shot factor range, if not in this range, z=1
        self.writefile = 0  # write file level

        # DF-KGW must use GDF integrals
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise NotImplementedError
        self._keys.update(['with_df'])

        ##################################################
        # don't modify the following attributes, they are not input options
        self._nocc = None  # number of NON-FROZEN occupied orbitals
        self._nmo = None  # number of NON-FROZEN orbitals
        self.kpts = mf.kpts  # k-point list
        self.nkpts = len(self.kpts)  # number of k-points
        self.mo_energy = None  # orbital energy
        self.mo_coeff = None  # orbital coefficient
        self.mo_occ = None  # occupiation number

        # results
        self.vk = None  # exchange matrix in MO
        self.vxc = None  # mean-field exchange-correlation matrix in MO
        self.freqs = None  # frequency grids
        self.wts = None  # weights of frequency grids
        self.ef = None  # Fermi level
        self.acobj = None  # analytical continuation object
        self.ac_coeff = None  # Pade fitting coefficient, old interface, to be deprecated
        self.omega_fit = None  # AC fitting frequency, old interface, to be deprecated
        self.sigmaI = None  # self-energy in the imaginary axis

        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        log.info('GW nocc = %d, nvir = %d, nkpts = %d', nocc, nvir, nkpts)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', str(self.frozen))
        if self.kptlist is not None:
            log.info('k-point list = %s', str(self.kptlist))
        if self.orbs is not None:
            log.info('orbital list = %s', str(self.orbs))
        log.info('off-diagonal self-energy = %s', self.fullsigma)
        log.info('GW density matrix = %s', self.rdm)
        log.info('density-fitting for exchange = %s', self.vhf_df)
        log.info('outcore for self-energy= %s', self.outcore)
        log.info('finite size corrections = %s', self.fc)
        if self.fc_grid is not None:
            log.info('grids for finite size corrections = %s', self.fc_grid)
        log.info('broadening parameter = %.3e', self.eta)
        log.info('number of grids = %d', self.nw)
        log.info('analytic continuation method = %s', self.ac)
        log.info('imaginary frequency cutoff = %s', str(self.ac_iw_cutoff))
        if self.ac == 'pade':
            log.info('Pade points = %d', self.ac_pade_npts)
            log.info('Pade step ratio = %.3f', self.ac_pade_step_ratio)
        log.info('use perturbative linearized QP eqn = %s', self.qpe_linearized)
        if self.qpe_linearized is True:
            log.info('linearized factor range = %s', self.qpe_linearized_range)
        else:
            log.info('QPE max iter = %d', self.qpe_max_iter)
            log.info('QPE tolerance = %.1e', self.qpe_tol)
        log.info('')
        return

    @property
    def nocc(self):
        frozen_mask = get_frozen_mask(self)
        nkpts = len(self._scf.mo_energy)
        nelec = 0.0
        for k in range(nkpts):
            nelec += np.sum(self._scf.mo_occ[k][frozen_mask[k]])
        nelec = int(nelec / nkpts)
        return nelec // 2

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        frozen_mask = get_frozen_mask(self)
        return len(self._scf.mo_energy[0][frozen_mask[0]])

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    def kernel(self, orbs=None, kptlist=None):
        """Run a G0W0 calculation.

        Parameters
        ----------
        orbs : list, optional
            orbital list to calculate self-energy, by default None
        kptlist : list, optional
            k-point list to calculate self-energy, by default None
        """
        if self.mo_energy is None:
            self.mo_energy = np.array(self._scf.mo_energy, copy=True)
        if self.mo_coeff is None:
            self.mo_coeff = np.array(self._scf.mo_coeff, copy=True)
        if self.mo_occ is None:
            self.mo_occ = np.array(self._scf.mo_occ, copy=True)

        self.orbs = orbs
        self.kptlist = kptlist

        if hasattr(self._scf, "sigma"):
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0
            self.fc = False

        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (2 * nkpts * nmo**2 * naux) * 16 / 1e6
        mem_now = lib.current_memory()[0]
        if mem_incore + mem_now > 0.99 * self.max_memory:
            if rank == 0:
                logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError

        cput0 = (time.process_time(), time.perf_counter())
        if rank == 0:
            self.dump_flags()
        kernel(self)
        if rank == 0:
            logger.timer(self, 'KRGWAC', *cput0)
        return

    set_frozen_orbs = set_frozen_orbs
    make_rdm1 = make_rdm1_linear
    make_gf = make_gf
    get_sigma_exchange = get_sigma_exchange


if __name__ == "__main__":
    import os
    from pyscf.pbc import gto, dft, scf
    from pyscf.pbc.lib import chkfile

    test_tol = 1e-4 # test tolerance for orbital energy

    cell_test = gto.Cell()
    cell_test.build(
        unit='angstrom',
        a="""
                0.000000     1.783500     1.783500
                1.783500     0.000000     1.783500
                1.783500     1.783500     0.000000
            """,
        atom='C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
        dimension=3,
        max_memory=8000,
        verbose=4,
        pseudo='gth-pade',
        basis='gth-szv',
        precision=1e-10,
    )
    kpts_test = cell_test.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    gdf_test = df.RSDF(cell_test, kpts_test)
    gdf_fname_test = 'gdf_ints_311.h5'
    gdf_test._cderi_to_save = gdf_fname_test
    if not os.path.isfile(gdf_fname_test):
        gdf_test.build()

    chkfname_test = 'diamond_311.chk'
    if os.path.isfile(chkfname_test):
        kmf_test = dft.KRKS(cell_test, kpts_test).rs_density_fit()
        kmf_test.xc = 'pbe'
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        data_test = chkfile.load(chkfname_test, 'scf')
        kmf_test.__dict__.update(data_test)
    else:
        kmf_test = dft.KRKS(cell_test, kpts_test).rs_density_fit()
        kmf_test.xc = 'pbe'
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        kmf_test.conv_tol = 1e-12
        kmf_test.chkfile = chkfname_test
        kmf_test.kernel()

    gw_test = KRGWAC(kmf_test)
    # without finite size correction
    gw_test.fc = False
    gw_test.kernel(kptlist=[0, 1, 2], orbs=range(0, 7))
    if rank == 0:
        print(gw_test.mo_energy)
    np.testing.assert_allclose(gw_test.mo_energy[0][3], 0.62047817, atol=test_tol)
    np.testing.assert_allclose(gw_test.mo_energy[0][4], 0.96570185, atol=test_tol)
    np.testing.assert_allclose(gw_test.mo_energy[1][3], 0.52637737, atol=test_tol)
    np.testing.assert_allclose(gw_test.mo_energy[1][4], 1.0751805, atol=test_tol)

    gw_test = KRGWAC(kmf_test)
    # without finite size correction out use outcore routine
    gw_test.fc = False
    gw_test.outcore = True
    gw_test.kernel(kptlist=[0, 1, 2], orbs=range(0, 7))
    if rank == 0:
        print(gw_test.mo_energy)
    np.testing.assert_allclose(gw_test.mo_energy[0][3], 0.62047817, atol=test_tol)
    np.testing.assert_allclose(gw_test.mo_energy[0][4], 0.96570185, atol=test_tol)
    np.testing.assert_allclose(gw_test.mo_energy[1][3], 0.52637737, atol=test_tol)
    np.testing.assert_allclose(gw_test.mo_energy[1][4], 1.0751805, atol=test_tol)

    # with finite size correction
    gw_test = KRGWAC(kmf_test)
    gw_test.fc = True
    gw_test.kernel(kptlist=[0, 1, 2], orbs=range(0, 7))
    if rank == 0:
        print(gw_test.mo_energy)
    np.testing.assert_allclose(gw_test.mo_energy[0][3], 0.44029997, atol=test_tol)
    np.testing.assert_allclose(gw_test.mo_energy[0][4], 0.80144906, atol=test_tol)
    np.testing.assert_allclose(gw_test.mo_energy[1][3], 0.35195222, atol=test_tol)
    np.testing.assert_allclose(gw_test.mo_energy[1][4], 0.92909667, atol=test_tol)

    # frozen core with finite size correction
    gw_test = KRGWAC(kmf_test)
    gw_test.fc = True
    gw_test.frozen = 1
    gw_test.kernel()
    if rank == 0:
        print(gw_test.mo_energy)
    np.testing.assert_allclose(gw_test.mo_energy[0][3], 0.44098086, atol=test_tol)
    np.testing.assert_allclose(gw_test.mo_energy[0][4], 0.79816375, atol=test_tol)

    # full self-energy and density of states
    gw_test = KRGWAC(kmf_test)
    gw_test.eta = 1e-2
    gw_test.fullsigma = True
    gw_test.fc = True
    gw_test.rdm = True
    omega_test = np.linspace(0.2, 1.2, 101)
    gw_test.kernel()
    gf_test, gf0_test, sigma_test = gw_test.make_gf(omega_test, gw_test.eta)
    if rank == 0:
        for i in range(len(omega_test)):
            print(
                omega_test[i],
                (-np.trace(gf0_test[0, :, :, i].imag)) / np.pi,
                (-np.trace(gf_test[0, :, :, i].imag)) / np.pi,
            )

    # Test on Na (metallic)
    cell_test = gto.Cell()
    cell_test.build(
        unit='angstrom',
        a="""
           -2.11250000000000   2.11250000000000   2.11250000000000
           2.11250000000000  -2.11250000000000   2.11250000000000
           2.11250000000000   2.11250000000000  -2.11250000000000
           """,
        atom="""
           Na   0.00000   0.00000   0.00000
           """,
        dimension=3,
        max_memory=126000,
        verbose=5,
        pseudo='gth-pade',
        basis='gth-dzvp-molopt-sr',
        precision=1e-10,
    )

    kpts_test = cell_test.make_kpts([2, 2, 1], scaled_center=[0, 0, 0])
    gdf_test = df.RSDF(cell_test, kpts_test)
    gdf_fname_test = 'gdf_ints_221.h5'
    gdf_test._cderi_to_save = gdf_fname_test
    if not os.path.isfile(gdf_fname_test):
        gdf_test.build()

    chkfname_test = 'na_221.chk'
    if os.path.isfile(chkfname_test):
        kmf_test = dft.KRKS(cell_test, kpts_test).rs_density_fit()
        kmf_test = scf.addons.smearing_(kmf_test, sigma=5e-3, method='fermi')
        kmf_test.xc = 'lda'
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        data_test = chkfile.load(chkfname_test, 'scf')
        kmf_test.__dict__.update(data_test)
    else:
        kmf_test = dft.KRKS(cell_test, kpts_test).rs_density_fit()
        kmf_test = scf.addons.smearing_(kmf_test, sigma=5e-3, method='fermi')
        kmf_test.xc = 'lda'
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        kmf_test.conv_tol = 1e-12
        kmf_test.chkfile = chkfname_test
        kmf_test.kernel()

    gw_test = KRGWAC(kmf_test)
    gw_test.kernel(kptlist=[0, 1, 2], orbs=range(1, 7))
    np.testing.assert_allclose(gw_test.mo_energy[0][3], -0.94581972, atol=test_tol)
    np.testing.assert_allclose(gw_test.mo_energy[0][4], 0.04603593, atol=test_tol)

    # outcore routine for metallic system
    gw_test = KRGWAC(kmf_test)
    gw_test.outcore = True
    gw_test.kernel(kptlist=[0, 1, 2], orbs=range(1, 7))
    np.testing.assert_allclose(gw_test.mo_energy[0][3], -0.94581972, atol=test_tol)
    np.testing.assert_allclose(gw_test.mo_energy[0][4], 0.04603593, atol=test_tol)

    # frozen core
    gw_test = KRGWAC(kmf_test)
    gw_test.frozen = 1
    gw_test.kernel()
    if rank == 0:
        print(gw_test.mo_energy)
    np.testing.assert_allclose(gw_test.mo_energy[0][3], -0.94911561, atol=test_tol)
    np.testing.assert_allclose(gw_test.mo_energy[0][4], 0.04504422, atol=test_tol)

    # full self-energy and density of states
    gw_test = KRGWAC(kmf_test)
    gw_test.eta = 1e-2
    gw_test.fullsigma = True
    omega_test = np.linspace(-0.5, 0.5, 101)
    gw_test.kernel()
    gf_test, gf0_test, sigma_test = gw_test.make_gf(omega_test, eta=0.01)
    print('\nk=0 DOS: KS, GW')
    if rank == 0:
        for i in range(len(omega_test)):
            print(
                omega_test[i],
                (-np.trace(gf0_test[0, :, :, i].imag)) / np.pi,
                (-np.trace(gf_test[0, :, :, i].imag)) / np.pi,
            )

    if rank == 0:
        print('passed tests!')
