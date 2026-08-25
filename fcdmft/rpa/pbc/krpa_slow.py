"""
Periodic spin-restricted random phase approximation (direct RPA) with N^4 scaling.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14, 053020 (2012)
"""

from functools import reduce
import time, h5py, os
import numpy
import numpy as np
from scipy.optimize import newton, least_squares

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc import df, dft, scf
from pyscf.pbc.mp.kmp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__

from fcdmft.gw.mol.gw_ac import _get_scaled_legendre_roots
from fcdmft.rpa.pbc.rpa import get_idx_metal
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

einsum = lib.einsum


def kernel(rpa, mo_energy, mo_coeff, nw=None, verbose=logger.NOTE):
    """
    RPA correlation and total energy

    Returns:
        e_tot : RPA total energy
        e_hf : EXX energy
        e_corr : RPA correlation energy
    """
    mf = rpa._scf

    nkpts = rpa.nkpts
    mo_occ = _mo_occ_frozen(rpa, rpa._scf.mo_occ)

    # Compute HF exchange energy (EXX)
    dm = mf.make_rdm1()
    rhf = scf.KRHF(rpa.mol, rpa.kpts, exxdiv=mf.exxdiv)
    if hasattr(rpa._scf, 'sigma'):
        rhf = scf.addons.smearing_(rhf, sigma=rpa._scf.sigma, method='fermi')
    rhf.with_df = mf.with_df
    rhf.with_df._cderi = mf.with_df._cderi
    e_hf = rhf.energy_elec(dm)[0]
    e_hf += mf.energy_nuc()

    # check metal
    mo_occ_1d = np.array(mo_occ).reshape(-1)
    is_metal = False
    if np.linalg.norm(np.abs(mo_occ_1d - 1.0) - 1.0) > 1e-5:
        is_metal = True

    # Grids for integration on imaginary axis
    freqs, wts = _get_scaled_legendre_roots(nw)

    # Compute RPA correlation energy
    if rpa.outcore and is_metal:
        e_corr = get_rpa_ecorr_outcore_metal(rpa, freqs, wts)
    elif rpa.outcore and (not is_metal):
        e_corr = get_rpa_ecorr_outcore(rpa, freqs, wts)
    else:
        e_corr = get_rpa_ecorr(rpa, freqs, wts)

    # Compute totol energy
    e_tot = e_hf + e_corr

    if rank == 0 and rpa.verbose >= logger.DEBUG:
        logger.debug(rpa, '  RPA total energy = %s', e_tot)
        logger.debug(rpa, '  EXX energy = %s, RPA corr energy = %s', e_hf, e_corr)

    return e_tot, e_hf, e_corr


def get_rpa_ecorr(rpa, freqs, wts, max_memory=8000):
    """
    Compute RPA correlation energy
    """
    mo_coeff = np.array(_mo_frozen(rpa, rpa._scf.mo_coeff))
    mo_energy = np.array(_mo_energy_frozen(rpa, rpa._scf.mo_energy))
    mo_occ = np.array(_mo_occ_frozen(rpa, rpa._scf.mo_occ))

    nocc = rpa.nocc
    nmo = rpa.nmo
    nvir = nmo - nocc
    nao = rpa._scf.mo_coeff[0].shape[0]
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    nw = len(freqs)
    mydf = rpa.with_df

    # possible kpts shift center
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    is_metal = False
    mo_occ_1d = np.array(mo_occ).reshape(-1)
    if np.linalg.norm(np.abs(mo_occ_1d - 1.0) - 1.0) > 1e-5:
        is_metal = True

    segsize = nkpts // size
    if rank >= size - (nkpts - segsize * size):
        start = rank * segsize + rank - (size - (nkpts - segsize * size))
        stop = min(nkpts, start + segsize + 1)
    else:
        start = rank * segsize
        stop = min(nkpts, start + segsize)

    e_corr = 0j
    for kL in range(start, stop):
        # Lij: (ki, L, i, j) for looping every kL
        Lij = []
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros((nkpts), dtype=np.int64)
        kidx_r = np.zeros((nkpts), dtype=np.int64)
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i
                    logger.debug(rpa, 'Read Lpq (kL: %s / %s, ki: %s, kj: %s @ Rank %d)' % (kL + 1, nkpts, i, j, rank))
                    Lij_out = None
                    # Read (L|pq) and ao2mo transform to (L|ij)
                    Lpq = []
                    for LpqR, LpqI, sign in mydf.sr_loop(
                        [kpti, kptj], max_memory=0.1 * rpa._scf.max_memory, compact=False
                    ):
                        Lpq.append(LpqR + LpqI * 1.0j)
                    # support uneqaul naux on different k points
                    Lpq = np.vstack(Lpq).reshape(-1, nao**2)
                    tao = []
                    ao_loc = None
                    moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]
                    if not is_metal:
                        ijslice = (0, nocc, nmo + nocc, 2 * nmo)
                        Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)
                        Lij.append(Lij_out.reshape(-1, nocc, nvir))
                    else:
                        Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)
                        Lij.append(Lij_out.reshape(-1, nmo, nmo))
        Lij = np.asarray(Lij)
        naux = Lij.shape[1]

        for w in range(nw):
            if is_metal:
                Pi = get_rho_response_metal(rpa, freqs[w], mo_energy, mo_occ, Lij, kL, kidx)
            else:
                Pi = get_rho_response(rpa, freqs[w], mo_energy, Lij, kL, kidx)
            ec_w = np.linalg.slogdet((np.eye(naux) - Pi))[1]
            ec_w += np.trace(Pi)
            e_corr += 1.0 / (2.0 * np.pi) * 1.0 / nkpts * ec_w * wts[w]
    comm.Barrier()
    ecorr_gather = comm.gather(e_corr)
    if rank == 0:
        e_corr = np.sum(ecorr_gather)
    comm.Barrier()
    e_corr = comm.bcast(e_corr, root=0)

    return e_corr.real


def get_rho_response(rpa, omega, mo_energy, Lpq, kL, kidx):
    """
    Compute density response function in auxiliary basis at freq iw
    (for gapped systems with integer occupations)
    """
    nkpts, naux, nocc, nvir = Lpq.shape
    kpts = rpa.kpts
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    # Compute Pi for kL
    Pi = np.zeros((naux, naux), dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        eia = mo_energy[i, :nocc, None] - mo_energy[a, None, nocc:]
        eia = eia / (omega**2 + eia * eia)
        Lov = np.asarray(Lpq[i])
        Pia = np.einsum('Pia,ia->Pia', Lov, eia, optimize=True)
        # Response from both spin-up and spin-down density
        Pi += (4.0 / nkpts) * np.einsum('Pia,Qia->PQ', Pia, Lov.conj(), optimize=True)
        Pia = Lov = None
    return Pi


def get_rho_response_metal(rpa, omega, mo_energy, mo_occ, Lpq, kL, kidx):
    """
    Compute density response function in auxiliary basis at freq iw
    (for metallic systems with fractional occupations)
    """
    nkpts, naux, nmo, nmo = Lpq.shape
    kpts = rpa.kpts
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    # Compute Pi for kL
    Pi = np.zeros((naux, naux), dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        idx_occ_i, idx_frac_i, idx_vir_i = get_idx_metal(mo_occ[i])
        idx_occ_a, idx_frac_a, idx_vir_a = get_idx_metal(mo_occ[a])

        # Pi from frac1 to all2 orbs
        eia = mo_energy[i][idx_frac_i][:, None] - mo_energy[a][None, :]
        fia = (mo_occ[i][idx_frac_i][:, None] - mo_occ[a][None, :]) / 2.0
        eia = eia * fia / (omega**2 + eia * eia)
        Lpq_i = Lpq[i][:, idx_frac_i, :]
        Pia = np.einsum('Pia,ia->Pia', Lpq_i, eia, optimize=True)
        Pi += (2.0 / nkpts) * np.einsum('Pia,Qia->PQ', Pia, Lpq_i.conj(), optimize=True)
        Pia = Lpq_i = None

        # Pi from all1 to frac2 orbs
        eia = mo_energy[i][:, None] - mo_energy[a][idx_frac_a][None, :]
        fia = (mo_occ[i][:, None] - mo_occ[a][idx_frac_a][None, :]) / 2.0
        eia = eia * fia / (omega**2 + eia * eia)
        Lpq_i = Lpq[i][:, :, idx_frac_a]
        Pia = np.einsum('Pia,ia->Pia', Lpq_i, eia, optimize=True)
        Pi += (2.0 / nkpts) * np.einsum('Pia,Qia->PQ', Pia, Lpq_i.conj(), optimize=True)
        Pia = Lpq_i = None

        # Pi from frac1 to frac2 orbs (double counting)
        eia = mo_energy[i][idx_frac_i][:, None] - mo_energy[a][idx_frac_a][None, :]
        fia = (mo_occ[i][idx_frac_i][:, None] - mo_occ[a][idx_frac_a][None, :]) / 2.0
        eia = eia * fia / (omega**2 + eia * eia)
        Lpq_i = Lpq[i][:, idx_frac_i, :][:, :, idx_frac_a]
        Pia = np.einsum('Pia,ia->Pia', Lpq_i, eia, optimize=True)
        Pi -= (2.0 / nkpts) * np.einsum('Pia,Qia->PQ', Pia, Lpq_i.conj(), optimize=True)
        Pia = Lpq_i = None

        # Pi from occ1 to vir2 orbs
        eia = mo_energy[i][idx_occ_i][:, None] - mo_energy[a][idx_vir_a][None, :]
        fia = (mo_occ[i][idx_occ_i][:, None] - mo_occ[a][idx_vir_a][None, :]) / 2.0
        eia = eia * fia / (omega**2 + eia * eia)
        Lpq_i = Lpq[i][:, idx_occ_i, :][:, :, idx_vir_a]
        Pia = np.einsum('Pia,ia->Pia', Lpq_i, eia, optimize=True)
        Pi += (2.0 / nkpts) * np.einsum('Pia,Qia->PQ', Pia, Lpq_i.conj(), optimize=True)
        Pia = Lpq_i = None

        # Pi from vir1 to occ2 orbs
        eia = mo_energy[i][idx_vir_i][:, None] - mo_energy[a][idx_occ_a][None, :]
        fia = (mo_occ[i][idx_vir_i][:, None] - mo_occ[a][idx_occ_a][None, :]) / 2.0
        eia = eia * fia / (omega**2 + eia * eia)
        Lpq_i = Lpq[i][:, idx_vir_i, :][:, :, idx_occ_a]
        Pia = np.einsum('Pia,ia->Pia', Lpq_i, eia, optimize=True)
        Pi += (2.0 / nkpts) * np.einsum('Pia,Qia->PQ', Pia, Lpq_i.conj(), optimize=True)
        Pia = Lpq_i = None

        # The code commented out below is equivalent but slower
        """
        eia = mo_energy[i,:,None] - mo_energy[a,None,:]
        fia = (mo_occ[i][:,None] - mo_occ[a][None,:]) / 2.
        eia = eia*fia/(omega**2+eia*eia)
        Lpq_i = np.asarray(Lpq[i])
        Pia = einsum('Pia,ia->Pia', Lpq_i, eia)
        # Response from both spin-up and spin-down density
        Pi += (2./nkpts) * einsum('Pia,Qia->PQ', Pia, Lpq_i.conj())
        Pia = Lpq_i = None
        """
    return Pi


def get_rpa_ecorr_outcore(rpa, freqs, wts, max_memory=8000):
    """
    Compute RPA correlation energy
    """
    mo_coeff = np.array(_mo_frozen(rpa, rpa._scf.mo_coeff))
    mo_energy = np.array(_mo_energy_frozen(rpa, rpa._scf.mo_energy))
    mo_occ = np.array(_mo_occ_frozen(rpa, rpa._scf.mo_occ))

    nocc = rpa.nocc
    nmo = rpa.nmo
    nvir = nmo - nocc
    nao = rpa._scf.mo_coeff[0].shape[0]
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    nw = len(freqs)
    mydf = rpa.with_df

    # possible kpts shift center
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    segsize = nkpts // size
    if rank >= size - (nkpts - segsize * size):
        start = rank * segsize + rank - (size - (nkpts - segsize * size))
        stop = min(nkpts, start + segsize + 1)
    else:
        start = rank * segsize
        stop = min(nkpts, start + segsize)

    e_corr = 0j
    for kL in range(start, stop):
        nseg = nocc // rpa.segsize + 1
        for iseg in range(nseg):
            orb_start = iseg * rpa.segsize
            orb_end = min((iseg + 1) * rpa.segsize, nocc)
            if orb_end == orb_start:
                continue

            # Lij: (ki, L, i, j) for looping every kL
            Lij = []
            # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
            # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
            kidx = np.zeros((nkpts), dtype=np.int64)
            kidx_r = np.zeros((nkpts), dtype=np.int64)
            for i, kpti in enumerate(kpts):
                for j, kptj in enumerate(kpts):
                    # Find (ki,kj) that satisfies momentum conservation with kL
                    kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                    is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                    if is_kconserv:
                        kidx[i] = j
                        kidx_r[j] = i
                        logger.debug(
                            rpa, 'Read Lpq (kL: %s / %s, ki: %s, kj: %s @ Rank %d)' % (kL + 1, nkpts, i, j, rank)
                        )
                        Lij_out = None
                        # Read (L|pq) and ao2mo transform to (L|ij)
                        Lpq = []
                        for LpqR, LpqI, sign in mydf.sr_loop(
                            [kpti, kptj], max_memory=0.1 * rpa._scf.max_memory, compact=False
                        ):
                            Lpq.append(LpqR + LpqI * 1.0j)
                        # support uneqaul naux on different k points
                        Lpq = np.vstack(Lpq).reshape(-1, nao**2)
                        tao = []
                        ao_loc = None
                        moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]

                        ijslice = (orb_start, orb_end, nmo + nocc, 2 * nmo)
                        Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)
                        Lij.append(Lij_out.reshape(-1, orb_end - orb_start, nvir))
            Lij = np.asarray(Lij)
            naux = Lij.shape[1]
            if iseg == 0:
                Pi = np.zeros((naux, naux, nw), dtype=np.complex128)

            for w in range(nw):
                for i_inner, kpti_inner in enumerate(kpts):
                    # Find ka that conserves with ki and kL (-ki+ka+kL=G)
                    a_inner = kidx[i_inner]
                    eia = mo_energy[i_inner][orb_start:orb_end, None] - mo_energy[a_inner][None, nocc:]
                    eia = eia / (freqs[w] ** 2 + eia * eia)
                    Lov = np.asarray(Lij[i_inner])
                    Pia = np.einsum('Pia,ia->Pia', Lov, eia, optimize=True)
                    # Response from both spin-up and spin-down density
                    Pi[:, :, w] += (4.0 / nkpts) * np.einsum('Pia,Qia->PQ', Pia, Lov.conj(), optimize=True)
                    Pia = Lov = None

        for w in range(nw):
            ec_w = np.linalg.slogdet((np.eye(naux) - Pi[:, :, w]))[1]
            ec_w += np.trace(Pi[:, :, w])
            e_corr += 1.0 / (2.0 * np.pi) * 1.0 / nkpts * ec_w * wts[w]
    comm.Barrier()
    ecorr_gather = comm.gather(e_corr)
    if rank == 0:
        e_corr = np.sum(ecorr_gather)
    comm.Barrier()
    e_corr = comm.bcast(e_corr, root=0)

    return e_corr.real


def get_rpa_ecorr_outcore_metal(rpa, freqs, wts, max_memory=8000):
    """
    Compute RPA correlation energy
    """
    mo_coeff = np.array(_mo_frozen(rpa, rpa._scf.mo_coeff))
    mo_energy = np.array(_mo_energy_frozen(rpa, rpa._scf.mo_energy))
    mo_occ = np.array(_mo_occ_frozen(rpa, rpa._scf.mo_occ))

    nocc = rpa.nocc
    nmo = rpa.nmo
    nvir = nmo - nocc
    nao = rpa._scf.mo_coeff[0].shape[0]
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    nw = len(freqs)
    mydf = rpa.with_df

    # possible kpts shift center
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    segsize = nkpts // size
    if rank >= size - (nkpts - segsize * size):
        start = rank * segsize + rank - (size - (nkpts - segsize * size))
        stop = min(nkpts, start + segsize + 1)
    else:
        start = rank * segsize
        stop = min(nkpts, start + segsize)

    # find largest nocc
    nocc_max = 0
    for k in range(nkpts):
        idx_occ, idx_frac, idx_vir = get_idx_metal(mo_occ[k])
        if len(idx_occ) > nocc_max:
            nocc_max = len(idx_occ)

    e_corr = 0j
    for kL in range(start, stop):
        # part 1: Pi from frac1 to all2 orbs
        # Lij: (ki, L, i, j) for looping every kL
        Lij = []
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros((nkpts), dtype=np.int64)
        kidx_r = np.zeros((nkpts), dtype=np.int64)
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i
                    logger.debug(rpa, 'Read Lpq (kL: %s / %s, ki: %s, kj: %s @ Rank %d)' % (kL + 1, nkpts, i, j, rank))
                    Lij_out = None
                    # Read (L|pq) and ao2mo transform to (L|ij)
                    Lpq = []
                    for LpqR, LpqI, sign in mydf.sr_loop(
                        [kpti, kptj], max_memory=0.1 * rpa._scf.max_memory, compact=False
                    ):
                        Lpq.append(LpqR + LpqI * 1.0j)
                    # support uneqaul naux on different k points
                    Lpq = np.vstack(Lpq).reshape(-1, nao**2)
                    naux = len(Lpq)
                    tao = []
                    ao_loc = None
                    moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]

                    idx_occ_i, idx_frac_i, idx_vir_i = get_idx_metal(mo_occ[i])
                    if len(idx_frac_i) > 0:
                        orb_start = idx_frac_i[0]
                        orb_end = idx_frac_i[-1] + 1
                        ijslice = (orb_start, orb_end, nmo, 2 * nmo)
                        Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)
                        Lij.append(Lij_out.reshape(-1, orb_end - orb_start, nmo))
                    else:
                        Lij.append(np.zeros((naux, 0, nmo), dtype=np.complex128))
        Pi = np.zeros((naux, naux, nw), dtype=np.complex128)

        for w in range(nw):
            for i_inner, kpti_inner in enumerate(kpts):
                a_inner = kidx[i_inner]
                idx_occ_i, idx_frac_i, idx_vir_i = get_idx_metal(mo_occ[i_inner])
                idx_occ_a, idx_frac_a, idx_vir_a = get_idx_metal(mo_occ[a_inner])
                if len(idx_frac_i) > 0:
                    eia = mo_energy[i_inner][idx_frac_i][:, None] - mo_energy[a_inner][None, :]
                    fia = (mo_occ[i_inner][idx_frac_i][:, None] - mo_occ[a_inner][None, :]) / 2.0
                    eia = eia * fia / (freqs[w] ** 2 + eia * eia)
                    Pia = np.einsum('Pia,ia->Pia', Lij[i_inner], eia, optimize=True)
                    Pi[:, :, w] += (2.0 / nkpts) * np.einsum('Pia,Qia->PQ', Pia, Lij[i_inner].conj(), optimize=True)
                    Pia = None

        # part 2: Pi from all1 to frac2 orbs
        # Lij: (ki, L, i, j) for looping every kL
        Lij = []
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros((nkpts), dtype=np.int64)
        kidx_r = np.zeros((nkpts), dtype=np.int64)
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i
                    Lij_out = None
                    # Read (L|pq) and ao2mo transform to (L|ij)
                    Lpq = []
                    for LpqR, LpqI, sign in mydf.sr_loop(
                        [kpti, kptj], max_memory=0.1 * rpa._scf.max_memory, compact=False
                    ):
                        Lpq.append(LpqR + LpqI * 1.0j)
                    # support uneqaul naux on different k points
                    Lpq = np.vstack(Lpq).reshape(-1, nao**2)
                    tao = []
                    ao_loc = None
                    moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]

                    idx_occ_j, idx_frac_j, idx_vir_j = get_idx_metal(mo_occ[j])
                    if len(idx_frac_j) > 0:
                        orb_start = idx_frac_j[0]
                        orb_end = idx_frac_j[-1] + 1
                        ijslice = (0, nmo, nmo + orb_start, nmo + orb_end)
                        Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)
                        Lij.append(Lij_out.reshape(-1, nmo, orb_end - orb_start))
                    else:
                        Lij.append(np.zeros((naux, nmo, 0), dtype=np.complex128))

        for w in range(nw):
            for i_inner, kpti_inner in enumerate(kpts):
                a_inner = kidx[i_inner]
                idx_occ_i, idx_frac_i, idx_vir_i = get_idx_metal(mo_occ[i_inner])
                idx_occ_a, idx_frac_a, idx_vir_a = get_idx_metal(mo_occ[a_inner])
                if len(idx_frac_a) > 0:
                    eia = mo_energy[i_inner][:, None] - mo_energy[a_inner][idx_frac_a][None, :]
                    fia = (mo_occ[i_inner][:, None] - mo_occ[a_inner][idx_frac_a][None, :]) / 2.0
                    eia = eia * fia / (freqs[w] ** 2 + eia * eia)
                    Pia = np.einsum('Pia,ia->Pia', Lij[i_inner], eia, optimize=True)
                    Pi[:, :, w] += (2.0 / nkpts) * np.einsum('Pia,Qia->PQ', Pia, Lij[i_inner].conj(), optimize=True)
                    Pia = None

        # part 3: Pi from frac1 to frac2 orbs (double counting)
        # Lij: (ki, L, i, j) for looping every kL
        Lij = []
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros((nkpts), dtype=np.int64)
        kidx_r = np.zeros((nkpts), dtype=np.int64)
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i
                    Lij_out = None
                    # Read (L|pq) and ao2mo transform to (L|ij)
                    Lpq = []
                    for LpqR, LpqI, sign in mydf.sr_loop(
                        [kpti, kptj], max_memory=0.1 * rpa._scf.max_memory, compact=False
                    ):
                        Lpq.append(LpqR + LpqI * 1.0j)
                    # support uneqaul naux on different k points
                    Lpq = np.vstack(Lpq).reshape(-1, nao**2)
                    tao = []
                    ao_loc = None
                    moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]

                    idx_occ_i, idx_frac_i, idx_vir_i = get_idx_metal(mo_occ[i])
                    idx_occ_j, idx_frac_j, idx_vir_j = get_idx_metal(mo_occ[j])
                    if len(idx_frac_i) > 0 and len(idx_frac_j) > 0:
                        orb_start_i = idx_frac_i[0]
                        orb_end_i = idx_frac_i[-1] + 1
                        orb_start_j = idx_frac_j[0]
                        orb_end_j = idx_frac_j[-1] + 1
                        ijslice = (orb_start_i, orb_end_i, nmo + orb_start_j, nmo + orb_end_j)
                        Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)
                        Lij.append(Lij_out.reshape(-1, orb_end_i - orb_start_i, orb_end_j - orb_start_j))
                    else:
                        Lij.append(np.zeros((naux, 0, 0), dtype=np.complex128))

        for w in range(nw):
            for i_inner, kpti_inner in enumerate(kpts):
                a_inner = kidx[i_inner]
                idx_occ_i, idx_frac_i, idx_vir_i = get_idx_metal(mo_occ[i_inner])
                idx_occ_a, idx_frac_a, idx_vir_a = get_idx_metal(mo_occ[a_inner])
                if len(idx_frac_i) > 0 and len(idx_frac_a) > 0:
                    eia = mo_energy[i_inner][idx_frac_i][:, None] - mo_energy[a_inner][idx_frac_a][None, :]
                    fia = (mo_occ[i_inner][idx_frac_i][:, None] - mo_occ[a_inner][idx_frac_a][None, :]) / 2.0
                    eia = eia * fia / (freqs[w] ** 2 + eia * eia)
                    Pia = np.einsum('Pia,ia->Pia', Lij[i_inner], eia, optimize=True)
                    Pi[:, :, w] -= (2.0 / nkpts) * np.einsum('Pia,Qia->PQ', Pia, Lij[i_inner].conj(), optimize=True)
                    Pia = None

        # part 4: Pi from occ1 to vir2 orbs
        nseg = nocc_max // rpa.segsize + 1
        for iseg in range(nseg):
            orb_start = iseg * rpa.segsize
            orb_end = min((iseg + 1) * rpa.segsize, nocc_max)
            if orb_end == orb_start:
                continue

            # Lij: (ki, L, i, j) for looping every kL
            Lij = []
            # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
            # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
            kidx = np.zeros((nkpts), dtype=np.int64)
            kidx_r = np.zeros((nkpts), dtype=np.int64)
            for i, kpti in enumerate(kpts):
                for j, kptj in enumerate(kpts):
                    # Find (ki,kj) that satisfies momentum conservation with kL
                    kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                    is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                    if is_kconserv:
                        kidx[i] = j
                        kidx_r[j] = i
                        Lij_out = None
                        # Read (L|pq) and ao2mo transform to (L|ij)
                        Lpq = []
                        for LpqR, LpqI, sign in mydf.sr_loop(
                            [kpti, kptj], max_memory=0.1 * rpa._scf.max_memory, compact=False
                        ):
                            Lpq.append(LpqR + LpqI * 1.0j)
                        # support uneqaul naux on different k points
                        Lpq = np.vstack(Lpq).reshape(-1, nao**2)
                        tao = []
                        ao_loc = None
                        moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]

                        ijslice = (orb_start, orb_end, nmo, 2 * nmo)
                        Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)
                        Lij.append(Lij_out.reshape(-1, orb_end - orb_start, nmo))
            Lij_out = None

            for w in range(nw):
                for i_inner, kpti_inner in enumerate(kpts):
                    a_inner = kidx[i_inner]
                    idx_occ_i, idx_frac_i, idx_vir_i = get_idx_metal(mo_occ[i_inner])
                    idx_occ_a, idx_frac_a, idx_vir_a = get_idx_metal(mo_occ[a_inner])
                    orb_end_min = min(orb_end, idx_occ_i[-1] + 1)
                    if orb_end_min > orb_start:
                        eia = mo_energy[i_inner][orb_start:orb_end_min, None] - mo_energy[a_inner][idx_vir_a][None, :]
                        fia = (mo_occ[i_inner][orb_start:orb_end_min, None] - mo_occ[a_inner][idx_vir_a][None, :]) / 2.0
                        eia = eia * fia / (freqs[w] ** 2 + eia * eia)
                        Lij_i = Lij[i_inner][:, :, idx_vir_a][:, : (orb_end_min - orb_start), :]
                        # Pia = np.einsum('Pia,ia->Pia', Lij_i, eia, optimize=True)
                        # Pi[:,:,w] += (2./nkpts) * np.einsum('Pia,Qia->PQ', Pia, Lij_i.conj(), optimize=True)
                        Pia = Lij_i * eia
                        Pi[:, :, w] += (2.0 / nkpts) * Pia.reshape(naux, -1) @ Lij_i.reshape(naux, -1).T.conj()
                        Pia = Lij_i = None

        # part 5: Pi from vir1 to occ2 orbs
        nseg = nocc_max // rpa.segsize + 1
        for iseg in range(nseg):
            orb_start = iseg * rpa.segsize
            orb_end = min((iseg + 1) * rpa.segsize, nocc_max)
            if orb_end == orb_start:
                continue

            # Lij: (ki, L, i, j) for looping every kL
            Lij = []
            # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
            # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
            kidx = np.zeros((nkpts), dtype=np.int64)
            kidx_r = np.zeros((nkpts), dtype=np.int64)
            for i, kpti in enumerate(kpts):
                for j, kptj in enumerate(kpts):
                    # Find (ki,kj) that satisfies momentum conservation with kL
                    kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                    is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                    if is_kconserv:
                        kidx[i] = j
                        kidx_r[j] = i
                        Lij_out = None
                        # Read (L|pq) and ao2mo transform to (L|ij)
                        Lpq = []
                        for LpqR, LpqI, sign in mydf.sr_loop(
                            [kpti, kptj], max_memory=0.1 * rpa._scf.max_memory, compact=False
                        ):
                            Lpq.append(LpqR + LpqI * 1.0j)
                        # support uneqaul naux on different k points
                        Lpq = np.vstack(Lpq).reshape(-1, nao**2)
                        tao = []
                        ao_loc = None
                        moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]

                        ijslice = (0, nmo, nmo + orb_start, nmo + orb_end)
                        Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)
                        Lij.append(Lij_out.reshape(-1, nmo, orb_end - orb_start))
            Lij_out = None

            for w in range(nw):
                for i_inner, kpti_inner in enumerate(kpts):
                    a_inner = kidx[i_inner]
                    idx_occ_i, idx_frac_i, idx_vir_i = get_idx_metal(mo_occ[i_inner])
                    idx_occ_a, idx_frac_a, idx_vir_a = get_idx_metal(mo_occ[a_inner])
                    orb_end_min = min(orb_end, idx_occ_a[-1] + 1)
                    if orb_end_min > orb_start:
                        eia = mo_energy[i_inner][idx_vir_i][:, None] - mo_energy[a_inner][None, orb_start:orb_end_min]
                        fia = (mo_occ[i_inner][idx_vir_i][:, None] - mo_occ[a_inner][None, orb_start:orb_end_min]) / 2.0
                        eia = eia * fia / (freqs[w] ** 2 + eia * eia)
                        Lij_i = Lij[i_inner][:, idx_vir_i, :][:, :, : (orb_end_min - orb_start)]
                        # Pia = np.einsum('Pia,ia->Pia', Lij_i, eia, optimize=True)
                        # Pi[:,:,w] += (2./nkpts) * np.einsum('Pia,Qia->PQ', Pia, Lij_i.conj(), optimize=True)
                        Pia = Lij_i * eia
                        Pi[:, :, w] += (2.0 / nkpts) * Pia.reshape(naux, -1) @ Lij_i.reshape(naux, -1).T.conj()
                        Pia = Lij_i = None

        for w in range(nw):
            ec_w = np.linalg.slogdet((np.eye(naux) - Pi[:, :, w]))[1]
            ec_w += np.trace(Pi[:, :, w])
            e_corr += 1.0 / (2.0 * np.pi) * 1.0 / nkpts * ec_w * wts[w]
    comm.Barrier()
    ecorr_gather = comm.gather(e_corr)
    if rank == 0:
        e_corr = np.sum(ecorr_gather)
    comm.Barrier()
    e_corr = comm.bcast(e_corr, root=0)

    return e_corr.real


def _mo_energy_frozen(gw, mo_energy):
    frozen_mask = get_frozen_mask(gw)
    nmo = gw.nmo
    nkpts = gw.nkpts
    mo_energy_frozen = np.zeros((nkpts, nmo))
    for k in range(nkpts):
        mo_energy_frozen[k] = mo_energy[k][frozen_mask[k]]
    return mo_energy_frozen


def _mo_frozen(gw, mo):
    frozen_mask = get_frozen_mask(gw)
    nmo = gw.nmo
    nkpts = gw.nkpts
    nao = mo[0].shape[0]
    mo_frozen = np.zeros((nkpts, nao, nmo), dtype=complex)
    for k in range(nkpts):
        mo_frozen[k] = mo[k][:, frozen_mask[k]]
    return mo_frozen


def _mo_occ_frozen(gw, mo_occ):
    frozen_mask = get_frozen_mask(gw)
    nmo = gw.nmo
    nkpts = gw.nkpts
    mo_occ_frozen = np.zeros((nkpts, nmo))
    for k in range(nkpts):
        mo_occ_frozen[k] = mo_occ[k][frozen_mask[k]]
    return mo_occ_frozen


class KRPA(lib.StreamObject):
    def __init__(self, mf, frozen=None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.frozen = frozen

        # DF-KGW must use GDF integrals
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise NotImplementedError
        self._keys.update(['with_df'])

        ##################################################
        # don't modify the following attributes, they are not input options
        self._nocc = None
        self._nmo = None
        self.kpts = mf.kpts
        self.nkpts = len(self.kpts)
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.e_corr = None
        self.e_hf = None
        self.e_tot = None
        self.outcore = False
        self.segsize = 50

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        log.info('RPA nocc = %d, nvir = %d, nkpts = %d', nocc, nvir, nkpts)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', str(self.frozen))
        return self

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

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None, nw=40):
        """
        Args:
            mo_energy : 2D array (nkpts, nmo), mean-field mo energy
            mo_coeff : 3D array (nkpts, nmo, nmo), mean-field mo coefficient
            nw: integer, grid number

        Returns:
            self.e_tot : RPA total eenrgy
            self.e_hf : EXX energy
            self.e_corr : RPA correlation energy
        """
        if mo_coeff is None:
            mo_coeff = _mo_frozen(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_frozen(self, self._scf.mo_energy)

        cput0 = (time.process_time(), time.perf_counter())
        if rank == 0:
            self.dump_flags()
        self.e_tot, self.e_hf, self.e_corr = kernel(self, mo_energy, mo_coeff, nw=nw, verbose=self.verbose)

        if rank == 0:
            logger.timer(self, 'RPA', *cput0)
        return self.e_tot, self.e_hf, self.e_corr


if __name__ == '__main__':
    from pyscf.pbc import gto, dft, scf
    from pyscf.pbc.lib import chkfile
    import os

    # This test takes a few minutes
    # Test on diamond
    cell = gto.Cell()
    cell.build(
        unit='angstrom',
        a="""
                0.000000     1.783500     1.783500
                1.783500     0.000000     1.783500
                1.783500     1.783500     0.000000
            """,
        atom='C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
        dimension=3,
        max_memory=12000,
        verbose=5,
        pseudo='gth-pbe',
        basis='gth-dzv',
        precision=1e-12,
    )

    kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    gdf = df.GDF(cell, kpts)
    gdf_fname = 'gdf_ints_311.h5'
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'diamond_311.chk'
    if os.path.isfile(chkfname):
        kmf = scf.KRHF(cell, kpts)
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts)
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    rpa = KRPA(kmf)
    rpa.kernel()
    assert abs(rpa.e_corr - -0.1852772037535004) < 1e-6
    assert abs(rpa.e_tot - -10.694392044197565) < 1e-6

    rpa = KRPA(kmf)
    rpa.outcore = True
    rpa.segsize = 2
    rpa.kernel()
    assert abs(rpa.e_corr - -0.1852772037535004) < 1e-6
    assert abs(rpa.e_tot - -10.694392044197565) < 1e-6

    # Test on Na (metallic)
    cell = gto.Cell()
    cell.build(
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

    kpts = cell.make_kpts([2, 2, 1], scaled_center=[0, 0, 0])
    gdf = df.GDF(cell, kpts)
    gdf_fname = 'gdf_ints_221.h5'
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'na_221.chk'
    if os.path.isfile(chkfname):
        kmf = dft.KRKS(cell, kpts)
        kmf = scf.addons.smearing_(kmf, sigma=5e-3, method='fermi')
        kmf.xc = 'lda'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = dft.KRKS(cell, kpts)
        kmf = scf.addons.smearing_(kmf, sigma=5e-3, method='fermi')
        kmf.xc = 'lda'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    rpa = KRPA(kmf)
    rpa.kernel()
    assert abs(rpa.e_corr - -0.04031792880477294) < 1e-6
    assert abs(rpa.e_tot - -47.60693294927457) < 1e-6

    rpa = KRPA(kmf)
    rpa.outcore = True
    rpa.segsize = 3
    rpa.kernel()
    assert abs(rpa.e_corr - -0.04031792880477294) < 1e-6
    assert abs(rpa.e_tot - -47.60693294927457) < 1e-6
