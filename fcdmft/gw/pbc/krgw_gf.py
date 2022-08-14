#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Tianyu Zhu <zhutianyu1991@gmail.com>
#

'''
PBC spin-restricted G0W0 Greens function with k-point sampling
This implementation has N^4 scaling, and is faster than GW-CD (N^4)
and analytic GW (N^6) methods.
GW-AC is recommended for valence states only, and is inaccuarate for core states.

Method:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    Compute Sigma on imaginary frequency with density fitting,
    then analytically continued to real frequency.
    Gaussian density fitting must be used (FFTDF and MDF are not supported).

Note: MPI only works for GW code (DFT should run in serial)
'''

from functools import reduce
import time, os
import numpy
import numpy as np
import h5py
from scipy.optimize import newton, least_squares

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc import df, dft, scf
from pyscf.pbc.mp.kmp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf.pbc.lib import chkfile
from pyscf import __config__

from fcdmft.gw.mol.gw_ac import _get_scaled_legendre_roots, \
        two_pole_fit, two_pole, AC_twopole_diag, thiele, pade_thiele, \
        AC_pade_thiele_diag, AC_twopole_full, AC_pade_thiele_full
from fcdmft.gw.pbc.krgw_ac import KRGWAC, get_rho_response, get_rho_response_metal, \
        get_sigma_diag, get_qij, get_rho_response_wing, \
        get_rho_response_head
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

einsum = lib.einsum

def kernel(gw, gfomega, mo_energy, mo_coeff, orbs=None,
           kptlist=None, writefile=None, nw=None, verbose=logger.NOTE):
    """
    GW Green's function

    Args:
        orbs : a list of orbital indices, default is range(nmo).
        nw : number of frequency point on imaginary axis.
        kptlist: a list of k-points, default is range(nkpts).
        writefile :
                0 - do not save files;
                1 - save vxc, sigma_imag, ac_coeff;
                2 - save gf + 1

    Returns:
        A tuple : gf, gf0, sigma
    """
    mf = gw._scf
    assert(gw.frozen == 0)

    if orbs is None:
        orbs = range(gw.nmo)
    if kptlist is None:
        kptlist = range(gw.nkpts)
    nkpts = gw.nkpts
    nklist = len(kptlist)
    norbs = len(orbs)
    gw.orbs = orbs

    if gw.load_sigma and os.path.isfile('vxc.h5') and os.path.isfile('sigma_imag.h5'):
        fn = 'vxc.h5'
        feri = h5py.File(fn, 'r')
        vk = np.array(feri['vk'])
        v_mf = np.array(feri['v_mf'])
        feri.close()

        fn = 'sigma_imag.h5'
        feri = h5py.File(fn, 'r')
        sigmaI = np.array(feri['sigmaI'])
        omega = np.array(feri['omega'])
        if gw.rdm:
            sigmaI_full = np.array(feri['sigmaI_full'])
        feri.close()

        if gw.rdm:
            gw.sigmaI = sigmaI_full
    elif gw.load_sigma:
        if rank == 0:
            logger.warn(gw, 'No saved files, computing sigma ...')
        gw.load_sigma = False

    # v_xc
    if not gw.load_sigma:
        dm = np.array(mf.make_rdm1())
        v_mf = np.array(mf.get_veff()) - np.array(mf.get_j(dm_kpts=dm))
        for k in range(nkpts):
            v_mf[k] = reduce(numpy.dot, (mo_coeff[k].T.conj(), v_mf[k], mo_coeff[k]))

    nocc = gw.nocc
    nmo = gw.nmo
    nvir = nmo-nocc
    mo_occ = gw.mo_occ

    # v_hf from DFT/HF density
    if not gw.load_sigma:
        dm = np.array(mf.make_rdm1())
        rhf = scf.KRHF(gw.mol, gw.kpts, exxdiv=None)
        if hasattr(gw._scf, 'sigma'):
            rhf = scf.addons.smearing_(rhf, sigma=gw._scf.sigma, method="fermi")
        rhf.with_df = gw.with_df
        rhf.with_df._cderi = gw.with_df._cderi
        vk = rhf.get_veff(gw.mol,dm_kpts=dm) - rhf.get_j(gw.mol,dm_kpts=dm)
        for k in range(nkpts):
            vk[k] = reduce(numpy.dot, (mo_coeff[k].T.conj(), vk[k], mo_coeff[k]))

    mo_occ_1d = np.array(mo_occ).reshape(-1)
    is_metal = False
    if np.linalg.norm(np.abs(mo_occ_1d - 1.) - 1.) > 1e-5:
        # metal must supply a gw.ef=mf.mu by user
        is_metal = True
        if gw.fc:
            if rank == 0:
                logger.warn(gw, 'FC not available for metals - setting gw.fc to False')
            gw.fc = False
        assert(gw.ef)
        ef = gw.ef
    else:
        homo = -99.; lumo = 99.
        for k in range(nkpts):
            if homo < mo_energy[k][nocc-1]:
                homo = mo_energy[k][nocc-1]
            if lumo > mo_energy[k][nocc]:
                lumo = mo_energy[k][nocc]
        ef = (homo+lumo)/2.
    gw.ef = ef

    # finite size correction for exchange self-energy
    if gw.fc and (not gw.load_sigma):
        vk_corr = -2./np.pi * (6.*np.pi**2/gw.mol.vol/nkpts)**(1./3.)
        for k in range(nkpts):
            for i in range(nocc):
                vk[k][i,i] = vk[k][i,i] + vk_corr

    # Grids for integration on imaginary axis
    if is_metal and nw < 400:
        nw = 400
    freqs,wts = _get_scaled_legendre_roots(nw)
    gw.freqs = freqs
    gw.wts = wts

    eta = gw.eta
    nomega = len(gfomega)
    sigma = np.zeros((nkpts,nmo,nmo,nomega),dtype=np.complex128)
    if gw.fullsigma:
        # Compute full self-energy on imaginary axis i*[0,iw_cutoff]
        if not gw.load_sigma:
            sigmaI, omega = get_sigma_full(gw, orbs, kptlist, freqs, wts, iw_cutoff=5.)

        # Analytic continuation
        coeff = None; omega_fit = None
        if rank == 0:
            coeff = []
            if gw.ac == 'twopole':
                for k in range(nklist):
                    coeff.append(AC_twopole_full(sigmaI[k], omega, orbs, nocc))
            elif gw.ac == 'pade':
                if is_metal:
                    nk_pade = 18; ratio_pade = 5./6.
                    for k in range(nklist):
                        coeff_tmp, omega_fit = AC_pade_thiele_full(sigmaI[k], omega,
                                                           npts=nk_pade, step_ratio=ratio_pade)
                        coeff.append(coeff_tmp)
                else:
                    for k in range(nklist):
                        coeff_tmp, omega_fit = AC_pade_thiele_full(sigmaI[k], omega)
                        coeff.append(coeff_tmp)
            coeff = np.array(coeff)
        comm.Barrier()
        coeff = comm.bcast(coeff, root=0)
        omega_fit = comm.bcast(omega_fit, root=0)

        # Compute self-energy on real axis
        for k in range(nklist):
            kn = kptlist[k]
            for p in orbs:
                for q in orbs:
                    if gw.ac == 'twopole':
                        sigma[kn,p,q] = two_pole(gfomega-ef+1j*eta, coeff[k,:,p-orbs[0],q-orbs[0]])
                    elif gw.ac == 'pade':
                        sigma[kn,p,q] = pade_thiele(gfomega-ef+1j*eta, omega_fit, coeff[k,:,p-orbs[0],q-orbs[0]])
                    sigma[kn,p,q] += vk[kn,p,q] - v_mf[kn,p,q]
    else:
        # Compute diagonal self-energy on imaginary axis i*[0,iw_cutoff]
        if not gw.load_sigma:
            sigmaI, omega = get_sigma_diag(gw, orbs, kptlist, freqs, wts, iw_cutoff=5.)

        # Analytic continuation
        coeff = None; omega_fit = None
        if rank == 0:
            coeff = []
            if gw.ac == 'twopole':
                for k in range(nklist):
                    coeff.append(AC_twopole_diag(sigmaI[k], omega, orbs, nocc))
            elif gw.ac == 'pade':
                if is_metal:
                    nk_pade = 18; ratio_pade = 5./6.
                    for k in range(nklist):
                        coeff_tmp, omega_fit = AC_pade_thiele_diag(sigmaI[k], omega,
                                                           npts=nk_pade, step_ratio=ratio_pade)
                        coeff.append(coeff_tmp)
                else:
                    for k in range(nklist):
                        coeff_tmp, omega_fit = AC_pade_thiele_diag(sigmaI[k], omega)
                        coeff.append(coeff_tmp)
            coeff = np.array(coeff)
        comm.Barrier()
        coeff = comm.bcast(coeff, root=0)
        omega_fit = comm.bcast(omega_fit, root=0)

        # Compute self-energy on real axis
        for k in range(nklist):
            kn = kptlist[k]
            for p in orbs:
                if gw.ac == 'twopole':
                    sigma[kn,p,p] = two_pole(gfomega-ef+1j*eta, coeff[k,:,p-orbs[0]])
                elif gw.ac == 'pade':
                    sigma[kn,p,p] = pade_thiele(gfomega-ef+1j*eta, omega_fit, coeff[k,:,p-orbs[0]])
                sigma[kn,p,p] += vk[kn,p,p] - v_mf[kn,p,p]

    if writefile > 0:
        if rank == 0:
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

            fn = 'ac_coeff.h5'
            feri = h5py.File(fn, 'w')
            feri['coeff'] = np.asarray(coeff)
            feri['fermi'] = np.asarray(ef)
            feri['omega_fit'] = np.asarray(omega_fit)
            feri.close()

    gf0 = get_g0_k(gfomega, mf.mo_energy, eta)
    gf = np.zeros_like(gf0)
    for k in range(nkpts):
        for iw in range(nomega):
            gf[k,:,:,iw] = np.linalg.inv(np.linalg.inv(gf0[k,:,:,iw]) - sigma[k,:,:,iw])

    if writefile > 1:
        if rank == 0:
            fn = 'GWGF_sigma_real.h5'
            feri = h5py.File(fn, 'w')
            feri['gf'] = np.asarray(gf)
            feri['sigma'] = np.asarray(sigma)
            feri['gf0'] = np.asarray(gf0)
            feri['gfomega'] = np.asarray(gfomega)
            feri['eta'] = gw.eta
            feri.close()
    comm.Barrier()

    if gw.ev:
        mo_energy = np.zeros_like(np.array(mf.mo_energy))
        for k in range(nklist):
            kn = kptlist[k]
            for p in orbs:
                # self-consistently solve QP equation
                def quasiparticle(omega):
                    if gw.ac == 'twopole':
                        if gw.fullsigma:
                            sigmaR = two_pole(omega-ef, coeff[k,:,p-orbs[0],p-orbs[0]]).real
                        else:
                            sigmaR = two_pole(omega-ef, coeff[k,:,p-orbs[0]]).real
                    elif gw.ac == 'pade':
                        if gw.fullsigma:
                            sigmaR = pade_thiele(omega-ef, omega_fit, coeff[k,:,p-orbs[0],p-orbs[0]]).real
                        else:
                            sigmaR = pade_thiele(omega-ef, omega_fit, coeff[k,:,p-orbs[0]]).real
                    return omega - mf.mo_energy[kn][p] - (sigmaR.real + vk[kn,p,p].real - v_mf[kn,p,p].real)
                try:
                    e = newton(quasiparticle, mf.mo_energy[kn][p], tol=1e-6, maxiter=100)
                    mo_energy[kn,p] = e
                except RuntimeError:
                    conv = False
        gw.mo_energy = mo_energy

    if rank == 0:
        numpy.set_printoptions(threshold=nmo)
        for k in range(nkpts):
            logger.info(gw, '  GW mo_energy @ k%d =\n%s', k,mo_energy[k])
        numpy.set_printoptions(threshold=1000)
    return gf, gf0, sigma


def get_sigma_full(gw, orbs, kptlist, freqs, wts, iw_cutoff=None, max_memory=8000):
    '''
    Compute GW correlation self-energy (full elements) in MO basis
    on imaginary axis
    '''
    mo_energy = np.array(gw._scf.mo_energy)
    mo_coeff = np.array(gw._scf.mo_coeff)
    nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts
    nklist = len(kptlist)
    nw = len(freqs)
    norbs = len(orbs)
    mydf = gw.with_df
    mo_occ = gw.mo_occ

    # possible kpts shift center
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    is_metal = False
    mo_occ_1d = np.array(mo_occ).reshape(-1)
    if np.linalg.norm(np.abs(mo_occ_1d - 1.) - 1.) > 1e-4:
        # metal must supply a gw.ef=mf.mu by user
        is_metal = True
        assert(not gw.fc)
        assert(gw.ef)
        ef = gw.ef
    else:
        homo = -99.; lumo = 99.
        for k in range(nkpts):
            if homo < mo_energy[k][nocc-1]:
                homo = mo_energy[k][nocc-1]
            if lumo > mo_energy[k][nocc]:
                lumo = mo_energy[k][nocc]
        ef = (homo+lumo)/2.
    gw.ef = ef

    # Integration on numerical grids
    if iw_cutoff is not None and (not gw.rdm):
        nw_sigma = sum(freqs < iw_cutoff) + 1
    else:
        nw_sigma = nw + 1
    nw_cutoff = sum(freqs < iw_cutoff) + 1

    omega = np.zeros((nw_sigma),dtype=np.complex128)
    omega[0] = 1j*0.
    omega[1:] = 1j*freqs[:(nw_sigma-1)].copy()

    emo = np.zeros((nkpts,nmo,nw_sigma),dtype=np.complex128)
    for k in range(nkpts):
        emo[k] = omega[None,:] + ef - mo_energy[k][:,None]

    sigma = np.zeros((nklist,norbs,norbs,nw_sigma),dtype=np.complex128)
    if gw.fc:
        # Set up q mesh for q->0 finite size correction
        if not gw.fc_grid:
            q_pts = np.array([1e-3,0,0]).reshape(1,3)
        else:
            Nq = 3
            q_pts = np.zeros((Nq**3-1,3))
            for i in range(Nq):
                for j in range(Nq):
                    for k in range(Nq):
                        if i == 0 and j == 0 and k== 0:
                            continue
                        else:
                            q_pts[i*Nq**2+j*Nq+k-1,0] = k * 5e-4
                            q_pts[i*Nq**2+j*Nq+k-1,1] = j * 5e-4
                            q_pts[i*Nq**2+j*Nq+k-1,2] = i * 5e-4
        nq_pts = len(q_pts)
        q_abs = gw.mol.get_abs_kpts(q_pts)

        # Get qij = 1/sqrt(Omega) * < psi_{ik} | e^{iqr} | psi_{ak-q} > at q: (nkpts, nocc, nvir)
        qij = np.zeros((nq_pts, nkpts, nocc, nmo - nocc),dtype=np.complex128)

        if not gw.fc_grid:
            for k in range(nq_pts):
                qij[k] = get_qij(gw, q_abs[k], mo_coeff)
        else:
            segsize = nq_pts // size
            if rank >= size-(nq_pts-segsize*size):
                start = rank * segsize + rank-(size-(nq_pts-segsize*size))
                stop = min(nq_pts, start+segsize+1)
            else:
                start = rank * segsize
                stop = min(nq_pts, start+segsize)
            for k in range(start, stop):
                qij[k] = get_qij(gw, q_abs[k], mo_coeff)
            comm.Barrier()
            qij_gather = comm.gather(qij)
            if rank == 0:
                for i in range(1, size):
                    qij += qij_gather[i]
            comm.Barrier()
            qij = comm.bcast(qij, root=0)

    segsize = nkpts // size
    if rank >= size-(nkpts-segsize*size):
        start = rank * segsize + rank-(size-(nkpts-segsize*size))
        stop = min(nkpts, start+segsize+1)
    else:
        start = rank * segsize
        stop = min(nkpts, start+segsize)

    for kL in range(start,stop):
        # Lij: (ki, L, i, j) for looping every kL
        Lij = []
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros((nkpts),dtype=np.int64)
        kidx_r = np.zeros((nkpts),dtype=np.int64)
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i
                    logger.debug(gw, "Read Lpq (kL: %s / %s, ki: %s, kj: %s @ Rank %d)"%(kL+1, nkpts, i, j, rank))
                    Lij_out = None
                    # Read (L|pq) and ao2mo transform to (L|ij)
                    Lpq = []
                    for LpqR, LpqI, sign in mydf.sr_loop([kpti, kptj], max_memory=0.1*gw._scf.max_memory, compact=False):
                        Lpq.append(LpqR+LpqI*1.0j)
                    # support uneqaul naux on different k points
                    Lpq = np.vstack(Lpq).reshape(-1,nmo**2)
                    tao = []
                    ao_loc = None
                    moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]
                    Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)
                    Lij.append(Lij_out.reshape(-1,nmo,nmo))
        Lij = np.asarray(Lij)
        naux = Lij.shape[1]

        if kL == 0:
            for w in range(nw):
                # body dielectric matrix eps_body
                if is_metal:
                    Pi = get_rho_response_metal(gw, freqs[w], mo_energy, mo_occ, Lij, kL, kidx)
                else:
                    Pi = get_rho_response(gw, freqs[w], mo_energy, Lij, kL, kidx)
                eps_body_inv = np.linalg.inv(np.eye(naux)-Pi)

                if gw.fc:
                    eps_inv_00 = 0j
                    eps_inv_P0 = np.zeros(naux,dtype=np.complex128)
                    for iq in range(nq_pts):
                        # head dielectric matrix eps_00
                        Pi_00 = get_rho_response_head(gw, freqs[w], mo_energy, qij[iq])
                        eps_00 = 1. - 4. * np.pi/np.linalg.norm(q_abs[iq])**2 * Pi_00
 
                        # wings dielectric matrix eps_P0
                        Pi_P0 = get_rho_response_wing(gw, freqs[w], mo_energy, Lij, qij[iq])
                        eps_P0 = -np.sqrt(4.*np.pi) / np.linalg.norm(q_abs[iq]) * Pi_P0
 
                        # inverse dielectric matrix
                        eps_inv_00 += 1./nq_pts * 1./(eps_00 - np.dot(np.dot(eps_P0.conj(),eps_body_inv),eps_P0))
                        eps_inv_P0 += 1./nq_pts * (-eps_inv_00) * np.dot(eps_body_inv, eps_P0)

                    # head correction
                    Del_00 = 2./np.pi * (6.*np.pi**2/gw.mol.vol/nkpts)**(1./3.) * (eps_inv_00 - 1.)

                eps_inv_PQ = eps_body_inv
                g0 = wts[w] * emo / (emo**2 + freqs[w]**2)
                for k in range(nklist):
                    kn = kptlist[k]
                    # Find km that conserves with kn and kL (-km+kn+kL=G)
                    km = kidx_r[kn]
                    Qmn = einsum('Pmn,PQ->Qmn',Lij[km][:,:,orbs].conj(),eps_inv_PQ-np.eye(naux))
                    Wmn = np.zeros((nmo,norbs,norbs),dtype=np.complex128)
                    for orbm in range(nmo):
                        Wmn[orbm] = 1./nkpts * np.dot(Qmn[:,orbm,:].transpose(),Lij[km][:,orbm,orbs])
                    sigma[k] += -einsum('mnl,mw->nlw',Wmn,g0[km]) / np.pi

                    if gw.fc:
                        # apply head correction to diagonal self-energy
                        assert(kn == km)
                        tmp = -Del_00 * g0[kn][orbs] / np.pi
                        for p in range(norbs):
                            sigma[k][p,p,:] += tmp[p,:]
                        # apply wing correction to diagonal self-energy
                        Wn_P0 = einsum('Pnm,P->nm',Lij[kn],eps_inv_P0).diagonal()
                        Wn_P0 = Wn_P0.real * 2.
                        Del_P0 = np.sqrt(gw.mol.vol/4./np.pi**3) * (6.*np.pi**2/gw.mol.vol/nkpts)**(2./3.) * Wn_P0[orbs]
                        tmp = -einsum('n,nw->nw',Del_P0,g0[kn][orbs]) / np.pi
                        for p in range(norbs):
                            sigma[k][p,p,:] += tmp[p,:]
        else:
            for w in range(nw):
                if is_metal:
                    Pi = get_rho_response_metal(gw, freqs[w], mo_energy, mo_occ, Lij, kL, kidx)
                else:
                    Pi = get_rho_response(gw, freqs[w], mo_energy, Lij, kL, kidx)
                Pi_inv = np.linalg.inv(np.eye(naux)-Pi)-np.eye(naux)
                g0 = wts[w] * emo / (emo**2 + freqs[w]**2)
                for k in range(nklist):
                    kn = kptlist[k]
                    # Find km that conserves with kn and kL (-km+kn+kL=G)
                    km = kidx_r[kn]
                    Qmn = einsum('Pmn,PQ->Qmn',Lij[km][:,:,orbs].conj(),Pi_inv)
                    Wmn = np.zeros((nmo,norbs,norbs),dtype=np.complex128)
                    for orbm in range(nmo):
                        Wmn[orbm] = 1./nkpts * np.dot(Qmn[:,orbm,:].transpose(),Lij[km][:,orbm,orbs])
                    sigma[k] += -einsum('mnl,mw->nlw',Wmn,g0[km])/np.pi

    comm.Barrier()
    fn = 'sigma_mpi_%d.h5'%(rank)
    feri = h5py.File(fn, 'w')
    feri['sigma'] = np.asarray(sigma)
    feri.close()
    comm.Barrier()
    if rank == 0:
        for i in range(1,size):
            fn = 'sigma_mpi_%d.h5'%(i)
            feri = h5py.File(fn, 'r')
            sigma_i = np.asarray(feri['sigma'])
            sigma += sigma_i
            feri.close()
    comm.Barrier()
    os.remove('sigma_mpi_%d.h5'%(rank))

    if gw.rdm:
        gw.sigmaI = sigma

    return sigma[:,:,:,:nw_cutoff], omega[:nw_cutoff]

def get_g0_k(omega, mo_energy, eta):
    nkpts = len(mo_energy)
    nmo = mo_energy[0].shape[0]
    nw = len(omega)
    gf0 = np.zeros((nkpts,nmo,nmo,nw),dtype=np.complex128)
    for k in range(nkpts):
        for iw in range(nw):
            gf0[k,:,:,iw] = np.diag(1.0/(omega[iw]+1j*eta - mo_energy[k]))
    return gf0

def make_rdm1_linear(gw):
    '''
    Linearized GW density matrix (default, conserving)
    Ref: JCTC 17, 2126-2136 (2021)
    '''
    assert(gw.sigmaI is not None)
    assert(gw.rdm and gw.fullsigma)
    sigmaI = gw.sigmaI[:,:,:,1:]
    freqs = 1j * gw.freqs
    wts = gw.wts
    nmo = gw.nmo
    nkpts = gw.nkpts
    nocc = gw.nocc
    if len(gw.orbs) != nmo:
        sigma = np.zeros((nkpts, nmo, nmo, len(freqs)),dtype=sigmaI.dtype)
        for k in range(nkpts):
            for ia,a in enumerate(gw.orbs):
                for ib,b in enumerate(gw.orbs):
                    sigma[k,a,b,:] = sigmaI[k,ia,ib,:]
    else:
        sigma = sigmaI

    # v_xc
    mf = gw._scf
    dm = np.array(mf.make_rdm1())
    v_mf = np.array(mf.get_veff()) - np.array(mf.get_j(dm_kpts=dm))
    for k in range(nkpts):
        v_mf[k] = reduce(numpy.dot, (mf.mo_coeff[k].T.conj(), v_mf[k], mf.mo_coeff[k]))

    # v_hf from DFT/HF density
    rhf = scf.KRHF(gw.mol, gw.kpts, exxdiv=None)
    if hasattr(gw._scf, 'sigma'):
        rhf = scf.addons.smearing_(rhf, sigma=gw._scf.sigma, method="fermi")
    rhf.with_df = gw.with_df
    rhf.with_df._cderi = gw.with_df._cderi
    vk = rhf.get_veff(gw.mol,dm_kpts=dm) - rhf.get_j(gw.mol,dm_kpts=dm)
    for k in range(nkpts):
        vk[k] = reduce(numpy.dot, (mf.mo_coeff[k].T.conj(), vk[k], mf.mo_coeff[k]))

    # finite size correction for exchange self-energy
    if gw.fc:
        vk_corr = -2./np.pi * (6.*np.pi**2/gw.mol.vol/nkpts)**(1./3.)
        for k in range(nkpts):
            for i in range(nocc):
                vk[k][i,i] = vk[k][i,i] + vk_corr

    # Compute GW Green's function on imag freq
    eta= 0.
    gf0 = get_g0_k(freqs, np.array(gw._scf.mo_energy)-gw.ef, eta)
    gf = np.zeros_like(gf0)
    print (gf0.shape, gf.shape, sigma.shape, v_mf.shape)
    for k in range(nkpts):
        for iw in range(len(freqs)):
            gf[k,:,:,iw] = gf0[k,:,:,iw] + np.dot(gf0[k,:,:,iw], (vk[k] + sigma[k,:,:,iw] - v_mf[k])).dot(gf0[k,:,:,iw])

    # GW density matrix
    rdm1 = np.zeros((nkpts, nmo, nmo))
    for k in range(nkpts):
        rdm1[k] = (2./np.pi * einsum('ijw,w->ij',gf[k],wts) + np.eye(nmo)).real
        logger.info(gw, 'GW particle number @ k%d = %s', k, np.trace(rdm1[k]))

    # Symmetrize density matrix
    for k in range(nkpts):
        rdm1[k] = 0.5 * (rdm1[k] + rdm1[k].T)

    return rdm1

class KRGWGF(KRGWAC):

    eta = getattr(__config__, 'krgw_gf_KRGWGF_eta', 5e-3)
    fullsigma = getattr(__config__, 'krgw_gf_KRGWGF_fullsigma', True)
    # analytic continuation: pade or twopole
    ac = getattr(__config__, 'krgw_gf_KRGWGF_ac', 'pade')
    # applying finite size corrections or not
    fc = getattr(__config__, 'krgw_gf_KRGWGF_fc', True)
    fc_grid = getattr(__config__, 'krgw_gf_KRGWGF_fc_grid', False)
    # compute GW-QP eigenvalues or not
    ev = getattr(__config__, 'krgw_gf_KRGWGF_ev', True)

    def __init__(self, mf, frozen=0):
        KRGWAC.__init__(self, mf, frozen=0)
        keys = set(('eta','fullsigma','ac','fc','ev','fc_grid'))
        self._keys = set(self.__dict__.keys()).union(keys)
        self.rdm = False
        self.sigmaI = None
        self.load_sigma = False

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        log.info('GW nocc = %d, nvir = %d, nkpts = %d', nocc, nvir, nkpts)
        if self.frozen > 0:
            log.info('frozen orbitals = %s', str(self.frozen))
        logger.info(self, 'analytic continuation method = %s', self.ac)
        logger.info(self, 'GW finite size corrections = %s', self.fc)
        logger.info(self, 'GW QP eigenvalues = %s', self.ev)
        logger.info(self, 'broadening = %s a.u.', self.eta)
        return self

    @property
    def nocc(self):
        return self.mol.nelectron // 2
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return len(self._scf.mo_energy[0])
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    make_rdm1 = make_rdm1_linear

    def kernel(self, omega, mo_energy=None, mo_coeff=None, orbs=None, kptlist=None, writefile=0, nw=100):
        """
        Args:
            omega : 1D array, frequency points
            mo_energy : 2D array (nkpts, nmo), mean-field mo energy
            mo_coeff : 3D array (nkpts, nmo, nmo), mean-field mo coefficient
            orbs : list, orbital indices
            nw : interger, grid number
            kptlist : list, GW self-energy k-points
            writefile :
                    0 - do not save files;
                    1 - save vxc, sigma_imag, ac_coeff;
                    2 - save gf + 1

        Returns:
            gf : 4D array (nkpts, nmo, nmo, nomega), GW Green's function
            gf0 : 4D array (nkpts, nmo, nmo, nomega), mean-field Green's function
            sigma : 4D array (nkpts, nmo, nmo, nomega), GW self-energy
        """
        if mo_coeff is None:
            mo_coeff = np.array(self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = np.array(self._scf.mo_energy)

        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (2 * nkpts * nmo**2 * naux) * 16/1e6
        mem_now = lib.current_memory()[0]
        if (mem_incore + mem_now > 0.99*self.max_memory):
            if rank == 0:
                logger.warn(self, 'Memory may not enough!')
            raise NotImplementedError

        cput0 = (time.process_time(), time.perf_counter())
        if rank == 0:
            self.dump_flags()
        self.gf, self.gf0, self.sigma = \
                kernel(self, omega, mo_energy, mo_coeff, orbs=orbs,
                       kptlist=kptlist, writefile=writefile, nw=nw, verbose=self.verbose)

        if rank == 0:
            logger.timer(self, 'KRGWGF', *cput0)
        return self.gf, self.gf0, self.sigma


if __name__ == '__main__':
    from pyscf.pbc import gto, dft, scf
    from pyscf.pbc.lib import chkfile
    import os
    # This test takes a few minutes
    # Test on diamond
    cell = gto.Cell()
    cell.build(unit = 'angstrom',
            a = '''
                0.000000     1.783500     1.783500
                1.783500     0.000000     1.783500
                1.783500     1.783500     0.000000
            ''',
            atom = 'C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
            dimension = 3,
            max_memory = 8000,
            verbose = 4,
            pseudo = 'gth-pade',
            basis='gth-szv',
            precision=1e-10)

    kpts = cell.make_kpts([3,1,1],scaled_center=[0,0,0])
    gdf = df.GDF(cell, kpts)
    gdf_fname = 'gdf_ints_311.h5'
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'diamond_311.chk'
    if os.path.isfile(chkfname):
        kmf = dft.KRKS(cell, kpts)
        kmf.xc = 'pbe'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = dft.KRKS(cell, kpts)
        kmf.xc = 'pbe'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    gw = KRGWGF(kmf)
    nmo = len(kmf.mo_energy[0])
    gw.ac = 'pade'
    gw.eta = 1e-2
    gw.fullsigma = True
    gw.fc = True
    gw.rdm = True
    omega = np.linspace(0.2,1.2,101)
    gf, gf0, sigma = gw.kernel(omega=omega, orbs=range(0,nmo), writefile=0)
    if rank == 0:
        for i in range(len(omega)):
            print (omega[i],(-np.trace(gf0[0,:,:,i].imag))/np.pi, \
                         (-np.trace(gf[0,:,:,i].imag))/np.pi)
    assert(abs(-np.trace(gf[0,:,:,0].imag)/np.pi-0.1697064710406389)<1e-3)

    dm = gw.make_rdm1()
    if rank == 0:
        for k in range(len(dm)):
            print (dm[k].diagonal())

    # Test on Na (metallic)
    cell = gto.Cell()
    cell.build(unit = 'angstrom',
           a = '''
           -2.11250000000000   2.11250000000000   2.11250000000000
           2.11250000000000  -2.11250000000000   2.11250000000000
           2.11250000000000   2.11250000000000  -2.11250000000000
           ''',
           atom = '''
           Na   0.00000   0.00000   0.00000
           ''',
           dimension = 3,
           max_memory = 126000,
           verbose = 5,
           pseudo = 'gth-pade',
           basis='gth-dzvp-molopt-sr',
           precision=1e-10)

    kpts = cell.make_kpts([2,2,1],scaled_center=[0,0,0])
    gdf = df.GDF(cell, kpts)
    gdf_fname = 'gdf_ints_221.h5'
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'na_221.chk'
    if os.path.isfile(chkfname):
        kmf = dft.KRKS(cell, kpts)
        kmf = scf.addons.smearing_(kmf, sigma=5e-3, method="fermi")
        kmf.xc = 'lda'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = dft.KRKS(cell, kpts)
        kmf = scf.addons.smearing_(kmf, sigma=5e-3, method="fermi")
        kmf.xc = 'lda'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    gw = KRGWGF(kmf)
    nmo = len(kmf.mo_energy[0])
    gw.ac = 'pade'
    gw.eta = 1e-2
    gw.fullsigma = True
    gw.ef = 0.113280699118
    gw.fc = False
    omega = np.linspace(-0.5,0.5,101)
    gf, gf0, sigma = gw.kernel(omega=omega, orbs=range(0,nmo), writefile=0)
    if rank == 0:
        for i in range(len(omega)):
            print (omega[i],(-np.trace(gf0[0,:,:,i].imag))/np.pi, \
                         (-np.trace(gf[0,:,:,i].imag))/np.pi)
    assert(abs(-np.trace(gf[0,:,:,0].imag)/np.pi-0.05663425306727705)<1e-3)
