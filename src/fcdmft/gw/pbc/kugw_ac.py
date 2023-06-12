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
PBC spin-unrestricted G0W0-AC QP eigenvalues with k-point sampling
This implementation has N^4 scaling, and is faster than GW-CD (N^4)
and analytic GW (N^6) methods.
GW-AC is recommended for valence states only, and is inaccuarate for core states.

Method:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    Compute Sigma on imaginary frequency with density fitting,
    then analytically continued to real frequency.
    Gaussian density fitting must be used (FFTDF and MDF are not supported).
'''

from functools import reduce
import time
import numpy
import numpy as np
import h5py, os
from scipy.optimize import newton, least_squares

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc import df, dft, scf
from pyscf.pbc.cc.kccsd_uhf import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__

from fcdmft.gw.pbc.krgw_ac import KRGWAC
from fcdmft.gw.mol.gw_ac import _get_scaled_legendre_roots, \
        two_pole_fit, two_pole, AC_twopole_diag, thiele, pade_thiele, \
        AC_pade_thiele_diag
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

einsum = lib.einsum

def kernel(gw, mo_energy, mo_coeff, orbs=None,
           kptlist=None, nw=None, verbose=logger.NOTE):
    """
    GW-corrected quasiparticle orbital energies

    Args:
        orbs : a list of orbital indices, default is range(nmo).
        nw : number of frequency point on imaginary axis.
        kptlist: a list of k-points, default is range(nkpts).

    Returns:
        A tuple : converged, mo_energy, mo_coeff
    """
    mf = gw._scf
    assert(gw.frozen == 0)

    nmoa, nmob = gw.nmo
    nocca, noccb = gw.nocc
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    mo_occ = gw.mo_occ

    if orbs is None:
        orbs = range(nmoa)
    if kptlist is None:
        kptlist = range(gw.nkpts)

    nkpts = gw.nkpts
    nklist = len(kptlist)
    norbs = len(orbs)

    # v_xc
    dm = np.array(mf.make_rdm1())
    v_mf = np.array(mf.get_veff())
    vj = np.array(mf.get_j(dm_kpts=dm))
    v_mf[0] = v_mf[0] - (vj[0] + vj[1])
    v_mf[1] = v_mf[1] - (vj[0] + vj[1])
    for s in range(2):
        for k in range(nkpts):
            v_mf[s,k] = reduce(numpy.dot, (mo_coeff[s,k].T.conj(), v_mf[s,k], mo_coeff[s,k]))

    # v_hf from DFT/HF density
    uhf = scf.KUHF(gw.mol, gw.kpts, exxdiv=None)
    uhf.with_df = gw.with_df
    uhf.with_df._cderi = gw.with_df._cderi
    vk = uhf.get_veff(gw.mol,dm_kpts=dm)
    vj = uhf.get_j(gw.mol,dm_kpts=dm)
    vk[0] = vk[0] - (vj[0] + vj[1])
    vk[1] = vk[1] - (vj[0] + vj[1])
    for s in range(2):
        for k in range(nkpts):
            vk[s,k] = reduce(numpy.dot, (mo_coeff[s,k].T.conj(), vk[s,k], mo_coeff[s,k]))

    mo_occ_1d = np.array(mo_occ).reshape(-1)
    is_metal = False
    if np.linalg.norm(np.abs(mo_occ_1d - 0.5) - 0.5) > 1e-5:
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
            if homo < max(mo_energy[0,k][nocca-1],mo_energy[1,k][noccb-1]):
                homo = max(mo_energy[0,k][nocca-1],mo_energy[1,k][noccb-1])
            if lumo > min(mo_energy[0,k][nocca],mo_energy[1,k][noccb]):
                lumo = min(mo_energy[0,k][nocca],mo_energy[1,k][noccb])
        ef = (homo+lumo)/2.

    # finite size correction for exchange self-energy
    if gw.fc:
        vk_corr = -2./np.pi * (6.*np.pi**2/gw.mol.vol/nkpts)**(1./3.)
        for k in range(nkpts):
            for i in range(nocca):
                vk[0][k][i,i] = vk[0][k][i,i] + vk_corr
                vk[1][k][i,i] = vk[1][k][i,i] + vk_corr

    # Grids for integration on imaginary axis
    if is_metal and nw < 200:
        nw = 200
    freqs,wts = _get_scaled_legendre_roots(nw)

    # Compute self-energy on imaginary axis i*[0,iw_cutoff]
    sigmaI, omega = get_sigma_diag(gw, orbs, kptlist, freqs, wts, iw_cutoff=5.)

    # Analytic continuation
    coeff = None; omega_fit = None
    if rank == 0:
        coeff_a = []; coeff_b = []
        if gw.ac == 'twopole':
            for k in range(nklist):
                coeff_a.append(AC_twopole_diag(sigmaI[0,k], omega, orbs, nocca))
                coeff_b.append(AC_twopole_diag(sigmaI[1,k], omega, orbs, noccb))
        elif gw.ac == 'pade':
            if is_metal:
                nk_pade = 18; ratio_pade = 5./6.
                for k in range(nklist):
                    coeff_a_tmp, omega_fit_a = AC_pade_thiele_diag(sigmaI[0,k], omega,
                                                       npts=nk_pade, step_ratio=ratio_pade)
                    coeff_b_tmp, omega_fit_b = AC_pade_thiele_diag(sigmaI[1,k], omega,
                                                       npts=nk_pade, step_ratio=ratio_pade)
                    coeff_a.append(coeff_a_tmp)
                    coeff_b.append(coeff_b_tmp)
            else:
                for k in range(nklist):
                    coeff_a_tmp, omega_fit_a = AC_pade_thiele_diag(sigmaI[0,k], omega)
                    coeff_b_tmp, omega_fit_b = AC_pade_thiele_diag(sigmaI[1,k], omega)
                    coeff_a.append(coeff_a_tmp)
                    coeff_b.append(coeff_b_tmp)
            omega_fit = np.asarray((omega_fit_a, omega_fit_b))
        coeff = np.asarray((coeff_a, coeff_b))
    comm.Barrier()
    coeff = comm.bcast(coeff,root=0)
    omega_fit = comm.bcast(omega_fit,root=0)

    conv = True
    mo_energy = np.zeros_like(np.array(mf.mo_energy))
    for s in range(2):
        for k in range(nklist):
            kn = kptlist[k]
            for p in orbs:
                if gw.linearized:
                    # linearized G0W0
                    de = 1e-6
                    ep = mf.mo_energy[s][kn][p]
                    #TODO: analytic sigma derivative
                    if gw.ac == 'twopole':
                        sigmaR = two_pole(ep-ef, coeff[s,k,:,p-orbs[0]]).real
                        dsigma = two_pole(ep-ef+de, coeff[s,k,:,p-orbs[0]]).real - sigmaR.real
                    elif gw.ac == 'pade':
                        sigmaR = pade_thiele(ep-ef, omega_fit[s], coeff[s,k,:,p-orbs[0]]).real
                        dsigma = pade_thiele(ep-ef+de, omega_fit[s], coeff[s,k,:,p-orbs[0]]).real - sigmaR.real
                    zn = 1.0/(1.0-dsigma/de)
                    e = ep + zn*(sigmaR.real + vk[s,kn,p,p].real - v_mf[s,kn,p,p].real)
                    mo_energy[s,kn,p] = e
                else:
                    # self-consistently solve QP equation
                    def quasiparticle(omega):
                        if gw.ac == 'twopole':
                            sigmaR = two_pole(omega-ef, coeff[s,k,:,p-orbs[0]]).real
                        elif gw.ac == 'pade':
                            sigmaR = pade_thiele(omega-ef, omega_fit[s], coeff[s,k,:,p-orbs[0]]).real
                        return omega - mf.mo_energy[s][kn][p] - (sigmaR.real + vk[s,kn,p,p].real - v_mf[s,kn,p,p].real)
                    try:
                        e = newton(quasiparticle, mf.mo_energy[s][kn][p], tol=1e-6, maxiter=100)
                        mo_energy[s,kn,p] = e
                    except RuntimeError:
                        conv = False
    mo_coeff = mf.mo_coeff

    if rank == 0 and gw.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmoa)
        for k in range(nkpts):
            logger.debug(gw, '  GW mo_energy spin-up @ k%d =\n%s', k,mo_energy[0,k])
        for k in range(nkpts):
            logger.debug(gw, '  GW mo_energy spin-down @ k%d =\n%s', k,mo_energy[1,k])
        numpy.set_printoptions(threshold=1000)

    return conv, mo_energy, mo_coeff

def get_rho_response(gw, omega, mo_energy, Lpq, kL, kidx):
    '''
    Compute density response function in auxiliary basis at freq iw
    '''
    spin, nkpts, naux, nmo, nmo = Lpq.shape
    nocca, noccb = gw.nocc
    kpts = gw.kpts
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    # Compute Pi for kL
    Pi = np.zeros((naux,naux),dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        eia_a = mo_energy[0,i,:nocca,None] - mo_energy[0,a,None,nocca:]
        eia_b = mo_energy[1,i,:noccb,None] - mo_energy[1,a,None,noccb:]
        eia_a = eia_a/(omega**2+eia_a*eia_a)
        eia_b = eia_b/(omega**2+eia_b*eia_b)
        Pia_a = einsum('Pia,ia->Pia',Lpq[0,i][:,:nocca,nocca:],eia_a)
        Pia_b = einsum('Pia,ia->Pia',Lpq[1,i][:,:noccb,noccb:],eia_b)
        # Response from both spin-up and spin-down density
        Pi += 2./nkpts * (einsum('Pia,Qia->PQ',Pia_a,Lpq[0,i][:,:nocca,nocca:].conj()) + \
                          einsum('Pia,Qia->PQ',Pia_b,Lpq[1,i][:,:noccb,noccb:].conj()))
    return Pi

def get_rho_response_metal(gw, omega, mo_energy, mo_occ, Lpq, kL, kidx):
    '''
    Compute density response function in auxiliary basis at freq iw
    '''
    spin, nkpts, naux, nmo, nmo = Lpq.shape
    nocca, noccb = gw.nocc
    kpts = gw.kpts
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    # Compute Pi for kL
    Pi = np.zeros((naux,naux),dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        eia_a = mo_energy[0,i,:,None] - mo_energy[0,a,None,:]
        eia_b = mo_energy[1,i,:,None] - mo_energy[1,a,None,:]
        fia_a = mo_occ[0][i][:,None] - mo_occ[0][a][None,:]
        fia_b = mo_occ[1][i][:,None] - mo_occ[1][a][None,:]
        eia_a = eia_a * fia_a / (omega**2 + eia_a*eia_a)
        eia_b = eia_b * fia_b / (omega**2 + eia_b*eia_b)

        Pia_a = einsum('Pia,ia->Pia',Lpq[0,i],eia_a)
        Pia_b = einsum('Pia,ia->Pia',Lpq[1,i],eia_b)
        # Response from both spin-up and spin-down density
        Pi += 1./nkpts * (einsum('Pia,Qia->PQ',Pia_a,Lpq[0,i].conj()) + \
                          einsum('Pia,Qia->PQ',Pia_b,Lpq[1,i].conj()))
    return Pi

def get_sigma_diag(gw, orbs, kptlist, freqs, wts, iw_cutoff=None, max_memory=8000):
    '''
    Compute GW correlation self-energy (diagonal elements) in MO basis
    on imaginary axis
    '''
    mo_energy = np.array(gw._scf.mo_energy)
    mo_coeff = np.array(gw._scf.mo_coeff)
    nocca, noccb = gw.nocc
    nmoa, nmob = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts
    nklist = len(kptlist)
    nw = len(freqs)
    norbs = len(orbs)
    mydf = gw.with_df
    mo_occ = gw.mo_occ

    # possible kpts shift
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    mo_occ_1d = np.array(mo_occ).reshape(-1)
    is_metal = False
    if np.linalg.norm(np.abs(mo_occ_1d - 0.5) - 0.5) > 1e-5:
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
            if homo < max(mo_energy[0,k][nocca-1],mo_energy[1,k][noccb-1]):
                homo = max(mo_energy[0,k][nocca-1],mo_energy[1,k][noccb-1])
            if lumo > min(mo_energy[0,k][nocca],mo_energy[1,k][noccb]):
                lumo = min(mo_energy[0,k][nocca],mo_energy[1,k][noccb])
        ef = (homo+lumo)/2.

    # Integration on numerical grids
    if iw_cutoff is not None:
        nw_sigma = sum(iw < iw_cutoff for iw in freqs) + 1
    else:
        nw_sigma = nw + 1

    omega = np.zeros((nw_sigma),dtype=np.complex128)
    omega[0] = 1j*0.; omega[1:] = 1j*freqs[:(nw_sigma-1)]
    emo_a = np.zeros((nkpts,nmoa,nw_sigma),dtype=np.complex128)
    emo_b = np.zeros((nkpts,nmob,nw_sigma),dtype=np.complex128)
    for k in range(nkpts):
        emo_a[k] = omega[None,:] + ef - mo_energy[0,k][:,None]
        emo_b[k] = omega[None,:] + ef - mo_energy[1,k][:,None]

    sigma = np.zeros((2,nklist,norbs,nw_sigma),dtype=np.complex128)
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
        qij_a = np.zeros((nq_pts, nkpts, nocca, nmoa - nocca),dtype=np.complex128)
        qij_b = np.zeros((nq_pts, nkpts, noccb, nmob - noccb),dtype=np.complex128)

        if not gw.fc_grid:
            for k in range(nq_pts):
                qij_tmp = get_qij(gw, q_abs[k], mo_coeff)
                qij_a[k] = qij_tmp[0]
                qij_b[k] = qij_tmp[1]
        else:
            segsize = nq_pts // size
            if rank >= size-(nq_pts-segsize*size):
                start = rank * segsize + rank-(size-(nq_pts-segsize*size))
                stop = min(nq_pts, start+segsize+1)
            else:
                start = rank * segsize
                stop = min(nq_pts, start+segsize)
            for k in range(start, stop):
                qij_tmp = get_qij(gw, q_abs[k], mo_coeff)
                qij_a[k] = qij_tmp[0]
                qij_b[k] = qij_tmp[1]
            comm.Barrier()
            qij_a_gather = comm.gather(qij_a)
            qij_b_gather = comm.gather(qij_b)
            if rank == 0:
                for i in range(1, size):
                    qij_a += qij_a_gather[i]
                    qij_b += qij_b_gather[i]
            comm.Barrier()
            qij_a = comm.bcast(qij_a, root=0)
            qij_b = comm.bcast(qij_b, root=0)

    segsize = nkpts // size
    if rank >= size-(nkpts-segsize*size):
        start = rank * segsize + rank-(size-(nkpts-segsize*size))
        stop = min(nkpts, start+segsize+1)
    else:
        start = rank * segsize
        stop = min(nkpts, start+segsize)

    for kL in range(start, stop):
        # Lij: (2, ki, L, i, j) for looping every kL
        #Lij = np.zeros((2,nkpts,naux,nmoa,nmoa),dtype=np.complex128)
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
                    Lij_out_a = None
                    Lij_out_b = None
                    # Read (L|pq) and ao2mo transform to (L|ij)
                    Lpq = []
                    for LpqR, LpqI, sign in mydf.sr_loop([kpti, kptj], max_memory=0.1*gw._scf.max_memory, compact=False):
                        Lpq.append(LpqR+LpqI*1.0j)
                    Lpq = np.vstack(Lpq).reshape(-1,nmoa**2)
                    moija, ijslicea = _conc_mos(mo_coeff[0,i], mo_coeff[0,j])[2:]
                    moijb, ijsliceb = _conc_mos(mo_coeff[1,i], mo_coeff[1,j])[2:]
                    tao = []
                    ao_loc = None
                    Lij_out_a = _ao2mo.r_e2(Lpq, moija, ijslicea, tao, ao_loc, out=Lij_out_a)
                    tao = []
                    ao_loc = None
                    Lij_out_b = _ao2mo.r_e2(Lpq, moijb, ijsliceb, tao, ao_loc, out=Lij_out_b)
                    Lij.append(np.asarray((Lij_out_a.reshape(-1,nmoa,nmoa),Lij_out_b.reshape(-1,nmob,nmob))))

        Lij = np.asarray(Lij)
        Lij = Lij.transpose(1,0,2,3,4)
        naux = Lij.shape[2]

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
                        Pi_00 = get_rho_response_head(gw, freqs[w], mo_energy, (qij_a[iq],qij_b[iq]))
                        eps_00 = 1. - 4. * np.pi/np.linalg.norm(q_abs[iq])**2 * Pi_00
 
                        # wings dielectric matrix eps_P0
                        Pi_P0 = get_rho_response_wing(gw, freqs[w], mo_energy, Lij, (qij_a[iq],qij_b[iq]))
                        eps_P0 = -np.sqrt(4.*np.pi) / np.linalg.norm(q_abs[iq]) * Pi_P0
 
                        # inverse dielectric matrix
                        eps_inv_00 += 1./nq_pts * 1./(eps_00 - np.dot(np.dot(eps_P0.conj(),eps_body_inv),eps_P0))
                        eps_inv_P0 += 1./nq_pts * (-eps_inv_00) * np.dot(eps_body_inv, eps_P0)

                    # head correction
                    Del_00 = 2./np.pi * (6.*np.pi**2/gw.mol.vol/nkpts)**(1./3.) * (eps_inv_00 - 1.)

                eps_inv_PQ = eps_body_inv
                g0_a = wts[w] * emo_a / (emo_a**2 + freqs[w]**2)
                g0_b = wts[w] * emo_b / (emo_b**2 + freqs[w]**2)
                for k in range(nklist):
                    kn = kptlist[k]
                    # Find km that conserves with kn and kL (-km+kn+kL=G)
                    km = kidx_r[kn]
                    Qmn_a = einsum('Pmn,PQ->Qmn',Lij[0,km][:,:,orbs].conj(),eps_inv_PQ-np.eye(naux))
                    Qmn_b = einsum('Pmn,PQ->Qmn',Lij[1,km][:,:,orbs].conj(),eps_inv_PQ-np.eye(naux))
                    Wmn_a = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn_a,Lij[0,km][:,:,orbs])
                    Wmn_b = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn_b,Lij[1,km][:,:,orbs])
                    sigma[0,k] += -einsum('mn,mw->nw',Wmn_a,g0_a[km]) / np.pi
                    sigma[1,k] += -einsum('mn,mw->nw',Wmn_b,g0_b[km]) / np.pi

                    if gw.fc:
                        # apply head correction
                        assert(kn == km)
                        sigma[0,k] += -Del_00 * g0_a[kn][orbs] / np.pi
                        sigma[1,k] += -Del_00 * g0_b[kn][orbs] / np.pi
                        # apply wing correction
                        Wn_P0_a = einsum('Pnm,P->nm',Lij[0,kn],eps_inv_P0).diagonal()
                        Wn_P0_b = einsum('Pnm,P->nm',Lij[1,kn],eps_inv_P0).diagonal()
                        Wn_P0_a = Wn_P0_a.real * 2.
                        Wn_P0_b = Wn_P0_b.real * 2.
                        Del_P0_a = np.sqrt(gw.mol.vol/4./np.pi**3) * (6.*np.pi**2/gw.mol.vol/nkpts)**(2./3.) * Wn_P0_a[orbs]
                        Del_P0_b = np.sqrt(gw.mol.vol/4./np.pi**3) * (6.*np.pi**2/gw.mol.vol/nkpts)**(2./3.) * Wn_P0_b[orbs]
                        sigma[0,k] += -einsum('n,nw->nw',Del_P0_a,g0_a[kn][orbs]) / np.pi
                        sigma[1,k] += -einsum('n,nw->nw',Del_P0_b,g0_b[kn][orbs]) / np.pi
        else:
            for w in range(nw):
                if is_metal:
                    Pi = get_rho_response_metal(gw, freqs[w], mo_energy, mo_occ, Lij, kL, kidx)
                else:
                    Pi = get_rho_response(gw, freqs[w], mo_energy, Lij, kL, kidx)
                Pi_inv = np.linalg.inv(np.eye(naux)-Pi)-np.eye(naux)
                g0_a = wts[w] * emo_a / (emo_a**2 + freqs[w]**2)
                g0_b = wts[w] * emo_b / (emo_b**2 + freqs[w]**2)
                for k in range(nklist):
                    kn = kptlist[k]
                    # Find km that conserves with kn and kL (-km+kn+kL=G)
                    km = kidx_r[kn]
                    Qmn_a = einsum('Pmn,PQ->Qmn',Lij[0,km][:,:,orbs].conj(),Pi_inv)
                    Qmn_b = einsum('Pmn,PQ->Qmn',Lij[1,km][:,:,orbs].conj(),Pi_inv)
                    Wmn_a = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn_a,Lij[0,km][:,:,orbs])
                    Wmn_b = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn_b,Lij[1,km][:,:,orbs])
                    sigma[0,k] += -einsum('mn,mw->nw',Wmn_a,g0_a[km]) / np.pi
                    sigma[1,k] += -einsum('mn,mw->nw',Wmn_b,g0_b[km]) / np.pi

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

    return sigma, omega

def get_rho_response_head(gw, omega, mo_energy, qij):
    '''
    Compute head (G=0, G'=0) density response function in auxiliary basis at freq iw
    '''
    qij_a, qij_b = qij
    nocca, noccb = gw.nocc
    kpts = gw.kpts
    nkpts = len(kpts)

    # Compute Pi head
    Pi_00 = 0j
    for i, kpti in enumerate(kpts):
        eia_a = mo_energy[0,i,:nocca,None] - mo_energy[0,i,None,nocca:]
        eia_b = mo_energy[1,i,:noccb,None] - mo_energy[1,i,None,noccb:]
        eia_a = eia_a/(omega**2+eia_a*eia_a)
        eia_b = eia_b/(omega**2+eia_b*eia_b)
        Pi_00 += 2./nkpts * (einsum('ia,ia->',eia_a,qij_a[i].conj()*qij_a[i]) + \
                        einsum('ia,ia->',eia_b,qij_b[i].conj()*qij_b[i]))
    return Pi_00

def get_rho_response_wing(gw, omega, mo_energy, Lpq, qij):
    '''
    Compute wing (G=P, G'=0) density response function in auxiliary basis at freq iw
    '''
    qij_a, qij_b = qij
    spin, nkpts, naux, nmo, nmo = Lpq.shape
    nocca, noccb = gw.nocc
    kpts = gw.kpts
    nkpts = len(kpts)

    # Compute Pi wing
    Pi = np.zeros(naux,dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        eia_a = mo_energy[0,i,:nocca,None] - mo_energy[0,i,None,nocca:]
        eia_b = mo_energy[1,i,:noccb,None] - mo_energy[1,i,None,noccb:]
        eia_a = eia_a/(omega**2+eia_a*eia_a)
        eia_b = eia_b/(omega**2+eia_b*eia_b)
        eia_q_a = eia_a * qij_a[i].conj()
        eia_q_b = eia_b * qij_b[i].conj()
        Pi += 2./nkpts * (einsum('Pia,ia->P',Lpq[0,i][:,:nocca,nocca:],eia_q_a) + \
                          einsum('Pia,ia->P',Lpq[1,i][:,:noccb,noccb:],eia_q_b))
    return Pi

def get_qij(gw, q, mo_coeff, uniform_grids=False):
    '''
    Compute qij = 1/Omega * |< psi_{ik} | e^{iqr} | psi_{ak-q} >|^2 at q: (nkpts, nocc, nvir)
    through kp perturbtation theory
    Ref: Phys. Rev. B 83, 245122 (2011)
    '''
    nocca, noccb = gw.nocc
    nmoa, nmob = gw.nmo
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    kpts = gw.kpts
    nkpts = len(kpts)
    cell = gw.mol
    mo_energy = np.asarray(gw._scf.mo_energy)

    if uniform_grids:
        mydf = df.FFTDF(cell, kpts=kpts)
        coords = cell.gen_uniform_grids(mydf.mesh)
    else:
        coords, weights = dft.gen_grid.get_becke_grids(cell,level=4)
    ngrid = len(coords)

    qij_a = np.zeros((nkpts,nocca,nvira),dtype=np.complex128)
    qij_b = np.zeros((nkpts,noccb,nvirb),dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        ao_p = dft.numint.eval_ao(cell, coords, kpt=kpti, deriv=1)
        ao = ao_p[0]
        ao_grad = ao_p[1:4]
        if uniform_grids:
            ao_ao_grad = einsum('mg,xgn->xmn',ao.T.conj(),ao_grad) * cell.vol / ngrid
        else:
            ao_ao_grad = einsum('g,mg,xgn->xmn',weights,ao.T.conj(),ao_grad)
        q_ao_ao_grad = -1j * einsum('x,xmn->mn',q,ao_ao_grad)
        q_mo_mo_grad_a = np.dot(np.dot(mo_coeff[0,i][:,:nocca].T.conj(), q_ao_ao_grad), mo_coeff[0,i][:,nocca:])
        q_mo_mo_grad_b = np.dot(np.dot(mo_coeff[1,i][:,:noccb].T.conj(), q_ao_ao_grad), mo_coeff[1,i][:,noccb:])
        enm_a = 1./(mo_energy[0,i][nocca:,None] - mo_energy[0,i][None,:nocca])
        enm_b = 1./(mo_energy[1,i][noccb:,None] - mo_energy[1,i][None,:noccb])
        dens_a = enm_a.T * q_mo_mo_grad_a
        dens_b = enm_b.T * q_mo_mo_grad_b
        qij_a[i] = dens_a / np.sqrt(cell.vol)
        qij_b[i] = dens_b / np.sqrt(cell.vol)

    return (qij_a, qij_b)


class KUGWAC(KRGWAC):

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira = nmoa - nocca
        nvirb = nmob - noccb
        nkpts = self.nkpts
        log.info('GW (nocca, noccb) = (%d, %d), (nvira, nvirb) = (%d, %d), nkpts = %d',
                 nocca, noccb, nvira, nvirb, nkpts)
        if self.frozen > 0:
            log.info('frozen orbitals %s', str(self.frozen))
        logger.info(self, 'use perturbative linearized QP eqn = %s', self.linearized)
        logger.info(self, 'analytic continuation method = %s', self.ac)
        logger.info(self, 'GW finite size corrections = %s', self.fc)
        return self

    @property
    def nocc(self):
        mo_occ = self._scf.mo_occ
        return (int(np.sum(mo_occ[0][0])), int(np.sum(mo_occ[1][0])))
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return (len(self._scf.mo_energy[0][0]), len(self._scf.mo_energy[1][0]))
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None, orbs=None, kptlist=None, nw=100):
        """
        Args:
            mo_energy : 3D array (2, nkpts, nmo), mean-field mo energy
            mo_coeff : 4D array (2, nkpts, nmo, nmo), mean-field mo coefficient
            orbs: list, orbital indices
            nw: interger, grid number
            kptlist: list, GW self-energy k-points

        Returns:
            self.mo_energy : GW quasiparticle energy
        """
        if mo_coeff is None:
            mo_coeff = np.array(self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = np.array(self._scf.mo_energy)

        nmoa, nmob = self.nmo
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (3*nkpts*nmoa**2*naux) * 16/1e6
        mem_now = lib.current_memory()[0]
        if (mem_incore + mem_now > 0.99*self.max_memory):
            if rank == 0:
                logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError

        cput0 = (time.process_time(), time.perf_counter())
        if rank == 0:
            self.dump_flags()
        self.converged, self.mo_energy, self.mo_coeff = \
                kernel(self, mo_energy, mo_coeff, orbs=orbs,
                       kptlist=kptlist, nw=nw, verbose=self.verbose)

        if rank == 0:
            logger.warn(self, 'GW QP energies may not be sorted from min to max')
            logger.timer(self, 'GW', *cput0)
        return self.mo_energy

if __name__ == '__main__':
    from pyscf.pbc import gto, dft, scf
    from pyscf.pbc.lib import chkfile
    import os
    cell = gto.Cell()
    cell.build(
        unit = 'B',
        a = [[ 0.,          6.74027466,  6.74027466],
             [ 6.74027466,  0.,          6.74027466],
             [ 6.74027466,  6.74027466,  0.        ]],
        atom = '''H 0 0 0
                  H 1.68506866 1.68506866 1.68506866
                  H 3.37013733 3.37013733 3.37013733''',
        basis = 'gth-dzvp',
        pseudo = 'gth-pade',
        verbose = 5,
        charge = 0,
        spin = 1)

    cell.spin = cell.spin * 3
    kpts = cell.make_kpts([3,1,1],scaled_center=[0,0,0])
    gdf = df.GDF(cell, kpts)
    gdf_fname = 'h3_ints_311.h5'
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'h_311.chk'
    if os.path.isfile(chkfname):
        kmf = scf.KUHF(cell, kpts, exxdiv='ewald')
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KUHF(cell, kpts, exxdiv='ewald')
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    gw = KUGWAC(kmf)
    gw.linearized = False
    gw.ac = 'pade'
    gw.fc = False
    nocca, noccb = gw.nocc
    gw.kernel(kptlist=[0,1,2],orbs=range(0,nocca+3))
    assert((abs(gw.mo_energy[0][0][nocca-1]--0.28661016))<1e-5)
    assert((abs(gw.mo_energy[0][0][nocca]-0.13952572))<1e-5)
    assert((abs(gw.mo_energy[1][1][noccb-1]--0.34174199))<1e-5)
    assert((abs(gw.mo_energy[1][1][noccb]-0.0829626))<1e-5)

    gw.fc = True
    nocca, noccb = gw.nocc
    gw.kernel(kptlist=[0,1,2],orbs=range(0,nocca+3))
    assert((abs(gw.mo_energy[0][0][nocca-1]--0.48063839))<1e-5)
    assert((abs(gw.mo_energy[0][0][nocca]-0.13870787))<1e-5)
    assert((abs(gw.mo_energy[1][1][noccb-1]--0.53502818))<1e-5)
    assert((abs(gw.mo_energy[1][1][noccb]--0.11519831))<1e-5)

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
        kmf = dft.KUKS(cell, kpts)
        kmf = scf.addons.smearing_(kmf, sigma=5e-3, method="fermi")
        kmf.xc = 'lda'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = dft.KUKS(cell, kpts)
        kmf = scf.addons.smearing_(kmf, sigma=5e-3, method="fermi")
        kmf.xc = 'lda'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    gw = KUGWAC(kmf)
    gw.linearized = False
    gw.ac = 'pade'
    # gw.ef is obtained from DFT calc (mu)
    gw.ef = 0.113280699118
    # without finite size corrections
    gw.fc = False
    nocca, noccb = gw.nocc
    gw.kernel(kptlist=[0,1,2],orbs=range(1,nocca+3))
    if rank == 0:
        print (gw.mo_energy)
    assert((abs(gw.mo_energy[0][0][3]--0.95982529))<1e-5)
    assert((abs(gw.mo_energy[0][0][4]-0.04583828))<1e-5)
