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

"""
Spin-restricted G0W0 double counting self-energy term in GW+DMFT

Method:
    T. Zhu and G.K.-L. Chan, Phys. Rev. X 11, 021006 (2021)
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    Compute polarizability on imaginary time, then transform to imag frequency
    to compute self-energy, then analytically continued to real frequency

Other useful references:
    J. Phys. Chem. Lett. 9, 306 (2018)
    Phys. Rev. B 94, 165109 (2016)
"""

import time, h5py
from functools import reduce
import numpy
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import gto, df, dft, scf
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__
from fcdmft.gw.mol.gw_ac import _get_scaled_legendre_roots, \
        two_pole, pade_thiele, GWAC, AC_twopole_full, AC_pade_thiele_full

einsum = lib.einsum

def kernel(gw, gfomega, Lpq=None, kmf=None, C_mo_lo=None, orbs=None,
           nw=None, nt=None, verbose=logger.NOTE, small_mem=False):
    '''
    Returns:
       sigma : GW double counting self-energy on real axis
    '''
    mf = gw._scf
    assert(gw.frozen == 0)

    if orbs is None:
        orbs = range(gw.nmo)
    norbs = len(orbs)

    nmo = gw.nmo
    nocc = gw.nocc
    assert(gw.ef)
    ef = gw.ef

    # Imaginary frequency grids
    freqs, wts_w = _get_scaled_legendre_roots(nw)

    # Imaginary time grids
    time, wts_t = _get_scaled_legendre_roots(nt, x0=1.0)

    eta = gw.eta
    nomega = len(gfomega)
    sigma = np.zeros((nmo,nmo,nomega),dtype=np.complex128)

    # Compute full self-energy on imaginary axis i*[0,iw_cutoff]
    sigmaI, omega = get_sigma_full(gw, orbs, Lpq, kmf, C_mo_lo,
                                  freqs, wts_w, time, wts_t, iw_cutoff=5., small_mem=small_mem)
    fn = 'gw_dc_sigmaI.h5'
    feri = h5py.File(fn, 'w')
    feri['sigmaI'] = np.asarray(sigmaI)
    feri['omega'] = np.asarray(omega)
    feri.close()

    # Analytic continuation
    if gw.ac == 'twopole':
        coeff = AC_twopole_full(sigmaI, omega, orbs, nocc)
    elif gw.ac == 'pade':
        coeff, omega_fit = AC_pade_thiele_full(sigmaI, omega, npts=18, step_ratio=2.5/3.0)

    # Compute real-axis self-energy
    for p in orbs:
        for q in orbs:
            if gw.ac == 'twopole':
                sigma[p,q] = two_pole(gfomega-ef+1j*eta, coeff[:,p-orbs[0],q-orbs[0]])
            elif gw.ac == 'pade':
                sigma[p,q] = pade_thiele(gfomega-ef+1j*eta, omega_fit, coeff[:,p-orbs[0],q-orbs[0]])

    fn = 'imp_ac_coeff.h5'
    feri = h5py.File(fn, 'w')
    feri['coeff'] = np.asarray(coeff)
    feri['fermi'] = np.asarray(ef)
    feri['omega_fit'] = np.asarray(omega_fit)
    feri.close()

    return sigma

def get_sigma_full(gw, orbs, Lpq, kmf, C_mo_lo, freqs, wts_w, time, wts_t, iw_cutoff=None, small_mem=False):
    '''
    Compute GW correlation self-energy (all elements) in LO basis
    on imaginary axis
    '''
    nmo = gw.nmo
    nw = len(freqs)
    naux = Lpq.shape[0]
    norbs = len(orbs)
    nocc = gw.nocc
    ef = gw.ef

    print('### computing polarization in imag time and freq domain ###')
    tchunk = 100
    n_tchunk = len(time) // tchunk
    tlist = []; wtslist = []
    for i in range(n_tchunk):
        tlist.append(time[i*tchunk:(i+1)*tchunk])
        wtslist.append(wts_t[i*tchunk:(i+1)*tchunk])
    if len(time) % tchunk != 0:
        tlist.append(time[n_tchunk*tchunk:])
        wtslist.append(wts_t[n_tchunk*tchunk:])
        n_tchunk += 1

    # Integration on numerical grids
    if iw_cutoff is not None:
        nw_sigma = sum(iw < iw_cutoff for iw in freqs) + 1
    else:
        nw_sigma = nw + 1

    omega = np.zeros((nw_sigma),dtype=np.complex128)
    omega[0] = 1j*0.
    omega[1:] = 1j*freqs[:(nw_sigma-1)]

    # Compute time-domain density response kernel and transform to freq domain
    Pi = np.zeros((naux,naux,nw),dtype=np.complex128)
    for i in range(n_tchunk):
        Pi_t = get_response_t(kmf, nocc, C_mo_lo, tlist[i], Lpq, ef)
        for w in range(nw):
            Pi[:,:,w] += CT_t_to_w(Pi_t, tlist[i], wtslist[i], freqs[w])

    print('### computing self-energy in imag freq domain ###')
    sigma = np.zeros((norbs,norbs,nw_sigma),dtype=np.complex128)
    if small_mem:    
        for w in range(nw):
            Pi_inv = np.linalg.inv(np.eye(naux)-Pi[:,:,w])-np.eye(naux)
            Qnu = einsum('Pnu,PQ->Qnu',Lpq[:,orbs,:],Pi_inv)
            g0_pos = get_g0_w_from_kmf(kmf, C_mo_lo, ef, omega+1j*freqs[w])
            g0_neg = get_g0_w_from_kmf(kmf, C_mo_lo, ef, omega-1j*freqs[w])
            for w2 in range(nw_sigma):
                Qnv = -wts_w[w] * einsum('Qnu,uv->Qnv',Qnu,g0_pos[:,:,w2]+g0_neg[:,:,w2]) / 2. / np.pi
                sigma[:,:,w2] += einsum('Qnv,Qvl->nl',Qnv,Lpq[:,:,orbs])
    else:
        Qnvw = np.zeros((naux,norbs,nmo,nw_sigma),dtype=np.complex128)
        for w in range(nw):
            Pi_inv = np.linalg.inv(np.eye(naux)-Pi[:,:,w])-np.eye(naux)
            Qnu = einsum('Pnu,PQ->Qnu',Lpq[:,orbs,:],Pi_inv)
            g0_pos = get_g0_w_from_kmf(kmf, C_mo_lo, ef, omega+1j*freqs[w])
            g0_neg = get_g0_w_from_kmf(kmf, C_mo_lo, ef, omega-1j*freqs[w])
            Qnvw += -wts_w[w] * einsum('Qnu,uvw->Qnvw',Qnu,g0_pos+g0_neg) / 2. / np.pi
        sigma = einsum('Qnvw,Qvl->nlw',Qnvw,Lpq[:,:,orbs])

    return sigma, omega

def get_response_t(kmf, nocc, C_mo_lo, time, Lpq, mu):
    """
    Compute polarization in time domain
    """
    mo_occ = [x/2. for x in kmf.mo_occ]
    nkpts = len(mo_occ)
    nmo = len(kmf.mo_energy[0])
    # Approximate metal as gapped system
    # TODO: exact treatment of metal
    for k in range(nkpts):
        for i in range(nmo):
            if mo_occ[k][i] > 0.5:
                mo_occ[k][i] = 1.0
            else:
                mo_occ[k][i] = 0.0

    nt = len(time)
    naux = Lpq.shape[0]
    PQ = np.zeros((naux,naux,nt))

    for t in range(nt):
        g0_pos = get_g0_t_from_kmf(kmf, mo_occ, C_mo_lo, mu, time[t])
        g0_neg = get_g0_t_from_kmf(kmf, mo_occ, C_mo_lo, mu, -time[t])
        Pkj = einsum('Pij,ki->Pkj',Lpq,g0_pos)
        Qkj = einsum('Qkl,lj->Qkj',Lpq,g0_neg)
        PQ[:,:,t] = 2. * einsum('Pkj,Qkj->PQ',Pkj,Qkj)
    return PQ

def get_g0_t_from_kmf(kmf, mo_occ, C_mo_lo, mu, time):
    """
    Get impurity imaginary-time mean-field Green's function in LO basis
    """
    mo_energy = np.asarray(kmf.mo_energy)
    nlo = C_mo_lo.shape[-1]
    nkpts, nmo = mo_energy.shape
    g0 = np.zeros((nkpts,nmo,nmo))
    for k in range(nkpts):
        if time < 0.:
            idx = np.where(mo_occ[k] > 0.99)[0]
            g0[k,idx,idx] = np.exp(-(mo_energy[k][idx]-mu)*time) * mo_occ[k][idx]
        else:
            idx = np.where(mo_occ[k] < 0.01)[0]
            g0[k,idx,idx] = -np.exp(-(mo_energy[k][idx]-mu)*time) * (1.-mo_occ[k][idx])

    g0_lo = np.zeros((nkpts,nlo,nlo),dtype=np.complex128)
    for k in range(nkpts):
        g0_lo[k,:,:] = np.dot(np.dot(C_mo_lo[0,k].T.conj(), g0[k,:,:]), C_mo_lo[0,k])

    g0_imp = g0_lo.sum(axis=0)/nkpts

    return g0_imp.real

def get_g0_w_from_kmf(kmf, C_mo_lo, mu, freqs):
    """
    Get impurity imaginary-freq mean-field Green's function in LO basis
    """
    mo_energy = np.asarray(kmf.mo_energy)
    nlo = C_mo_lo.shape[-1]
    nkpts, nmo = mo_energy.shape
    nw = len(freqs)
    g0 = np.zeros((nkpts,nmo,nmo,nw),np.complex128)
    for iw in range(nw):
        for k in range(nkpts):
            g0[k,:,:,iw] = np.diag(1./(freqs[iw]+mu-mo_energy[k]))

    g0_lo = np.zeros((nkpts,nlo,nlo,nw),np.complex128)
    for iw in range(nw):
        for k in range(nkpts):
            g0_lo[k,:,:,iw] = reduce(numpy.dot, (C_mo_lo[0,k].T.conj(), g0[k,:,:,iw], C_mo_lo[0,k]))

    g0_imp = g0_lo.sum(axis=0)/nkpts

    return g0_imp

def CT_t_to_w(Gt, time, wts, omega):
    """
    Cosine transform of even function from time to frequency
    """
    cos_tw = einsum('t,t->t',np.cos(time*omega),wts)
    Gw = 2. * einsum('PQt,t->PQ',Gt,cos_tw)
    return Gw


class GWGF(GWAC):

    eta = getattr(__config__, 'gw_dc_GWGF_eta', 5e-3)
    # Analytic continuation: pade or twopole
    ac = getattr(__config__, 'gw_dc_GWGF_ac', 'pade')

    def __init__(self, mf, frozen=0):
        GWAC.__init__(self, mf, frozen=0)
        #TODO: implement frozen orbs
        if frozen > 0:
            raise NotImplementedError
        self.frozen = frozen
        self.ef = None

        keys = set(('eta','ac'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen > 0:
            log.info('frozen orbitals = %d', self.frozen)
        logger.info(self, 'analytic continuation method = %s', self.ac)
        return self

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, omega, Lpq=None, kmf=None, C_mo_lo=None, orbs=None, nw=100, nt=2000, small_mem=False):
        """
        Args:
            omega : 1D array (nomega), real freq
            kmf : PBC mean-field class
            Lpq : 3D array (naux, nlo, nlo), 3-index ERI
            C_mo_lo: 4D array (spin, nkpts, nmo, nlo), transformation matrix MO -> LO
            orbs: list, orbital indices
            nw: interger, grid number

        Returns:
            self.sigma : 3D array (nlo, nlo, nomega), GW self-energy (double counting)
        """
        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()

        naux, nao, nao = Lpq.shape
        mem_incore = (2*nao**2*naux*nw + naux**2*100) * 8 / 1e6
        mem_now = lib.current_memory()[0]
        if (mem_incore + mem_now > 0.95 * self.max_memory):
            mem_incore_small = (nao**2*naux + naux**2*100) * 16 / 1e6
            if (mem_incore_small + mem_now < 0.95 * self.max_memory):
                small_mem = True
            else:
                logger.warn(self, 'Memory may not be enough, even with the small memory option!')
                raise NotImplementedError

        self.sigma = kernel(self, omega, Lpq=Lpq, kmf=kmf,
                   C_mo_lo=C_mo_lo, orbs=orbs, nw=nw, nt=nt, verbose=self.verbose, small_mem=small_mem)

        logger.timer(self, 'GWGF', *cput0)
        return self.sigma
