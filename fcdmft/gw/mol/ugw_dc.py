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
Spin-unrestricted G0W0 double counting self-energy term in GW+DMFT

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
from pyscf.mp.ump2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__
from fcdmft.gw.mol.gw_ac import _get_scaled_legendre_roots, \
        two_pole, pade_thiele, AC_twopole_full, AC_pade_thiele_full
from fcdmft.gw.mol.ugw_ac import UGWAC
from fcdmft.gw.mol.gw_dc import CT_t_to_w

einsum = lib.einsum

def kernel(gw, gfomega, Lpq=None, kmf=None, C_mo_lo=None, orbs=None,
           nw=None, nt=None, verbose=logger.NOTE, small_mem=False):
    '''
    Returns:
       sigma : GW double counting self-energy on real axis
    '''
    mf = gw._scf
    assert(gw.frozen == 0)

    nocca, noccb = gw.nocc
    nmoa, nmob = gw.nmo
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    assert(gw.ef)
    ef = gw.ef

    if orbs is None:
        orbs = range(nmoa)

    # Imaginary frequency grids
    freqs, wts_w = _get_scaled_legendre_roots(nw)

    # Imaginary time grids
    time, wts_t = _get_scaled_legendre_roots(nt, x0=1.0)

    eta = gw.eta
    nomega = len(gfomega)
    sigma = np.zeros((2,nmoa,nmoa,nomega),dtype=np.complex128)

    # Compute full self-energy on imaginary axis i*[0,iw_cutoff]
    sigmaI,omega = get_sigma_full(gw, orbs, Lpq, kmf, C_mo_lo,
                                  freqs, wts_w, time, wts_t, iw_cutoff=5., small_mem=small_mem)
    fn = 'gw_dc_sigmaI.h5'
    feri = h5py.File(fn, 'w')
    feri['sigmaI'] = np.asarray(sigmaI)
    feri['omega'] = np.asarray(omega)
    feri.close()

    # Analytic continuation
    if gw.ac == 'twopole':
        coeff_a = AC_twopole_full(sigmaI[0], omega, orbs, nocca)
        coeff_b = AC_twopole_full(sigmaI[1], omega, orbs, noccb)
    elif gw.ac == 'pade':
        coeff_a, omega_fit_a = AC_pade_thiele_full(sigmaI[0], omega, npts=18, step_ratio=2.5/3.0)
        coeff_b, omega_fit_b = AC_pade_thiele_full(sigmaI[1], omega, npts=18, step_ratio=2.5/3.0)
        omega_fit = omega_fit_a
    coeff = np.asarray((coeff_a,coeff_b))

    # Compute real-axis self-energy
    for s in range(2):
        for p in orbs:
            for q in orbs:
                if gw.ac == 'twopole':
                    sigma[s,p,q] = two_pole(gfomega-ef+1j*eta, coeff[s,:,p-orbs[0],q-orbs[0]])
                elif gw.ac == 'pade':
                    sigma[s,p,q] = pade_thiele(gfomega-ef+1j*eta, omega_fit, coeff[s,:,p-orbs[0],q-orbs[0]])

    fn = 'imp_ac_coeff.h5'
    feri = h5py.File(fn, 'w')
    feri['coeff'] = np.asarray(coeff)
    feri['fermi'] = np.asarray(ef)
    feri['omega_fit'] = np.asarray(omega_fit)
    feri.close()

    return sigma

def get_sigma_full(gw, orbs, Lpq, kmf, C_mo_lo, freqs, wts_w, time, wts_t, iw_cutoff=None, small_mem=False):
    '''
    Compute GW correlation self-energy (all elements) in AO basis
    on imaginary axis
    '''
    nocca, noccb = gw.nocc
    nmoa, nmob = gw.nmo
    nw = len(freqs)
    naux = Lpq.shape[0]
    norbs = len(orbs)
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
        Pi_t = get_response_t(kmf, gw.nocc, C_mo_lo, tlist[i], Lpq, ef)
        for w in range(nw):
            Pi[:,:,w] += CT_t_to_w(Pi_t, tlist[i], wtslist[i], freqs[w])

    print('### computing self-energy in imag freq domain ###')
    sigma = np.zeros((2,norbs,norbs,nw_sigma),dtype=np.complex128)
    if small_mem:
        for w in range(nw):
            Pi_inv = np.linalg.inv(np.eye(naux)-Pi[:,:,w])-np.eye(naux)
            Qnu = einsum('Pnu,PQ->Qnu',Lpq[:,orbs,:],Pi_inv)
            g0_pos = get_g0_w_from_kmf(kmf, C_mo_lo, ef, omega+1j*freqs[w])
            g0_neg = get_g0_w_from_kmf(kmf, C_mo_lo, ef, omega-1j*freqs[w])
            for w2 in range(nw_sigma):
                Qnv = -wts_w[w] * einsum('Qnu,suv->sQnv',Qnu,g0_pos[:,:,:,w2]+g0_neg[:,:,:,w2])/2./np.pi
                sigma[:,:,:,w2] += einsum('sQnv,Qvl->snl',Qnv,Lpq[:,:,orbs])
    else:
        Qnvw = np.zeros((2,naux,norbs,nmoa,nw_sigma),dtype=np.complex128)
        for w in range(nw):
            Pi_inv = np.linalg.inv(np.eye(naux)-Pi[:,:,w])-np.eye(naux)
            Qnu = einsum('Pnu,PQ->Qnu',Lpq[:,orbs,:],Pi_inv)
            g0_pos = get_g0_w_from_kmf(kmf, C_mo_lo, ef, omega+1j*freqs[w])
            g0_neg = get_g0_w_from_kmf(kmf, C_mo_lo, ef, omega-1j*freqs[w])
            Qnvw[0] += -wts_w[w] * einsum('Qnu,uvw->Qnvw',Qnu,g0_pos[0]+g0_neg[0])/2./np.pi
            Qnvw[1] += -wts_w[w] * einsum('Qnu,uvw->Qnvw',Qnu,g0_pos[1]+g0_neg[1])/2./np.pi
        sigma[0] = einsum('Qnvw,Qvl->nlw',Qnvw[0],Lpq[:,:,orbs])
        sigma[1] = einsum('Qnvw,Qvl->nlw',Qnvw[1],Lpq[:,:,orbs])

    return sigma, omega

def get_response_t(kmf, nocc, C_mo_lo, time, Lpq, mu):
    """
    Compute polarization in time domain
    """
    nt = len(time)
    naux = Lpq.shape[0]
    PQ = np.zeros((naux,naux,nt))
    for t in range(nt):
        g0_pos = get_g0_t_from_kmf(kmf, nocc, C_mo_lo, mu, time[t])
        g0_neg = get_g0_t_from_kmf(kmf, nocc, C_mo_lo, mu, -time[t])
        Pkj_a = einsum('Pij,ki->Pkj',Lpq,g0_pos[0]).reshape(naux,-1)
        Pkj_b = einsum('Pij,ki->Pkj',Lpq,g0_pos[1]).reshape(naux,-1)
        Qkj_a = einsum('Qkl,lj->Qkj',Lpq,g0_neg[0]).reshape(naux,-1)
        Qkj_b = einsum('Qkl,lj->Qkj',Lpq,g0_neg[1]).reshape(naux,-1)
        PQ[:,:,t] = np.dot(Pkj_a,Qkj_a.transpose()) + np.dot(Pkj_b,Qkj_b.transpose())
    return PQ

def get_g0_t_from_kmf(kmf, nocc, C_mo_lo, mu, time):
    """
    Get impurity imaginary-time mean-field Green's function in IAO basis
    """
    nocca, noccb = nocc
    mo_energy = np.asarray(kmf.mo_energy)
    nlo = C_mo_lo.shape[-1]
    spin, nkpts, nmo = mo_energy.shape
    g0 = np.zeros((spin,nkpts,nmo,nmo))
    for s in range(spin):
        noccs = nocc[s]
        if time < 0.:
            for k in range(nkpts):
                g0[s,k,:noccs,:noccs] = np.diag(np.exp(-(mo_energy[s][k,:noccs]-mu)*time))
        else:
            for k in range(nkpts):
                g0[s,k,noccs:,noccs:] = -np.diag(np.exp(-(mo_energy[s][k,noccs:]-mu)*time))

    g0_lo = np.zeros((spin,nkpts,nlo,nlo),dtype=np.complex128)
    for s in range(spin):
        for k in range(nkpts):
            g0_lo[s,k,:,:] = reduce(numpy.dot, (C_mo_lo[s,k].T.conj(), g0[s,k,:,:], C_mo_lo[s,k]))

    g0_imp = g0_lo.sum(axis=1)/nkpts

    return g0_imp.real

def get_g0_w_from_kmf(kmf, C_mo_lo, mu, freqs):
    """
    Get impurity imaginary-freq mean-field Green's function in IAO basis
    """
    mo_energy = np.asarray(kmf.mo_energy)
    spin, nkpts, nmo = mo_energy.shape
    nlo = C_mo_lo.shape[-1]
    nw = len(freqs)
    g0 = np.zeros((spin,nkpts,nmo,nmo,nw),np.complex128)
    for s in range(spin):
        for iw in range(nw):
            for k in range(nkpts):
                g0[s,k,:,:,iw] = np.diag(1./(freqs[iw]+mu-mo_energy[s,k]))

    g0_lo = np.zeros((spin,nkpts,nlo,nlo,nw),np.complex128)
    for s in range(spin):
        for iw in range(nw):
            for k in range(nkpts):
                g0_lo[s,k,:,:,iw] = reduce(numpy.dot, (C_mo_lo[s,k].T.conj(), g0[s,k,:,:,iw], C_mo_lo[s,k]))

    g0_imp = g0_lo.sum(axis=1)/nkpts

    return g0_imp


class UGWGF(UGWAC):

    eta = getattr(__config__, 'ugw_dc_UGWGF_eta', 5e-3)
    # Analytic continuation: pade or twopole
    ac = getattr(__config__, 'ugw_dc_UGWGF_ac', 'pade')

    def __init__(self, mf, frozen=0):
        UGWAC.__init__(self, mf, frozen=0)
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
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira = nmoa - nocca
        nvirb = nmob - noccb
        log.info('GW (nocca, noccb) = (%d, %d), (nvira, nvirb) = (%d, %d)',
                 nocca, noccb, nvira, nvirb)
        if self.frozen > 0:
            log.info('frozen orbitals = %s', str(self.frozen))
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
            Lpq : 4D array (2, naux, nlo, nlo), 3-index ERI
            C_mo_lo: 4D array (spin, nkpts, nmo, nlo), transformation matrix MO -> LO
            orbs: list, orbital indices
            nw: interger, grid number

        Returns:
            self.sigma : 4D array (2, nlo, nlo, nomega), GW self-energy (double counting)
        """
        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()

        naux, nao, nao = Lpq.shape
        mem_incore = (3*nao**2*naux*nw + naux**2*100) * 16 / 1e6
        mem_now = lib.current_memory()[0]
        if (mem_incore + mem_now > 0.95 * self.max_memory):
            mem_incore_small = (2*nao**2*naux + naux**2*100) * 16 / 1e6
            if (mem_incore_small + mem_now < 0.95 * self.max_memory):
                small_mem = True
            else:
                logger.warn(self, 'Memory may not be enough, even with the small memory option!')
                raise NotImplementedError

        self.sigma = kernel(self, omega, Lpq=Lpq, kmf=kmf,
                   C_mo_lo=C_mo_lo, orbs=orbs, nw=nw, nt=nt, verbose=self.verbose, small_mem=small_mem)

        logger.timer(self, 'UGWGF', *cput0)
        return self.sigma
