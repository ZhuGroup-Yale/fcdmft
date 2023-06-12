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
Spin-restricted G0W0 Greens function
This implementation has N^4 scaling, and is faster than GW-CD (N^4)
and analytic GW (N^6) methods.
GW-AC is recommended for valence states only, and is inaccuarate for core states.

Method:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    Compute self-energy on imaginary frequency with density fitting,
    then analytically continued to real frequency

Other useful references:
    J. Chem. Theory Comput. 12, 3623-3635 (2016)
    New J. Phys. 14, 053020 (2012)
"""

import time, h5py
from functools import reduce
import numpy
import numpy as np
from scipy.optimize import newton, least_squares

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df, dft, scf
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__
from fcdmft.gw.mol.gw_ac import _get_scaled_legendre_roots, \
        two_pole_fit, two_pole, AC_twopole_diag, thiele, pade_thiele, \
        AC_pade_thiele_diag, GWAC, get_sigma_diag, \
        AC_twopole_full, AC_pade_thiele_full, as_scanner, \
        _mo_energy_without_core, _mo_without_core, get_rho_response

einsum = lib.einsum

def kernel(gw, gfomega, mo_energy, mo_coeff, Lpq=None, orbs=None,
           nw=None, vhf_df=False, verbose=logger.NOTE):
    """
    GW-corrected quasiparticle orbital energies

    Args:
        Lpq : density fitting 3-center integral in MO basis.
        orbs : a list of orbital indices, default is range(nmo).
        nw : number of frequency point on imaginary axis.
        vhf_df : using density fitting integral to compute HF exchange.

    Returns:
        A list :  gf, gf0, sigma
    """
    mf = gw._scf
    # only support frozen core
    assert isinstance(gw.frozen, int)
    assert gw.frozen < gw.nocc

    if Lpq is None:
        Lpq = gw.ao2mo(mo_coeff)
    if orbs is None:
        orbs = range(gw.nmo)
    else:
        orbs = [x - gw.frozen for x in orbs]
        if orbs[0] < 0:
            raise RuntimeError('GW orbs must be larger than frozen core!')
    gw.orbs = orbs

    # v_xc
    v_mf = mf.get_veff() - mf.get_j()
    v_mf = reduce(np.dot, (mo_coeff.T, v_mf, mo_coeff))

    nmo  = gw.nmo
    nocc = gw.nocc
    nvir = nmo - nocc

    # v_hf from DFT/HF density
    if vhf_df and gw.frozen == 0:
        # density fitting for vk
        vk = -einsum('Lni, Lim -> nm', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
    else:
        # exact vk without density fitting
        dm = mf.make_rdm1()
        if (not isinstance(mf, dft.rks.RKS)) and isinstance(mf, scf.hf.RHF):
            rhf = mf
        else:
            rhf = scf.RHF(gw.mol)
        vk = rhf.get_veff(gw.mol, dm) - rhf.get_j(gw.mol,dm)
        vk = reduce(np.dot, (mo_coeff.T, vk, mo_coeff))

    # Grids for integration on imaginary axis
    freqs, wts = _get_scaled_legendre_roots(nw)
    gw.freqs = freqs
    gw.wts = wts

    ef = (gw._scf.mo_energy[nocc-1] + gw._scf.mo_energy[nocc])/2.
    eta = gw.eta
    nomega = len(gfomega)
    sigma = np.zeros((nmo,nmo,nomega),dtype=np.complex128)
    if gw.fullsigma:
        # Compute full self-energy on imaginary axis i*[0,iw_cutoff]
        sigmaI,omega = get_sigma_full(gw, orbs, Lpq, freqs, wts, iw_cutoff=5.)

        # Analytic continuation
        if gw.ac == 'twopole':
            coeff = AC_twopole_full(sigmaI, omega, orbs, nocc)
        elif gw.ac == 'pade':
            coeff, omega_fit = AC_pade_thiele_full(sigmaI, omega)

        # Compute retarded self-energy
        for p in orbs:
            for q in orbs:
                if gw.ac == 'twopole':
                    sigma[p,q] = two_pole(gfomega-ef+1j*eta, coeff[:,p-orbs[0],q-orbs[0]])
                elif gw.ac == 'pade':
                    sigma[p,q] = pade_thiele(gfomega-ef+1j*eta, omega_fit, coeff[:,p-orbs[0],q-orbs[0]]) 
                sigma[p,q] += vk[p,q] - v_mf[p,q]
    else:
        # Compute diagonal self-energy on imaginary axis
        sigmaI,omega = get_sigma_diag(gw, orbs, Lpq, freqs, wts, iw_cutoff=5.)

        if gw.ac == 'twopole':
            coeff = AC_twopole_diag(sigmaI, omega, orbs, nocc)
        elif gw.ac == 'pade':
            coeff, omega_fit = AC_pade_thiele_diag(sigmaI, omega)

        for p in orbs:
            if gw.ac == 'twopole':
                sigma[p,p] = two_pole(gfomega-ef+1j*eta, coeff[:,p-orbs[0]])
            elif gw.ac == 'pade':
                sigma[p,p] = pade_thiele(gfomega-ef+1j*eta, omega_fit, coeff[:,p-orbs[0]])
            sigma[p,p] += vk[p,p] - v_mf[p,p]

    # Compute Green's function
    gf0 = get_g0(gfomega, mo_energy, eta)
    gf = np.zeros_like(gf0)
    for iw in range(nomega):
        gf[:,:,iw] = np.linalg.inv(np.linalg.inv(gf0[:,:,iw]) - sigma[:,:,iw])

    if gw.ev:
        mo_energy = np.zeros_like(gw._scf.mo_energy)
        for p in orbs:
            # self-consistently solve QP equation
            def quasiparticle(omega):
                if gw.ac == 'twopole':
                    if gw.fullsigma:
                        sigmaR = two_pole(omega-ef, coeff[:, p-orbs[0], p-orbs[0]]).real
                    else:
                        sigmaR = two_pole(omega-ef, coeff[:, p-orbs[0]]).real
                elif gw.ac == 'pade':
                    if gw.fullsigma:
                        sigmaR = pade_thiele(omega-ef, omega_fit, coeff[:, p-orbs[0], p-orbs[0]]).real
                    else:
                        sigmaR = pade_thiele(omega-ef, omega_fit, coeff[:, p-orbs[0]]).real
                return omega - gw._scf.mo_energy[p] - (sigmaR.real + vk[p, p] - v_mf[p, p])
            try:
                e = newton(quasiparticle, gw._scf.mo_energy[p], tol=1e-6, maxiter=100)
                if gw.frozen is not None:
                    mo_energy[p + gw.frozen] = e
                else:
                    mo_energy[p] = e
            except RuntimeError:
                conv = False

        gw.mo_energy = mo_energy
        with np.printoptions(threshold=len(mo_energy)):
            logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)

        if gw.omega_emo:
            gfomega2 = gw.mo_energy
            sigma_2 = np.zeros((nmo,nmo,len(gfomega2)),dtype=np.complex128)
            if gw.fullsigma:
                # Compute retarded self-energy
                for p in orbs:
                    for q in orbs:
                        if gw.ac == 'twopole':
                            sigma_2[p,q] = two_pole(gfomega2-ef+1j*eta, coeff[:,p-orbs[0],q-orbs[0]])
                        elif gw.ac == 'pade':
                            sigma_2[p,q] = pade_thiele(gfomega2-ef+1j*eta, omega_fit, coeff[:,p-orbs[0],q-orbs[0]]) 
                        sigma_2[p,q] += vk[p,q] - v_mf[p,q]
            else:
                for p in orbs:
                    if gw.ac == 'twopole':
                        sigma_2[p,p] = two_pole(gfomega2-ef+1j*eta, coeff[:,p-orbs[0]])
                    elif gw.ac == 'pade':
                        sigma_2[p,p] = pade_thiele(gfomega2-ef+1j*eta, omega_fit, coeff[:,p-orbs[0]])
                    sigma_2[p,p] += vk[p,p] - v_mf[p,p]

            # Compute Green's function
            gf0_2 = get_g0(gfomega2, gw._scf.mo_energy, eta)
            gf_2 = np.zeros_like(gf0_2)
            for iw in range(len(gfomega2)):
                gf_2[:,:,iw] = np.linalg.inv(np.linalg.inv(gf0_2[:,:,iw]) - sigma_2[:,:,iw])

            gf = np.concatenate((gf_2, gf), axis=2)
            gf0 = np.concatenate((gf0_2, gf0), axis=2)
            sigma = np.concatenate((sigma_2, sigma), axis=2)
            gfomega = np.concatenate((gfomega2, gfomega), axis=0)

    gw.omega = gfomega

    return gf, gf0, sigma

def get_sigma_full(gw, orbs, Lpq, freqs, wts, iw_cutoff=None):
    '''
    Compute GW correlation self-energy (all elements) in MO basis
    on imaginary axis
    '''
    mo_energy = _mo_energy_without_core(gw, gw._scf.mo_energy)
    nocc = gw.nocc
    nmo = gw.nmo
    nw = len(freqs)
    naux = Lpq.shape[0]
    norbs = len(orbs)

    # TODO: Treatment of degeneracy
    if (mo_energy[nocc] - mo_energy[nocc-1]) < 1e-3:
        logger.warn(gw, 'GW not well-defined for degeneracy!')
    ef = (mo_energy[nocc-1] + mo_energy[nocc]) * 0.5
    gw.ef = ef

    # Integration on numerical grids
    if iw_cutoff is not None and (not gw.rdm):
        nw_sigma = sum(freqs < iw_cutoff) + 1
    else:
        nw_sigma = nw + 1
    nw_cutoff = sum(freqs < iw_cutoff) + 1

    omega = np.zeros((nw_sigma),dtype=np.complex128)
    omega[0] = 1j*0.
    omega[1:] = 1j*freqs[:(nw_sigma-1)]
    emo = omega[None,:] + ef - mo_energy[:,None]

    sigma = np.zeros((norbs,norbs,nw_sigma),dtype=np.complex128)
    for w in range(nw):
        Pi = get_rho_response(freqs[w], mo_energy, Lpq[:,:nocc,nocc:])
        Pi_inv = np.linalg.inv(np.eye(naux)-Pi)-np.eye(naux)
        Qnm = einsum('Pnm,PQ->Qnm', Lpq[:,orbs,:], Pi_inv)
        Wmn = np.zeros((nmo,norbs,norbs),dtype=np.complex128)
        for orbm in range(nmo):
            Wmn[orbm] = np.dot(Qnm[:,:,orbm].transpose(),Lpq[:,orbm,orbs])
        g0 = wts[w]*emo / (emo**2+freqs[w]**2)
        sigma += -einsum('mnl,mw->nlw', Wmn, g0)/np.pi

    if gw.rdm:
        gw.sigmaI = sigma

    return sigma[:,:,:nw_cutoff], omega[:nw_cutoff]

def get_g0(omega, mo_energy, eta):
    nmo = len(mo_energy)
    nw = len(omega)
    gf0 = np.zeros((nmo,nmo,nw),dtype=np.complex128)
    for iw in range(nw):
        gf0[:,:,iw] = np.diag(1.0/(omega[iw]+1j*eta - mo_energy))
    return gf0

def make_rdm1_dyson(gw):
    '''
    GW density matrix from Dyson's equation (non-conserving)
    '''
    assert(gw.sigmaI is not None)
    assert(gw.rdm and gw.fullsigma)
    sigmaI = gw.sigmaI[:,:,1:]
    freqs = 1j * gw.freqs
    wts = gw.wts
    nmo = gw.nmo
    if len(gw.orbs) != nmo:
        sigma = np.zeros((nmo, nmo, len(freqs)),dtype=sigmaI.dtype)
        for ia,a in enumerate(gw.orbs):
            for ib,b in enumerate(gw.orbs):
                sigma[a,b,:] = sigmaI[ia,ib,:]
    else:
        sigma = sigmaI

    # v_xc
    mf = gw._scf
    v_mf = mf.get_veff() - mf.get_j()
    v_mf = reduce(np.dot, (mf.mo_coeff.T, v_mf, mf.mo_coeff))

    # v_hf from DFT/HF density
    dm = mf.make_rdm1()
    if (not isinstance(mf, dft.rks.RKS)) and isinstance(mf, scf.hf.RHF):
        rhf = mf
    else:
        rhf = scf.RHF(gw.mol)
    vk = rhf.get_veff(gw.mol, dm) - rhf.get_j(gw.mol,dm)
    vk = reduce(np.dot, (mf.mo_coeff.T, vk, mf.mo_coeff))

    # Compute GW Green's function on imag freq
    eta= 0.
    gf0 = get_g0(freqs, np.array(gw._scf.mo_energy)-gw.ef, eta)
    gf = np.zeros_like(gf0)
    for iw in range(len(freqs)):
        gf[:,:,iw] = np.linalg.inv(np.linalg.inv(gf0[:,:,iw]) - (vk + sigma[:,:,iw] - v_mf))

    # GW density matrix
    rdm1 = 2./np.pi * einsum('ijw,w->ij',gf,wts) + np.eye(nmo)
    rdm1 = rdm1.real
    logger.info(gw, 'GW particle number = %s', np.trace(rdm1))

    # Symmetrize density matrix
    rdm1 = 0.5 * (rdm1 + rdm1.T)

    return rdm1

def make_rdm1_linear(gw):
    '''
    Linearized GW density matrix (default, conserving)
    Ref: JCTC 17, 2126-2136 (2021)
    '''
    assert(gw.sigmaI is not None)
    assert(gw.rdm and gw.fullsigma)
    sigmaI = gw.sigmaI[:,:,1:]
    freqs = 1j * gw.freqs
    wts = gw.wts
    nmo = gw.nmo
    if len(gw.orbs) != nmo:
        sigma = np.zeros((nmo, nmo, len(freqs)),dtype=sigmaI.dtype)
        for ia,a in enumerate(gw.orbs):
            for ib,b in enumerate(gw.orbs):
                sigma[a,b,:] = sigmaI[ia,ib,:]
    else:
        sigma = sigmaI

    # v_xc
    mf = gw._scf
    v_mf = mf.get_veff() - mf.get_j()
    v_mf = reduce(np.dot, (mf.mo_coeff.T, v_mf, mf.mo_coeff))

    # v_hf from DFT/HF density
    dm = mf.make_rdm1()
    if (not isinstance(mf, dft.rks.RKS)) and isinstance(mf, scf.hf.RHF):
        rhf = mf
    else:
        rhf = scf.RHF(gw.mol)
    vk = rhf.get_veff(gw.mol, dm) - rhf.get_j(gw.mol,dm)
    vk = reduce(np.dot, (mf.mo_coeff.T, vk, mf.mo_coeff))

    # Compute GW Green's function on imag freq
    eta= 0.
    gf0 = get_g0(freqs, np.array(gw._scf.mo_energy)-gw.ef, eta)
    gf = np.zeros_like(gf0)
    for iw in range(len(freqs)):
        gf[:,:,iw] = gf0[:,:,iw] + np.dot(gf0[:,:,iw], (vk + sigma[:,:,iw] - v_mf)).dot(gf0[:,:,iw])

    # GW density matrix
    rdm1 = 2./np.pi * einsum('ijw,w->ij',gf,wts) + np.eye(nmo)
    rdm1 = rdm1.real
    logger.info(gw, 'GW particle number = %s', np.trace(rdm1))

    # Symmetrize density matrix
    rdm1 = 0.5 * (rdm1 + rdm1.T)

    return rdm1

def energy_tot(gw):
    """
    Experimental feature: Compute GW total energy according to
    Galitskii-Migdal formula. Ref: Phys. Rev. B 86, 081102 (2012)
    """
    assert(gw.sigmaI is not None)
    assert(gw.rdm and gw.fullsigma)
    sigmaI = gw.sigmaI[:,:,1:]
    freqs = 1j * gw.freqs
    wts = gw.wts
    nmo = gw.nmo

    if len(gw.orbs) != nmo:
        sigma = np.zeros((nmo, nmo, len(freqs)),dtype=sigmaI.dtype)
        for ia,a in enumerate(gw.orbs):
            for ib,b in enumerate(gw.orbs):
                sigma[a,b,:] = sigmaI[ia,ib,:]
    else:
        sigma = sigmaI

    # Compute mean-field Green's function on imag freq
    eta= 0.
    gf0 = get_g0(freqs, np.array(gw._scf.mo_energy)-gw.ef, eta)

    # Compute GW correlation energy
    g_sigma = 1./2./np.pi * einsum('ijw,ijw,w->ij',gf0,sigma,wts)
    Ec = 2. * np.trace(g_sigma).real

    # Compute HF energy using DFT density matrix
    dm = gw._scf.make_rdm1()
    rhf = scf.RHF(gw.mol)
    e_hf = rhf.energy_elec(dm=dm)[0]
    e_hf += gw._scf.energy_nuc()

    E_tot = e_hf + Ec

    return E_tot, e_hf, Ec

class GWGF(GWAC):

    eta = getattr(__config__, 'gw_gf_GWGF_eta', 5e-3)
    fullsigma = getattr(__config__, 'gw_gf_GWGF_fullsigma', False)
    # Analytic continuation: pade or twopole
    ac = getattr(__config__, 'gw_gf_GWGF_ac', 'pade')
    ev = getattr(__config__, 'gw_gf_GWGF_ev', True)

    def __init__(self, mf, frozen=0, auxbasis=None):
        GWAC.__init__(self, mf, frozen=frozen, auxbasis=auxbasis)
        self.freqs = None
        self.wts = None
        self.rdm = False
        self.sigmaI = None
        self.ef = None
        self.orbs = None
        self.omega_emo = False
        self.gf = None
        self.gf0 = None
        self.sigma = None
        self.omega = None
        keys = set(('eta','fullsigma','ac','ev'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self, verbose=None):
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

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    make_rdm1 = make_rdm1_linear
    energy_tot = energy_tot
    as_scanner = as_scanner

    def kernel(self, omega, mo_energy=None, mo_coeff=None, Lpq=None, orbs=None, nw=100, vhf_df=False):
        """
        Args:
            mo_energy : 1D array (nmo), mean-field mo energy
            mo_coeff : 2D array (nmo, nmo), mean-field mo coefficient
            Lpq : 3D array (naux, nmo, nmo), 3-index ERI
            orbs: list, orbital indices
            nw: interger, grid number
            vhf_df: bool, use density fitting for HF exchange or not

        Returns:
            self.gf : 3D array (nmo, nmo, nomega), GW Greens function
            self.gf0 : 3D array (nmo, nmo, nomega), mean-field Greens function
            self.sigma : 3D array (nmo, nmo, nomega), GW self-energy
        """
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self._scf.mo_energy)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.gf, self.gf0, self.sigma = kernel(self, omega, mo_energy, mo_coeff,
                   Lpq=Lpq, orbs=orbs, nw=nw, vhf_df=vhf_df, verbose=self.verbose)

        logger.timer(self, 'GWGF', *cput0)
        return self.gf, self.gf0, self.sigma

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        naux = self.with_df.get_naoaux()
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair*naux, nmo**2*naux) + nmo_pair*naux) * 8/1e6
        mem_now = lib.current_memory()[0]

        mo = numpy.asarray(mo_coeff, order='F')
        ijslice = (0, nmo, 0, nmo)
        Lpq = None
        if (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway:
            Lpq = _ao2mo.nr_e2(self.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
            return Lpq.reshape(naux,nmo,nmo)
        else:
            logger.warn(self, 'Memory not enough!')
            raise NotImplementedError


if __name__ == '__main__':
    from pyscf import gto, dft, scf
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.7571 , 0.5861)],
        [1 , (0. , 0.7571 , 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel()
    nocc = mol.nelectron // 2

    gw = GWGF(mf)
    gw.ac = 'pade'
    gw.fullsigma = True
    gw.eta = 1e-2
    omega = np.linspace(-0.5,0.5,101)
    gf, gf0, sigma = gw.kernel(omega=omega, orbs=range(0,nocc+10))
    for i in range(len(omega)):
        print (omega[i],-np.trace(gf0[:,:,i].imag)/np.pi, \
                        -np.trace(gf[:,:,i].imag)/np.pi)
    assert(abs(-np.trace(gf[:,:,0].imag)/np.pi-16.899357092485513)<1e-3)

    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = 'H 0 0 0; H 0 0 1'
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'hf'
    mf.kernel()
    nocc = mol.nelectron // 2

    gw = GWGF(mf)
    gw.rdm = True
    gw.ac = 'pade'
    gw.fullsigma = True
    gw.eta = 1e-2
    omega = np.linspace(-0.5,0.5,2)
    gf, gf0, sigma = gw.kernel(omega=omega, nw=200)
    rdm1 = gw.make_rdm1()
    print (rdm1.diagonal())

    egw, ehf, ec = gw.energy_tot()
    print (egw, ehf, ec)
