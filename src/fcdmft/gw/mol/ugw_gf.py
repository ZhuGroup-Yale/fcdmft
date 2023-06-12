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
Spin-unrestricted G0W0 Greens function
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
import numpy as np
from scipy.optimize import newton

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df, dft, scf
from pyscf.mp.ump2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__
from fcdmft.gw.mol.gw_ac import _get_scaled_legendre_roots, \
        two_pole_fit, two_pole, AC_twopole_diag, thiele, pade_thiele, \
        AC_pade_thiele_diag, AC_twopole_full, AC_pade_thiele_full
from fcdmft.gw.mol.ugw_ac import _mo_energy_without_core, _mo_without_core, \
        get_rho_response, get_sigma_diag, UGWAC, as_scanner

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
    if gw.frozen is not None:
        assert isinstance(gw.frozen, int)
        assert (gw.frozen < gw.nocc[0] and gw.frozen < gw.nocc[1])

    nocca, noccb = gw.nocc
    nmoa, nmob = gw.nmo
    nvira = nmoa - nocca
    nvirb = nmob - noccb

    if Lpq is None:
        Lpq = gw.ao2mo(mo_coeff)
    if orbs is None:
        orbs = range(nmoa)
    if orbs is not None and gw.frozen is not None:
        orbs = [x - gw.frozen for x in orbs]
        if orbs[0] < 0:
            raise RuntimeError('GW orbs must be larger than frozen core!')
    gw.orbs = orbs

    # v_xc
    v_mf = mf.get_veff()
    vj = mf.get_j()
    v_mf[0] = v_mf[0] - (vj[0] + vj[1])
    v_mf[1] = v_mf[1] - (vj[0] + vj[1])
    v_mf_frz = np.zeros((2, nmoa, nmoa))
    for s in range(2):
        v_mf_frz[s] = reduce(np.dot, (mo_coeff[s].T, v_mf[s], mo_coeff[s]))
    v_mf = v_mf_frz

    # v_hf from DFT/HF density
    if vhf_df and gw.frozen is not None:
        # density fitting vk
        vk = np.zeros_like(v_mf)
        vk[0] = -einsum('Lni, Lim -> nm', Lpq[0,:,:,:nocca], Lpq[0,:,:nocca,:])
        vk[1] = -einsum('Lni, Lim -> nm', Lpq[1,:,:,:noccb], Lpq[1,:,:noccb,:])
    else:
        # exact vk without density fitting
        dm = mf.make_rdm1()
        if (not isinstance(mf, dft.uks.UKS)) and isinstance(mf, scf.uhf.UHF):
            uhf = mf
        else:
            uhf = scf.UHF(gw.mol)
        vk = uhf.get_veff(gw.mol,dm)
        vj = uhf.get_j(gw.mol,dm)
        vk[0] = vk[0] - (vj[0] + vj[1])
        vk[1] = vk[1] - (vj[0] + vj[1])
        vk_frz = np.zeros((2, nmoa, nmoa))
        for s in range(2):
            vk_frz[s] = reduce(np.dot, (mo_coeff[s].T, vk[s], mo_coeff[s]))
        vk = vk_frz

    # Grids for integration on imaginary axis
    freqs, wts = _get_scaled_legendre_roots(nw)
    gw.freqs = freqs
    gw.wts = wts

    homo = max(mo_energy[0][nocca-1], mo_energy[1][noccb-1])
    lumo = min(mo_energy[0][nocca], mo_energy[1][noccb])
    ef = (homo + lumo) * 0.5
    eta = gw.eta
    nomega = len(gfomega)
    sigma = np.zeros((2,nmoa,nmoa,nomega),dtype=np.complex128)
    if gw.fullsigma:
        # Compute full self-energy on imaginary axis i*[0,iw_cutoff]
        sigmaI,omega = get_sigma_full(gw, orbs, Lpq, freqs, wts, iw_cutoff=5.)

        # Analytic continuation
        if gw.ac == 'twopole':
            coeff_a = AC_twopole_full(sigmaI[0], omega, orbs, nocca)
            coeff_b = AC_twopole_full(sigmaI[1], omega, orbs, noccb)
        elif gw.ac == 'pade':
            coeff_a, omega_fit_a = AC_pade_thiele_full(sigmaI[0], omega)
            coeff_b, omega_fit_b = AC_pade_thiele_full(sigmaI[1], omega)
            omega_fit = np.asarray((omega_fit_a, omega_fit_b))
        coeff = np.asarray((coeff_a, coeff_b))

        # Compute retarded self-energy
        for s in range(2):
            for p in orbs:
                for q in orbs:
                    if gw.ac == 'twopole':
                        sigma[s,p,q] = two_pole(gfomega-ef+1j*eta, coeff[s,:,p-orbs[0],q-orbs[0]])
                    elif gw.ac == 'pade':
                        sigma[s,p,q] = pade_thiele(gfomega-ef+1j*eta, omega_fit[s], coeff[s,:,p-orbs[0],q-orbs[0]])
                    sigma[s,p,q] += vk[s][p,q] - v_mf[s][p,q]
    else:
        # Compute self-energy on imaginary axis i*[0,iw_cutoff]
        sigmaI, omega = get_sigma_diag(gw, orbs, Lpq, freqs, wts, iw_cutoff=5.0)

        # Analytic continuation
        if gw.ac == 'twopole':
            coeff_a = AC_twopole_diag(sigmaI[0], omega, orbs, nocca)
            coeff_b = AC_twopole_diag(sigmaI[1], omega, orbs, noccb)
        elif gw.ac == 'pade':
            coeff_a, omega_fit_a = AC_pade_thiele_diag(sigmaI[0], omega)
            coeff_b, omega_fit_b = AC_pade_thiele_diag(sigmaI[1], omega)
            omega_fit = np.asarray((omega_fit_a, omega_fit_b))
        coeff = np.asarray((coeff_a, coeff_b))

        for s in range(2):
            for p in orbs:
                if gw.ac == 'twopole':
                    sigma[s,p,p] = two_pole(gfomega-ef+1j*eta, coeff[s,:,p-orbs[0]])
                elif gw.ac == 'pade':
                    sigma[s,p,p] = pade_thiele(gfomega-ef+1j*eta, omega_fit[s], coeff[s,:,p-orbs[0]])
                sigma[s,p,p] += vk[s][p,p] - v_mf[s][p,p]

    # Compute Green's function
    gf0 = get_g0(gfomega, mo_energy, eta)
    gf = np.zeros_like(gf0)
    for s in range(2):
        for iw in range(nomega):
            gf[s,:,:,iw] = np.linalg.inv(np.linalg.inv(gf0[s,:,:,iw]) - sigma[s,:,:,iw])

    if gw.ev:
        mo_energy = np.zeros_like(gw._scf.mo_energy)
        for s in range(2):
            for p in orbs:
                # self-consistently solve QP equation
                def quasiparticle(omega):
                    if gw.ac == 'twopole':
                        if gw.fullsigma:
                            sigmaR = two_pole(omega-ef, coeff[s,:, p-orbs[0], p-orbs[0]]).real
                        else:
                            sigmaR = two_pole(omega-ef, coeff[s,:,p-orbs[0]]).real
                    elif gw.ac == 'pade':
                        if gw.fullsigma:
                            sigmaR = pade_thiele(omega-ef, omega_fit[s], coeff[s,:,p-orbs[0],p-orbs[0]]).real
                        else:
                            sigmaR = pade_thiele(omega-ef, omega_fit[s], coeff[s,:,p-orbs[0]]).real
                    return omega - gw._scf.mo_energy[s][p] - (sigmaR.real + vk[s,p,p] - v_mf[s,p,p])
                try:
                    e = newton(quasiparticle, gw._scf.mo_energy[s][p], tol=1e-6, maxiter=100)
                    if gw.frozen is not None:
                        mo_energy[s,p+gw.frozen] = e
                    else:
                        mo_energy[s,p] = e
                except RuntimeError:
                    conv = False

        gw.mo_energy = mo_energy
        with np.printoptions(threshold=len(mo_energy[0])):
            logger.debug(gw, '  GW mo_energy spin-up   =\n%s', mo_energy[0])
            logger.debug(gw, '  GW mo_energy spin-down =\n%s', mo_energy[1])

        if gw.omega_emo:
            gfomega2 = np.concatenate((gw.mo_energy[0],gw.mo_energy[1]))
            sigma_2 = np.zeros((2,nmoa,nmoa,len(gfomega2)),dtype=np.complex128)
            for s in range(2):
                if gw.fullsigma:
                    # Compute retarded self-energy
                    for p in orbs:
                        for q in orbs:
                            if gw.ac == 'twopole':
                                sigma_2[s,p,q] = two_pole(gfomega2-ef+1j*eta, coeff[s,:,p-orbs[0],q-orbs[0]])
                            elif gw.ac == 'pade':
                                sigma_2[s,p,q] = pade_thiele(gfomega2-ef+1j*eta, omega_fit[s], coeff[s,:,p-orbs[0],q-orbs[0]])
                            sigma_2[s,p,q] += vk[s][p,q] - v_mf[s][p,q]
                else:
                    for p in orbs:
                        if gw.ac == 'twopole':
                            sigma_2[s,p,p] = two_pole(gfomega2-ef+1j*eta, coeff[s,:,p-orbs[0]])
                        elif gw.ac == 'pade':
                            sigma_2[s,p,p] = pade_thiele(gfomega2-ef+1j*eta, omega_fit[s], coeff[s,:,p-orbs[0]])
                        sigma_2[s,p,p] += vk[s][p,p] - v_mf[s][p,p]

            # Compute Green's function
            gf0_2 = get_g0(gfomega2, gw._scf.mo_energy, eta)
            gf_2 = np.zeros_like(gf0_2)
            for s in range(2):
                for iw in range(len(gfomega2)):
                    gf_2[s,:,:,iw] = np.linalg.inv(np.linalg.inv(gf0_2[s,:,:,iw]) - sigma_2[s,:,:,iw])

            gf = np.concatenate((gf_2, gf), axis=3)
            gf0 = np.concatenate((gf0_2, gf0), axis=3)
            sigma = np.concatenate((sigma_2, sigma), axis=3)
            gfomega = np.concatenate((gfomega2, gfomega), axis=0)

    gw.omega = gfomega

    return gf, gf0, sigma

def get_sigma_full(gw, orbs, Lpq, freqs, wts, iw_cutoff=None):
    '''
    Compute GW correlation self-energy (all elements) in MO basis
    on imaginary axis
    '''
    mo_energy = _mo_energy_without_core(gw, gw._scf.mo_energy)
    nocca, noccb = gw.nocc
    nmoa, nmob = gw.nmo
    nw = len(freqs)
    naux = Lpq[0].shape[0]
    norbs = len(orbs)

    # TODO: Treatment of degeneracy
    homo = max(mo_energy[0][nocca-1], mo_energy[1][noccb-1])
    lumo = min(mo_energy[0][nocca], mo_energy[1][noccb])
    if (lumo-homo) < 1e-3:
        logger.warn(gw, 'GW not well-defined for degeneracy!')
    ef = (homo + lumo) * 0.5
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
    emo_a = omega[None] + ef - mo_energy[0, :, None]
    emo_b = omega[None] + ef - mo_energy[1, :, None]

    sigma = np.zeros((2, norbs, norbs, nw_sigma), dtype=np.complex128)
    for w in range(nw):
        Pi = get_rho_response(freqs[w], mo_energy, Lpq[0,:,:nocca,nocca:], Lpq[1,:,:noccb,noccb:])
        Pi_inv = np.linalg.inv(np.eye(naux) - Pi)
        Pi_inv[range(naux), range(naux)] -= 1.0
        g0_a = wts[w] * emo_a / (emo_a**2 + freqs[w]**2)
        g0_b = wts[w] * emo_b / (emo_b**2 + freqs[w]**2)

        Qnm_a = einsum('Pnm,PQ->Qnm',Lpq[0][:,orbs,:],Pi_inv)
        Qnm_b = einsum('Pnm,PQ->Qnm',Lpq[1][:,orbs,:],Pi_inv)
        Wmn_a = np.zeros((nmoa,norbs,norbs),dtype=np.complex128)
        Wmn_b = np.zeros((nmob,norbs,norbs),dtype=np.complex128)
        for orbm in range(nmoa):
            Wmn_a[orbm] = np.dot(Qnm_a[:,:,orbm].transpose(),Lpq[0][:,orbm,orbs])
            Wmn_b[orbm] = np.dot(Qnm_b[:,:,orbm].transpose(),Lpq[1][:,orbm,orbs])
        sigma[0] += -einsum('mnl,mw->nlw', Wmn_a, g0_a)/np.pi
        sigma[1] += -einsum('mnl,mw->nlw', Wmn_b, g0_b)/np.pi

    if gw.rdm:
        gw.sigmaI = sigma

    return sigma[:,:,:,:nw_cutoff], omega[:nw_cutoff]

def get_g0(omega, mo_energy, eta):
    nmo = len(mo_energy[0])
    nw = len(omega)
    gf0 = np.zeros((2,nmo,nmo,nw),dtype=np.complex128)
    for s in range(2):
        for iw in range(nw):
            gf0[s,:,:,iw] = np.diag(1.0/(omega[iw]+1j*eta - mo_energy[s]))
    return gf0

def make_rdm1_dyson(gw):
    '''
    GW density matrix from Dyson's equation (non-conserving)
    '''
    assert(gw.sigmaI is not None)
    assert(gw.rdm and gw.fullsigma)
    sigmaI = gw.sigmaI[:,:,:,1:]
    freqs = 1j * gw.freqs
    wts = gw.wts
    nmoa, nmob = gw.nmo
    if len(gw.orbs) != nmoa:
        sigma = np.zeros((2, nmoa, nmoa, len(freqs)),dtype=sigmaI.dtype)
        for s in range(2):
            for ia,a in enumerate(gw.orbs):
                for ib,b in enumerate(gw.orbs):
                    sigma[s,a,b,:] = sigmaI[s,ia,ib,:]
    else:
        sigma = sigmaI

    # v_xc
    mf = gw._scf
    v_mf = mf.get_veff()
    vj = mf.get_j()
    v_mf[0] = v_mf[0] - (vj[0] + vj[1])
    v_mf[1] = v_mf[1] - (vj[0] + vj[1])
    for s in range(2):
        v_mf[s] = reduce(np.dot, (mf.mo_coeff[s].T, v_mf[s], mf.mo_coeff[s]))

    # v_hf from DFT/HF density
    dm = mf.make_rdm1()
    if (not isinstance(mf, dft.uks.UKS)) and isinstance(mf, scf.uhf.UHF):
        uhf = mf
    else:
        uhf = scf.UHF(gw.mol)
    vk = uhf.get_veff(gw.mol,dm)
    vj = uhf.get_j(gw.mol,dm)
    vk[0] = vk[0] - (vj[0] + vj[1])
    vk[1] = vk[1] - (vj[0] + vj[1])
    for s in range(2):
        vk[s] = reduce(np.dot, (mf.mo_coeff[s].T, vk[s], mf.mo_coeff[s]))

    # Compute GW Green's function on imag freq
    eta= 0.
    gf0 = get_g0(freqs, np.array(gw._scf.mo_energy)-gw.ef, eta)
    gf = np.zeros_like(gf0)
    for s in range(2):
        for iw in range(len(freqs)):
            gf[s,:,:,iw] = np.linalg.inv(np.linalg.inv(gf0[s,:,:,iw]) - (vk[s] + sigma[s,:,:,iw] - v_mf[s]))

    # GW density matrix
    rdm1 = 1./np.pi * einsum('sijw,w->sij',gf,wts) + 0.5 * np.array((np.eye(nmoa), np.eye(nmoa)))
    rdm1 = rdm1.real
    logger.info(gw, 'GW particle number up = %s, dn = %s, total = %s', 
                np.trace(rdm1[0]), np.trace(rdm1[1]), np.trace(rdm1[0]+rdm1[1]))

    # Symmetrize density matrix
    for s in range(2):
        rdm1[s] = 0.5 * (rdm1[s] + rdm1[s].T)

    return rdm1

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
    nmoa, nmob = gw.nmo
    if len(gw.orbs) != nmoa:
        sigma = np.zeros((2, nmoa, nmoa, len(freqs)),dtype=sigmaI.dtype)
        for s in range(2):
            for ia,a in enumerate(gw.orbs):
                for ib,b in enumerate(gw.orbs):
                    sigma[s,a,b,:] = sigmaI[s,ia,ib,:]
    else:
        sigma = sigmaI

    # v_xc
    mf = gw._scf
    v_mf = mf.get_veff()
    vj = mf.get_j()
    v_mf[0] = v_mf[0] - (vj[0] + vj[1])
    v_mf[1] = v_mf[1] - (vj[0] + vj[1])
    for s in range(2):
        v_mf[s] = reduce(np.dot, (mf.mo_coeff[s].T, v_mf[s], mf.mo_coeff[s]))

    # v_hf from DFT/HF density
    dm = mf.make_rdm1()
    if (not isinstance(mf, dft.uks.UKS)) and isinstance(mf, scf.uhf.UHF):
        uhf = mf
    else:
        uhf = scf.UHF(gw.mol)
    vk = uhf.get_veff(gw.mol,dm)
    vj = uhf.get_j(gw.mol,dm)
    vk[0] = vk[0] - (vj[0] + vj[1])
    vk[1] = vk[1] - (vj[0] + vj[1])
    for s in range(2):
        vk[s] = reduce(np.dot, (mf.mo_coeff[s].T, vk[s], mf.mo_coeff[s]))

    # Compute GW Green's function on imag freq
    eta= 0.
    gf0 = get_g0(freqs, np.array(gw._scf.mo_energy)-gw.ef, eta)
    gf = np.zeros_like(gf0)
    for s in range(2):
        for iw in range(len(freqs)):
            gf[s,:,:,iw] = gf0[s,:,:,iw] + np.dot(gf0[s,:,:,iw], (vk[s] + sigma[s,:,:,iw] - v_mf[s])).dot(gf0[s,:,:,iw])

    # GW density matrix
    rdm1 = 1./np.pi * einsum('sijw,w->sij',gf,wts) + 0.5 * np.array((np.eye(nmoa), np.eye(nmoa)))
    rdm1 = rdm1.real
    logger.info(gw, 'GW particle number up = %s, dn = %s, total = %s', 
                np.trace(rdm1[0]), np.trace(rdm1[1]), np.trace(rdm1[0]+rdm1[1]))

    # Symmetrize density matrix
    for s in range(2):
        rdm1[s] = 0.5 * (rdm1[s] + rdm1[s].T)

    return rdm1

class UGWGF(UGWAC):

    eta = getattr(__config__, 'gw_gf_UGWGF_eta', 5e-3)
    fullsigma = getattr(__config__, 'gw_gf_UGWGF_fullsigma', False)
    # Analytic continuation: pade or twopole
    ac = getattr(__config__, 'gw_gf_UGWGF_ac', 'pade')
    ev = getattr(__config__, 'gw_gf_UGWGF_ev', True)

    def __init__(self, mf, frozen=0):
        UGWAC.__init__(self, mf, frozen=0)
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
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira = nmoa - nocca
        nvirb = nmob - noccb
        log.info('GW (nocca, noccb) = (%d, %d), (nvira, nvirb) = (%d, %d)',
                 nocca, noccb, nvira, nvirb)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', str(self.frozen))
        logger.info(self, 'analytic continuation method = %s', self.ac)
        return self

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    make_rdm1 = make_rdm1_linear
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

        logger.timer(self, 'UGWGF', *cput0)
        return self.gf, self.gf0, self.sigma

    def ao2mo(self, mo_coeff=None):
        nmoa, nmob = self.nmo
        nao = self.mo_coeff[0].shape[0]
        naux = self.with_df.get_naoaux()
        mem_incore = (nmoa**2*naux + nmob**2*naux + nao**2*naux) * 8/1e6
        mem_now = lib.current_memory()[0]

        moa = np.asarray(mo_coeff[0], order='F')
        mob = np.asarray(mo_coeff[1], order='F')
        ijslicea = (0, nmoa, 0, nmoa)
        ijsliceb = (0, nmob, 0, nmob)
        Lpqa = None
        Lpqb = None
        if (mem_incore + mem_now < 0.99*self.max_memory) or self.mol.incore_anyway:
            Lpqa = _ao2mo.nr_e2(self.with_df._cderi, moa, ijslicea, aosym='s2', out=Lpqa)
            Lpqb = _ao2mo.nr_e2(self.with_df._cderi, mob, ijsliceb, aosym='s2', out=Lpqb)
            return np.asarray((Lpqa.reshape(naux,nmoa,nmoa),Lpqb.reshape(naux,nmob,nmob)))
        else:
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError

if __name__ == '__main__':
    from pyscf import gto, dft, scf
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = 'O 0 0 0'
    mol.basis = 'aug-cc-pvdz'
    mol.spin = 2
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'hf'
    mf.kernel()

    nocca = (mol.nelectron + mol.spin) // 2
    noccb = mol.nelectron - nocca
    nmo = len(mf.mo_energy[0])
    nvira = nmo - nocca
    nvirb = nmo - noccb

    gw = UGWGF(mf)
    gw.rdm = True
    gw.ac = 'pade'
    gw.fullsigma = True
    gw.eta = 1e-2
    omega = np.linspace(-0.8,0.5,131)
    gf, gf0, sigma = gw.kernel(omega=omega)
    for i in range(len(omega)):
        print (omega[i],-np.trace(gf0[0,:,:,i].imag)/np.pi, \
                        -np.trace(gf[0,:,:,i].imag)/np.pi)
    assert(abs(-np.trace(gf[0,:,:,0].imag)/np.pi-0.3465604077261225)<1e-3)
    rdm1 = gw.make_rdm1()
    print (rdm1[0].diagonal())
    print (rdm1[1].diagonal())
