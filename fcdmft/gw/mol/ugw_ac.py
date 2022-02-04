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
Spin-restricted G0W0-AC QP eigenvalues
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
        AC_pade_thiele_diag, GWAC

einsum = lib.einsum

def kernel(gw, mo_energy, mo_coeff, Lpq=None, orbs=None,
           nw=None, vhf_df=False, verbose=logger.NOTE):
    """
    GW-corrected quasiparticle orbital energies

    Args:
        Lpq : density fitting 3-center integral in MO basis.
        orbs : a list of orbital indices, default is range(nmo).
        nw : number of frequency point on imaginary axis.
        vhf_df : using density fitting integral to compute HF exchange.

    Returns:
        A tuple : converged, mo_energy, mo_coeff
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
    else:
        raise ValueError("Unknown GW-AC type %s"%(gw.ac))
    coeff = np.asarray((coeff_a, coeff_b))

    conv = True
    homo = max(mo_energy[0][nocca-1], mo_energy[1][noccb-1])
    lumo = min(mo_energy[0][nocca], mo_energy[1][noccb])
    ef = (homo + lumo) * 0.5
    mf_mo_energy = mo_energy.copy()
    mo_energy = np.zeros_like(np.asarray(gw._scf.mo_energy))
    for s in range(2):
        for p in orbs:
            if gw.linearized:
                # linearized G0W0
                de = 1e-6
                ep = mf_mo_energy[s][p]
                #TODO: analytic sigma derivative
                if gw.ac == 'twopole':
                    sigmaR = two_pole(ep-ef, coeff[s,:,p-orbs[0]]).real
                    dsigma = two_pole(ep-ef+de, coeff[s,:,p-orbs[0]]).real - sigmaR.real
                elif gw.ac == 'pade':
                    sigmaR = pade_thiele(ep-ef, omega_fit[s], coeff[s,:,p-orbs[0]]).real
                    dsigma = pade_thiele(ep-ef+de, omega_fit[s], coeff[s,:,p-orbs[0]]).real - sigmaR.real
                zn = 1.0 / (1.0 - dsigma/de)
                e = ep + zn*(sigmaR.real + vk[s,p,p] - v_mf[s,p,p])
                if gw.frozen is not None:
                    mo_energy[s,p+gw.frozen] = e
                else:
                    mo_energy[s,p] = e
            else:
                # self-consistently solve QP equation
                def quasiparticle(omega):
                    if gw.ac == 'twopole':
                        sigmaR = two_pole(omega-ef, coeff[s,:,p-orbs[0]]).real
                    elif gw.ac == 'pade':
                        sigmaR = pade_thiele(omega-ef, omega_fit[s], coeff[s,:,p-orbs[0]]).real
                    return omega - mf_mo_energy[s][p] - (sigmaR.real + vk[s,p,p] - v_mf[s,p,p])
                try:
                    e = newton(quasiparticle, mf_mo_energy[s][p], tol=1e-6, maxiter=100)
                    if gw.frozen is not None:
                        mo_energy[s,p+gw.frozen] = e
                    else:
                        mo_energy[s,p] = e
                except RuntimeError:
                    conv = False
    mo_coeff = gw._scf.mo_coeff

    with np.printoptions(threshold=len(mo_energy[0])):
        logger.debug(gw, '  GW mo_energy spin-up   =\n%s', mo_energy[0])
        logger.debug(gw, '  GW mo_energy spin-down =\n%s', mo_energy[1])

    return conv, mo_energy, mo_coeff

def get_rho_response(omega, mo_energy, Lpqa, Lpqb):
    '''
    Compute density response function in auxiliary basis at freq iw
    '''
    naux, nocca, nvira = Lpqa.shape
    naux, noccb, nvirb = Lpqb.shape
    eia_a = mo_energy[0,:nocca,None] - mo_energy[0,None,nocca:]
    eia_b = mo_energy[1,:noccb,None] - mo_energy[1,None,noccb:]
    eia_a = eia_a / (omega**2 + eia_a*eia_a)
    eia_b = eia_b / (omega**2 + eia_b*eia_b)
    Pia_a = Lpqa * (eia_a * 2.0)
    Pia_b = Lpqb * (eia_b * 2.0)
    # Response from both spin-up and spin-down density
    Pi = einsum('Pia, Qia -> PQ', Pia_a, Lpqa) + einsum('Pia, Qia -> PQ', Pia_b, Lpqb)
    return Pi

def get_sigma_diag(gw, orbs, Lpq, freqs, wts, iw_cutoff=None):
    '''
    Compute GW correlation self-energy (diagonal elements)
    in MO basis on imaginary axis
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

    # Integration on numerical grids
    if iw_cutoff is not None:
        nw_sigma = sum(freqs < iw_cutoff) + 1
    else:
        nw_sigma = nw + 1

    omega = np.zeros((nw_sigma),dtype=np.complex128)
    omega[1:] = 1j*freqs[:(nw_sigma-1)]
    emo_a = omega[None] + ef - mo_energy[0, :, None]
    emo_b = omega[None] + ef - mo_energy[1, :, None]

    sigma = np.zeros((2, norbs, nw_sigma), dtype=np.complex128)
    for w in range(nw):
        Pi = get_rho_response(freqs[w], mo_energy, Lpq[0,:,:nocca,nocca:], Lpq[1,:,:noccb,noccb:])
        Pi_inv = np.linalg.inv(np.eye(naux) - Pi)
        Pi_inv[range(naux), range(naux)] -= 1.0
        g0_a = wts[w] * emo_a / (emo_a**2 + freqs[w]**2)
        g0_b = wts[w] * emo_b / (emo_b**2 + freqs[w]**2)

        Qnm_a = einsum('Pnm,PQ->Qnm',Lpq[0][:,orbs,:],Pi_inv)
        Qnm_b = einsum('Pnm,PQ->Qnm',Lpq[1][:,orbs,:],Pi_inv)
        Wmn_a = einsum('Qnm,Qmn->mn',Qnm_a,Lpq[0][:,:,orbs])
        Wmn_b = einsum('Qnm,Qmn->mn',Qnm_b,Lpq[1][:,:,orbs])

        sigma[0] -= einsum('mn,mw->nw',Wmn_a,g0_a) / np.pi
        sigma[1] -= einsum('mn,mw->nw',Wmn_b,g0_b) / np.pi

    return sigma, omega

def _mo_energy_without_core(gw, mo_energy):
    moidx = get_frozen_mask(gw)
    mo_energy = (mo_energy[0][moidx[0]], mo_energy[1][moidx[1]])
    return np.asarray(mo_energy)

def _mo_without_core(gw, mo):
    moidx = get_frozen_mask(gw)
    mo = (mo[0][:,moidx[0]], mo[1][:,moidx[1]])
    return np.asarray(mo)

def as_scanner(gw):
    '''Generating a scanner/solver for GW PES.'''
    if isinstance(gw, lib.SinglePointScanner):
        return gw

    logger.info(gw, 'Set %s as a scanner', gw.__class__)

    class GW_Scanner(gw.__class__, lib.SinglePointScanner):
        def __init__(self, gw):
            self.__dict__.update(gw.__dict__)
            self._scf = gw._scf.as_scanner()
        def __call__(self, mol_or_geom, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            self.reset(mol)

            mf_scanner = self._scf
            mf_scanner(mol)
            self.mo_coeff = mf_scanner.mo_coeff
            self.mo_occ = mf_scanner.mo_occ
            self.kernel(**kwargs)
            return self.e_tot
    return GW_Scanner(gw)


class UGWAC(GWAC):

    linearized = getattr(__config__, 'ugw_ac_UGWAC_linearized', False)
    # Analytic continuation: pade or twopole
    ac = getattr(__config__, 'ugw_ac_UGWAC_ac', 'pade')

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
        logger.info(self, 'use perturbative linearized QP eqn = %s', self.linearized)
        logger.info(self, 'analytic continuation method = %s', self.ac)
        return self

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    as_scanner = as_scanner

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, orbs=None, nw=100, vhf_df=False):
        """
        Args:
            mo_energy : 2D array (2, nmo), mean-field mo energy
            mo_coeff : 3D array (2, nmo, nmo), mean-field mo coefficient
            Lpq : 4D array (2, naux, nmo, nmo), 3-index ERI
            orbs: list, orbital indices
            nw: interger, grid number
            vhf_df: bool, use density fitting for HF exchange or not

        Returns:
            self.mo_energy : GW quasiparticle energy
        """
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self._scf.mo_energy)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.converged, self.mo_energy, self.mo_coeff = \
                kernel(self, mo_energy, mo_coeff,
                       Lpq=Lpq, orbs=orbs, nw=nw, vhf_df=vhf_df, verbose=self.verbose)

        logger.warn(self, 'GW QP energies may not be sorted from min to max')
        logger.timer(self, 'GW', *cput0)
        return self.mo_energy

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
    mf.xc = 'pbe0'
    mf.kernel()

    nocca = (mol.nelectron + mol.spin) // 2
    noccb = mol.nelectron - nocca
    nmo = len(mf.mo_energy[0])
    nvira = nmo - nocca
    nvirb = nmo - noccb

    gw = UGWAC(mf)
    gw.linearized = False
    gw.ac = 'pade'
    gw.kernel(orbs=range(nocca-3, nocca+3))
    print(gw.mo_energy)
    assert abs(gw.mo_energy[0][nocca-1] - -0.521932084529) < 1e-5
    assert abs(gw.mo_energy[0][nocca] - 0.167547592784) < 1e-5
    assert abs(gw.mo_energy[1][noccb-1] - -0.464605523684) < 1e-5
    assert abs(gw.mo_energy[1][noccb] - -0.0133557793765) < 1e-5
