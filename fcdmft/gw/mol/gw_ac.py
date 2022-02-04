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
from scipy.optimize import newton, least_squares

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df, dft, scf
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__

einsum = lib.einsum

# ****************************************************************************
# core routines, kernel, sigma, rho_response
# ****************************************************************************

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
        assert gw.frozen < gw.nocc

    if Lpq is None:
        Lpq = gw.ao2mo(mo_coeff)
    if orbs is None:
        orbs = range(gw.nmo)
    if orbs is not None and gw.frozen is not None:
        orbs = [x - gw.frozen for x in orbs]
        if orbs[0] < 0:
            raise RuntimeError('GW orbs must be larger than frozen core!')

    # v_xc
    v_mf = mf.get_veff() - mf.get_j()
    v_mf = reduce(np.dot, (mo_coeff.T, v_mf, mo_coeff))

    nmo  = gw.nmo
    nocc = gw.nocc
    nvir = nmo - nocc

    # v_hf from DFT/HF density
    if vhf_df and gw.frozen is None:
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

    # Compute self-energy on imaginary axis i*[0,iw_cutoff]
    sigmaI, omega = get_sigma_diag(gw, orbs, Lpq, freqs, wts, iw_cutoff=5.0)

    # Analytic continuation
    if gw.ac == 'twopole':
        coeff = AC_twopole_diag(sigmaI, omega, orbs, nocc)
    elif gw.ac == 'pade':
        coeff, omega_fit = AC_pade_thiele_diag(sigmaI, omega, \
                step_ratio=2.0/3.0)
    else:
        raise ValueError("Unknown GW-AC type %s"%(str(gw.ac)))

    conv = True
    mf_mo_energy = np.array(mo_energy, copy=True)
    ef = (mo_energy[nocc-1] + mo_energy[nocc]) * 0.5
    mo_energy = np.zeros_like(gw._scf.mo_energy)
    for p in orbs:
        if gw.linearized:
            # linearized G0W0
            de = 1e-6
            ep = mf_mo_energy[p]
            #TODO: analytic sigma derivative
            if gw.ac == 'twopole':
                sigmaR = two_pole(ep-ef, coeff[:, p-orbs[0]]).real
                dsigma = two_pole(ep-ef+de, coeff[:, p-orbs[0]]).real - sigmaR.real
            elif gw.ac == 'pade':
                sigmaR = pade_thiele(ep-ef, omega_fit, coeff[:,p-orbs[0]]).real
                dsigma = pade_thiele(ep-ef+de, omega_fit, coeff[:,p-orbs[0]]).real - sigmaR.real
            zn = 1.0 / (1.0 - dsigma / de)
            e = ep + zn * (sigmaR.real + vk[p, p] - v_mf[p, p])
            if gw.frozen is not None:
                mo_energy[p + gw.frozen] = e
            else:
                mo_energy[p] = e
        else:
            # self-consistently solve QP equation
            def quasiparticle(omega):
                if gw.ac == 'twopole':
                    sigmaR = two_pole(omega-ef, coeff[:, p-orbs[0]]).real
                elif gw.ac == 'pade':
                    sigmaR = pade_thiele(omega-ef, omega_fit, coeff[:, p-orbs[0]]).real
                return omega - mf_mo_energy[p] - (sigmaR.real + vk[p, p] - v_mf[p, p])
            try:
                e = newton(quasiparticle, mf_mo_energy[p], tol=1e-6, maxiter=100)
                if gw.frozen is not None:
                    mo_energy[p + gw.frozen] = e
                else:
                    mo_energy[p] = e
            except RuntimeError:
                conv = False

    with np.printoptions(threshold=len(mo_energy)):
        logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)

    return conv, mo_energy, mo_coeff

def get_rho_response(omega, mo_energy, Lpq):
    """
    Compute density response function in auxiliary basis at freq iw.
    """
    naux, nocc, nvir = Lpq.shape
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    eia = eia / (omega**2 + eia * eia)
    # Response from both spin-up and spin-down density
    Pia = Lpq * (eia * 4.0)
    Pi = einsum('Pia, Qia -> PQ', Pia, Lpq)
    return Pi

def get_sigma_diag(gw, orbs, Lpq, freqs, wts, iw_cutoff=None):
    """
    Compute GW correlation self-energy (diagonal elements)
    in MO basis on imaginary axis
    """
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

    # Integration on numerical grids
    if iw_cutoff is not None:
        nw_sigma = sum(freqs < iw_cutoff) + 1
    else:
        nw_sigma = nw + 1
    
    omega = np.zeros((nw_sigma),dtype=np.complex128)
    omega[1:] = 1j*freqs[:(nw_sigma-1)]
    emo = omega[None] + ef - mo_energy[:, None]

    sigma = np.zeros((norbs, nw_sigma),dtype=np.complex128)
    for w in range(nw):
        # Pi_inv = 1 - (1 - Pi)^{-1} - 1 = -[1 + (Pi - 1)^{-1}]
        Pi = get_rho_response(freqs[w], mo_energy, Lpq[:, :nocc, nocc:])
        Pi[range(naux), range(naux)] -= 1.0
        Pi_inv = np.linalg.inv(Pi)
        Pi_inv[range(naux), range(naux)] += 1.0
        Qnm = einsum('Pnm, PQ -> Qnm', Lpq[:, orbs], Pi_inv)
        Wmn = einsum('Qnm, Qmn -> mn', Qnm, Lpq[:, :, orbs])
        g0 = wts[w] * emo / (emo**2 + freqs[w]**2)
        sigma += einsum('mn, mw -> nw', Wmn, g0) / np.pi

    return sigma, omega

# ****************************************************************************
# frequency integral quadrature, legendre, clenshaw_curtis
# ****************************************************************************

def _get_scaled_legendre_roots(nw, x0=0.5):
    """
    Scale nw Legendre roots, which lie in the
    interval [-1, 1], so that they lie in [0, inf)
    Ref: www.cond-mat.de/events/correl19/manuscripts/ren.pdf

    Returns:
        freqs : 1D array
        wts : 1D array
    """
    freqs, wts = np.polynomial.legendre.leggauss(nw)
    freqs_new = x0 * (1.0 + freqs) / (1.0 - freqs)
    wts = wts * 2.0 * x0 / (1.0 - freqs)**2
    return freqs_new, wts

def _get_clenshaw_curtis_roots(nw):
    """
    Clenshaw-Curtis qaudrature on [0,inf)
    Ref: J. Chem. Phys. 132, 234114 (2010)
    Returns:
        freqs : 1D array
        wts : 1D array
    """
    freqs = np.zeros(nw)
    wts = np.zeros(nw)
    a = 0.2
    for w in range(nw):
        t = (w + 1.0) / nw * np.pi * 0.5
        freqs[w] = a / np.tan(t)
        if w != nw - 1:
            wts[w] = a*np.pi * 0.5 / nw / (np.sin(t)**2)
        else:
            wts[w] = a*np.pi * 0.25 / nw / (np.sin(t)**2)
    return freqs[::-1], wts[::-1]

# ****************************************************************************
# AC routines, two_pole, pade
# ****************************************************************************

def two_pole_fit(coeff, omega, sigma):
    cf = coeff[:5] + 1j * coeff[5:]
    f = cf[0] + cf[1] / (omega+cf[3]) + cf[2] / (omega+cf[4]) - sigma
    f[0] = f[0] / 0.01
    return np.array([f.real,f.imag]).reshape(-1)

def two_pole(freqs, coeff):
    cf = coeff[:5] + 1j * coeff[5:]
    return cf[0] + cf[1] / (freqs+cf[3]) + cf[2] / (freqs+cf[4])

def AC_twopole_diag(sigma, omega, orbs, nocc):
    """
    Analytic continuation to real axis using a two-pole model

    Args:
        sigma : 2D array (norbs, nomega)
        omega : 1D array (nomega)
        orbs : list
        nocc : integer

    Returns:
        coeff: 2D array (ncoeff, norbs)
    """
    norbs, nw = sigma.shape
    coeff = np.zeros((10, norbs))
    for p in range(norbs):
        target = np.array([sigma[p].real, sigma[p].imag]).reshape(-1)
        # randonly generated initial guess
        if orbs[p] < nocc:
            x0 = np.array([0, 1, 1, 1, -1, 0, 0, 0, -1.0, -0.5])
        else:
            x0 = np.array([0, 1, 1, 1, -1, 0, 0, 0, 1.0, 0.5])
        #TODO: analytic gradient
        xopt = least_squares(two_pole_fit, x0, jac='3-point', method='trf', xtol=1e-10,
                             gtol = 1e-10, max_nfev=1000, verbose=0, args=(omega, sigma[p]))
        if not xopt.success:
            log = logger.Logger()
            log.warn('2P-Fit Orb %d not converged, cost function %e'%(p,xopt.cost))
        coeff[:, p] = xopt.x.copy()
    return coeff

def AC_twopole_full(sigma, omega, orbs, nocc):
    """
    Analytic continuation to real axis using a two-pole model

    Args:
        sigma : 3D array (norbs, norbs, nomega)
        omega : 1D array (nomega)
        orbs : list
        nocc : integer

    Returns:
        coeff: 3D array (ncoeff, norbs, norbs)
    """
    norbs, norbs, nw = sigma.shape
    coeff = np.zeros((10, norbs, norbs))
    for p in range(norbs):
        for q in range(norbs):
            target = np.array([sigma[p, q].real, sigma[p, q].imag]).reshape(-1)
            if orbs[p] < nocc:
                x0 = np.array([0, 1, 1, 1, -1, 0, 0, 0, -1.0, -0.5])
            else:
                x0 = np.array([0, 1, 1, 1, -1, 0, 0, 0, 1.0, 0.5])
            #TODO: analytic gradient
            xopt = least_squares(two_pole_fit, x0, jac='3-point', method='trf', xtol=1e-10,
                             gtol = 1e-10, max_nfev=1000, verbose=0, args=(omega, sigma[p, q]))
            if not xopt.success:
                log = logger.Logger()
                log.warn('2P-Fit Orbs (%d,%d) not converged, cost function %e'%(p, q, xopt.cost))
            coeff[:, p, q] = xopt.x.copy()
    return coeff

def thiele(fn, zn):
    nfit = len(zn)
    g = np.zeros((nfit, nfit), dtype=np.complex128)
    g[:, 0] = fn.copy()
    for i in range(1, nfit):
        g[i:, i] = (g[i-1, i-1] - g[i:,i-1]) / ((zn[i:] - zn[i-1]) * g[i:,i-1])
    a = g.diagonal()
    return a

def pade_thiele(freqs, zn, coeff):
    nfit = len(coeff)
    X = coeff[-1] * (freqs - zn[-2])
    for i in range(nfit - 1):
        idx = nfit - i - 1
        X = coeff[idx] * (freqs - zn[idx - 1]) / (1.0 + X)
    X = coeff[0] / (1.0 + X)
    return X

def _get_ac_idx(nw, npts=18, step_ratio=2.0/3.0, idx_start=1):
    """
    Get an array of indices, with stepsize decreasing.

    Args:
        nw : number of frequency points
        npts : final number of selected points
        step_ratio : final stepsize / initial stepsize.
        idx_start : first index of final array

    Returns:
        idx : an array for indexing frequency and omega.
    """
    if nw <= npts:
        raise ValueError("nw (%s) should be larger than npts (%s)" %(nw, npts))
    steps = np.linspace(1.0, step_ratio, npts)
    steps /= np.sum(steps)
    steps = np.cumsum(steps * nw) 
    steps += (idx_start - steps[0])
    steps = np.round(steps).astype(np.int)
    return steps

def AC_pade_thiele_diag(sigma, omega, npts=18, step_ratio=2.0/3.0):
    """
    Analytic continuation to real axis using a Pade approximation
    from Thiele's reciprocal difference method
    Reference: J. Low Temp. Phys. 29, 179 (1977)

    Args:
        sigma : 2D array (norbs, nomega)
        omega : 1D array (nomega)

    Returns:
        coeff : 2D array (ncoeff, norbs)
        omega : 1D array (ncoeff)
    """
    idx = _get_ac_idx(omega.shape[-1], npts=npts, step_ratio=step_ratio)
    omega = omega[idx]
    sigma = sigma[:, idx]
    norbs, nw = sigma.shape
    npade = nw // 2 # take even number of points.
    coeff = np.zeros((npade*2, norbs),dtype=np.complex128)
    for p in range(norbs):
        coeff[:, p] = thiele(sigma[p,:npade*2], omega[:npade*2])
    return coeff, omega[:npade*2]

def AC_pade_thiele_full(sigma, omega, npts=18, step_ratio=2.0/3.0):
    """
    Analytic continuation to real axis using a Pade approximation

    Args:
        sigma : 3D array (norbs, norbs, nomega)
        omega : 1D array (nomega)

    Returns:
        coeff: 2D array (ncoeff, norbs, norbs)
        omega : 1D array (ncoeff)
    """
    idx = _get_ac_idx(omega.shape[-1], npts=npts, step_ratio=step_ratio)
    omega = omega[idx]
    sigma = sigma[:,:,idx]
    norbs, norbs, nw = sigma.shape
    npade = nw // 2 # take even number of points.
    coeff = np.zeros((npade*2, norbs, norbs),dtype=np.complex128)
    for p in range(norbs):
        for q in range(norbs):
            coeff[:,p,q] = thiele(sigma[p,q,:npade*2], omega[:npade*2])
    return coeff, omega[:npade*2]

def _mo_energy_without_core(gw, mo_energy):
    return mo_energy[get_frozen_mask(gw)]

def _mo_without_core(gw, mo):
    return mo[:,get_frozen_mask(gw)]

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

class GWAC(lib.StreamObject):

    linearized = getattr(__config__, 'gw_ac_GWAC_linearized', False)
    # Analytic continuation: pade or twopole
    ac = getattr(__config__, 'gw_ac_GWAC_ac', 'pade')

    def __init__(self, mf, frozen=None, auxbasis=None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.frozen = frozen

        # DF-GW must use density fitting integrals
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            if auxbasis is not None:
                self.with_df.auxbasis = auxbasis
            else:
                try:
                    self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
                except:
                    self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=False)
        self._keys.update(['with_df'])

##################################################
# don't modify the following attributes, they are not input options
        self._nocc = None
        self._nmo = None
        # self.mo_energy: GW quasiparticle energy, not scf mo_energy
        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.sigma = None

        keys = set(('linearized','ac'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not None:
            log.info('frozen orbitals = %d', self.frozen)
        logger.info(self, 'use perturbative linearized QP eqn = %s', self.linearized)
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

    as_scanner = as_scanner

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, orbs=None, nw=100, vhf_df=False):
        """
        Args:
            mo_energy : 1D array (nmo), mean-field mo energy
            mo_coeff : 2D array (nmo, nmo), mean-field mo coefficient
            Lpq : 3D array (naux, nmo, nmo), 3-index ERI
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
                kernel(self, mo_energy, mo_coeff, Lpq=Lpq, orbs=orbs, nw=nw, \
                vhf_df=vhf_df, verbose=self.verbose)

        logger.warn(self, 'GW QP energies may not be sorted from min to max')
        logger.timer(self, 'GW', *cput0)
        return self.mo_energy

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        naux = self.with_df.get_naoaux()
        mem_incore = (2 * nmo**2*naux) * 8 / 1e6
        mem_now = lib.current_memory()[0]

        mo = np.asarray(mo_coeff, order='F')
        ijslice = (0, nmo, 0, nmo)
        Lpq = None
        if (mem_incore + mem_now < 0.99 * self.max_memory) or self.mol.incore_anyway:
            Lpq = _ao2mo.nr_e2(self.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
            return Lpq.reshape(naux, nmo, nmo)
        else:
            logger.warn(self, 'Memory may not be enough!')
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
    nmo = mf.mo_energy.size
    nvir = nmo-nocc

    gw = GWAC(mf)
    gw.linearized = False
    gw.ac = 'pade'
    gw.kernel(orbs=range(nocc - 3, nocc + 3))
    print (gw.mo_energy)
    assert abs(gw.mo_energy[nocc-1] - -0.412849230989) < 1e-5
    assert abs(gw.mo_energy[nocc] - 0.165745160102) < 1e-5
