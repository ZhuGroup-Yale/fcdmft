"""
PBC gamma-point spin-unrestricted G0W0 method based on the analytic continuation scheme.
This implementation has N^4 scaling, and is faster than GW-CD (N^4~N^5)
and fully analytic GW (N^6) methods.
GW-AC is recommended for valence states only, and is inaccurate for core states.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14 053020 (2012)
"""

from functools import reduce
import numpy as np
from scipy.optimize import newton
import time

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.pbc import scf
from pyscf.mp.ump2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__
from fcdmft.gw.mol.gw_ac import _get_scaled_legendre_roots, two_pole, AC_twopole_diag, pade_thiele, \
        AC_pade_thiele_diag, AC_twopole_full, AC_pade_thiele_full
from fcdmft.gw.mol.ugw_ac import _mo_energy_without_core, _mo_without_core, get_sigma_diag, UGWAC, as_scanner
from fcdmft.gw.mol.ugw_gf import get_sigma_full, get_g0

einsum = lib.einsum

def kernel(gw, gfomega, mo_energy, mo_coeff, Lpq=None, orbs=None,
           nw=None, vhf_df=False, is_metal=False, verbose=logger.NOTE):
    """
    GW-corrected quasiparticle orbital energies

    Args:
        Lpq : density fitting 3-center integral in MO basis.
        orbs : a list of orbital indices, default is range(nmo).
        nw : number of frequency point on imaginary axis.
        vhf_df : using density fitting integral to compute HF exchange.
        is_metal: being used in DMFT for metals.

    Returns:
        A list :  gf, gf0, sigma
    """
    mf = gw._scf
    # only support frozen core
    if gw.frozen is not None:
        assert isinstance(gw.frozen, int)

    nocca, noccb = gw.nocc
    nmoa, nmob = gw.nmo

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
        uhf = scf.UHF(gw.mol).density_fit()
        uhf.exxdiv = None
        uhf.with_df = gw.with_df
        uhf.with_df._cderi = gw.with_df._cderi
        vk = uhf.get_veff(gw.mol, dm)
        vj = uhf.get_j(gw.mol, dm)
        vk[0] = vk[0] - (vj[0] + vj[1])
        vk[1] = vk[1] - (vj[0] + vj[1])
        vk_frz = np.zeros((2, nmoa, nmoa))
        for s in range(2):
            vk_frz[s] = reduce(np.dot, (mo_coeff[s].T, vk[s], mo_coeff[s]))
        vk = vk_frz

    # Grids for integration on imaginary axis
    if is_metal and nw < 400:
        nw = 400
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
            if is_metal:
                nk_pade = 18; ratio_pade = 5./6.
                coeff_a, omega_fit_a = AC_pade_thiele_full(sigmaI[0], omega, npts=nk_pade, step_ratio=ratio_pade)
                coeff_b, omega_fit_b = AC_pade_thiele_full(sigmaI[1], omega, npts=nk_pade, step_ratio=ratio_pade)
            else:
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
            if is_metal:
                nk_pade = 18; ratio_pade = 5./6.
                coeff_a, omega_fit_a = AC_pade_thiele_diag(sigmaI[0], omega, npts=nk_pade, step_ratio=ratio_pade)
                coeff_b, omega_fit_b = AC_pade_thiele_diag(sigmaI[1], omega, npts=nk_pade, step_ratio=ratio_pade)
            else:
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
    v_mf_frz = np.zeros((2, nmoa, nmoa))
    for s in range(2):
        v_mf_frz[s] = reduce(np.dot, (mf.mo_coeff[s].T, v_mf[s], mf.mo_coeff[s]))
    v_mf = v_mf_frz

    # v_hf from DFT/HF density
    dm = mf.make_rdm1()
    uhf = scf.UHF(gw.mol).density_fit()
    uhf.exxdiv = None
    uhf.with_df = gw.with_df
    uhf.with_df._cderi = gw.with_df._cderi
    vk = uhf.get_veff(gw.mol, dm)
    vj = uhf.get_j(gw.mol, dm)
    vk[0] = vk[0] - (vj[0] + vj[1])
    vk[1] = vk[1] - (vj[0] + vj[1])
    vk_frz = np.zeros((2, nmoa, nmoa))
    for s in range(2):
        vk_frz[s] = reduce(np.dot, (mf.mo_coeff[s].T, vk[s], mf.mo_coeff[s]))
    vk = vk_frz

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
    v_mf_frz = np.zeros((2, nmoa, nmoa))
    for s in range(2):
        v_mf_frz[s] = reduce(np.dot, (mf.mo_coeff[s].T, v_mf[s], mf.mo_coeff[s]))
    v_mf = v_mf_frz

    # v_hf from DFT/HF density
    dm = mf.make_rdm1()
    uhf = scf.UHF(gw.mol).density_fit()
    uhf.exxdiv = None
    uhf.with_df = gw.with_df
    uhf.with_df._cderi = gw.with_df._cderi
    vk = uhf.get_veff(gw.mol, dm)
    vj = uhf.get_j(gw.mol, dm)
    vk[0] = vk[0] - (vj[0] + vj[1])
    vk[1] = vk[1] - (vj[0] + vj[1])
    vk_frz = np.zeros((2, nmoa, nmoa))
    for s in range(2):
        vk_frz[s] = reduce(np.dot, (mf.mo_coeff[s].T, vk[s], mf.mo_coeff[s]))
    vk = vk_frz

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
        if abs(mf.kpt).max() > 1e-9:
            raise NotImplementedError
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(mf)

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

    def kernel(self, omega, mo_energy=None, mo_coeff=None, Lpq=None, orbs=None, nw=100, vhf_df=False, is_metal=False):
        """
        Args:
            mo_energy : 1D array (nmo), mean-field mo energy
            mo_coeff : 2D array (nmo, nmo), mean-field mo coefficient
            Lpq : 3D array (naux, nmo, nmo), 3-index ERI
            orbs: list, orbital indices
            nw: integer, grid number
            vhf_df: bool, use density fitting for HF exchange or not
            is_metal: being used in DMFT for metals

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
                   Lpq=Lpq, orbs=orbs, nw=nw, vhf_df=vhf_df, is_metal=is_metal, verbose=self.verbose)

        logger.timer(self, 'UGWGF', *cput0)
        return self.gf, self.gf0, self.sigma

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmoa, nmob = self.nmo
        nao = self.mo_coeff[0].shape[0]
        naux = self.with_df.get_naoaux()
        kpts = self._scf.with_df.kpts
        mem_incore = nmoa**2*naux * 8 /1e6
        max_memory = max(2000, self._scf.max_memory-lib.current_memory()[0]-nao**2*naux*8/1e6)

        mo_a = np.asarray(mo_coeff[0], order='F')
        mo_b = np.asarray(mo_coeff[1], order='F')
        ijslice_a = (0, nmoa, 0, nmoa)
        ijslice_b = (0, nmob, 0, nmob)

        from pyscf.pbc.df.fft_ao2mo import _format_kpts
        kptijkl = _format_kpts(kpts)
        eri_3d = []
        Lpq_a, Lpq_b = [], []
        for LpqR, _, _ in self._scf.with_df.sr_loop(kptijkl[:2], max_memory=0.3*max_memory, compact=False):
            tmp_a, tmp_b = None, None
            tmp_a = _ao2mo.nr_e2(LpqR.reshape(-1,nao,nao), mo_a, ijslice_a, aosym='s1', mosym='s1', out=tmp_a)
            tmp_b = _ao2mo.nr_e2(LpqR.reshape(-1,nao,nao), mo_b, ijslice_b, aosym='s1', mosym='s1', out=tmp_b)
            Lpq_a.append(tmp_a)
            Lpq_b.append(tmp_b)
        Lpq_a = np.vstack(Lpq_a).reshape(-1,nmoa,nmoa)
        Lpq_b = np.vstack(Lpq_b).reshape(-1,nmob,nmob)
        eri_3d = np.asarray([Lpq_a, Lpq_b])

        return eri_3d
