"""
Spin-restricted fully self-consistent GW method.

References:
    Phys. Rev. B 98, 155143 (2018)
    J. Phys. Chem. Lett. 9, 2, 306-312 (2018)
    J. Chem. Theory Comput. 17, 3, 1662-1677 (2021)
    Phys. Rev. B 94, 165109 (2016)
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import newton
import time

from pyscf import dft, scf
from pyscf.lib import logger, einsum, temporary_env, diis

from fcdmft.ac.pade import PadeAC
from fcdmft.ac.two_pole import TwoPoleAC
from fcdmft.gw.mol.gw_ac import _mo_energy_without_core, _mo_without_core
from fcdmft.gw.mol.gw_space_time import GWSpaceTime, get_g0_t, get_pi_t, get_wc_w, get_imag_sigma_t, \
    get_minimax_grid, get_ft_weight

def kernel(gw, Lpq_ao=None):
    """run full self-consistent GW calculation.

    Parameters
    ----------
    gw : fcdmft.gw.mol.scgw.SCGW
        SCGW object
    Lpq_ao : double 3d array, optional
        3-center DF matrix in AO space, by default None
    """
    # local variables for convenience
    nmo = gw.nmo
    nocc = gw.nocc
    ngrid = gw.ngrid
    ovlp = gw._scf.get_ovlp()
    if Lpq_ao is None:
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            gw.with_df.get_naoaux()

    # initialize HF Hamiltonian and Fermi level
    rhf = gw._scf if (not isinstance(gw._scf, dft.rks.RKS)) and isinstance(gw._scf, scf.hf.RHF) else gw._scf.to_hf()
    gw.dm_ao = dm_ao = gw._scf.make_rdm1()
    mo_energy_hf = np.array(gw._scf.mo_energy, copy=True)
    C_ao_hfmo = np.array(gw._scf.mo_coeff, copy=True)
    C_ao_no = np.array(gw._scf.mo_coeff, copy=True)
    mo_energy_ef = np.array(gw._scf.mo_energy, copy=True)

    if gw.fix_ef is not None:
        gw.ef = ef = gw.fix_ef
    else:
        gw.ef = ef = (mo_energy_hf[nocc - 1] + mo_energy_hf[nocc]) * 0.5

    gw.E_min = E_min = (mo_energy_ef[nocc] - mo_energy_ef[nocc - 1]) * gw.E_min_scale
    gw.E_max = E_max = (mo_energy_ef[-1] - mo_energy_ef[0]) * gw.E_max_scale
    t, t_wt, w, w_wt = get_minimax_grid(k=ngrid, E_min=E_min, E_max=E_max)
    wt_t_to_w_cos, wt_t_to_w_sin, wt_w_to_t_cos, wt_w_to_t_sin = get_ft_weight(E_min=E_min, E_max=E_max, t=t, w=w)
    gw.t, gw.t_wt, gw.w, gw.w_wt = t, t_wt, w, w_wt
    gw.wt_t_to_w_cos, gw.wt_t_to_w_sin, gw.wt_w_to_t_cos, gw.wt_w_to_t_sin = \
        wt_t_to_w_cos, wt_t_to_w_sin, wt_w_to_t_cos, wt_w_to_t_sin

    # G(it) is initialized as GHF(it)
    gc_t_pos_hfmo = np.zeros(shape=[ngrid, nmo, nmo], dtype=np.double)
    gc_t_neg_hfmo = np.zeros(shape=[ngrid, nmo, nmo], dtype=np.double)
    g_t_pos_ao, g_t_neg_ao = get_g0_t(nocc=nocc, mo_energy=mo_energy_hf, t=t, ef=ef, mo_coeff=C_ao_hfmo)
    g_t_pos_no, g_t_neg_no = get_g0_t(nocc=nocc, mo_energy=mo_energy_hf, t=t, ef=ef)

    if gw.diis is True:
        # DIIS for positive and negative Green's function separately
        gw_diis_pos = diis.DIIS(gw)
        gw_diis_pos.space = gw.diis_space
        gw_diis_neg = diis.DIIS(gw)
        gw_diis_neg.space = gw.diis_space
    else:
        gw_diis_pos = gw_diis_neg = None
    diis_start_cycle = gw.diis_start_cycle

    cycle = 0
    e_tot = 0.0
    while gw.conv is False and cycle < max(1, gw.max_cycle):
        logger.info(gw, '\nbegin scGW cycle: %d', cycle + 1)
        dm_ao_old = dm_ao.copy()
        e_tot_old = e_tot
        if Lpq_ao is None:
            Lpq_no = gw.ao2mo(mo_coeff=C_ao_no)
        else:
            Lpq_no = einsum('Lmn,mp,nq->Lpq', Lpq_ao, C_ao_no, C_ao_no)

        # P(it) = G(it)G(-it), Pi(it)=P(it)V
        Pi_t_no = get_pi_t(g_t_pos=g_t_pos_no, g_t_neg=g_t_neg_no, Lpq=Lpq_no)

        # Pi(iw) = FT[Pi(it)]
        Pi_w_no = einsum('tPQ,tw->wPQ', Pi_t_no, np.cos(gw.w[None, :] * gw.t[:, None]) * gw.wt_t_to_w_cos)
        gw.FT_test_bosonic(Pi_w_no.transpose(1, 2, 0), 'Pi(iw)<->Pi(it)')

        # Wc^-1(iw) = [1-Pi(iw)]^-1 - 1
        wc_w_no = get_wc_w(Pi_w=Pi_w_no)
        wc_t_no = einsum('wPQ,wt->tPQ', wc_w_no, np.cos(gw.w[:, None] * gw.t[None, :]) * gw.wt_w_to_t_cos)
        gw.FT_test_bosonic(wc_w_no, 'Wc(iw)<->Wc(it)')

        # Sigma(it)=G(it)W(it), Sigma(iw)=FT[Sigma(it)]
        sigma_t_pos_no, sigma_t_neg_no = get_imag_sigma_t(
            g_t_pos=g_t_pos_no, g_t_neg=g_t_neg_no, wc_t=wc_t_no, Lpq=Lpq_no, fullsigma=True
        )
        sigma_t_cos_no = 0.5 * (sigma_t_pos_no + sigma_t_neg_no)
        sigma_t_sin_no = 0.5 * (sigma_t_neg_no - sigma_t_pos_no)
        sigma_w_cos_no = einsum('tpq,tw->wpq', sigma_t_cos_no, np.cos(gw.w[None, :] * gw.t[:, None]) * gw.wt_t_to_w_cos)
        sigma_w_sin_no = einsum('tpq,tw->wpq', sigma_t_sin_no, np.sin(gw.w[None, :] * gw.t[:, None]) * gw.wt_t_to_w_sin)
        sigma_w_no = sigma_w_cos_no + 1j * sigma_w_sin_no
        gw.FT_test_fermionic(sigma_w_no, 'Sigma(iw)<->Sigma(it)')

        # check scGW energy
        ec_pos = einsum('tij,tji,t->', g_t_pos_no, sigma_t_neg_no, gw.t_wt)
        ec_neg = einsum('tij,tji,t->', g_t_neg_no, sigma_t_pos_no, gw.t_wt)
        e_c = (ec_pos + ec_neg).real
        with temporary_env(rhf, verbose=0):
            e_hf = rhf.energy_elec(dm=dm_ao)[0] + rhf.energy_nuc()
        e_tot = e_hf + e_c
        logger.info(gw, 'EHF = %.8f, Ec = %.8f, E_total = %.8f', e_hf, e_c, e_tot)

        # get new Gc(iw) = G(iw) - GHF(iw)
        # G(iw) = [iw + ef - HF - Sigma(iw)]^-1 and GHF(iw) = [iw + ef - HF]^-1
        C_no_hf = C_ao_no.T @ ovlp @ C_ao_hfmo
        sigma_w_hfmo = C_no_hf.T @ sigma_w_no @ C_no_hf
        ghf_w_hfmo = np.zeros(shape=[ngrid, nmo, nmo], dtype=np.complex128)
        g_w_hfmo = np.zeros(shape=[ngrid, nmo, nmo], dtype=np.complex128)
        ham_hf_ao = rhf.get_fock(dm=dm_ao)
        ham_hf_hfmo = C_ao_hfmo.T @ ham_hf_ao @ C_ao_hfmo
        for iw in range(ngrid):
            ghf_w_hfmo[iw] = np.linalg.inv((1j * gw.w[iw] + gw.ef) * np.eye(nmo) - ham_hf_hfmo)
            g_w_hfmo[iw] = np.linalg.inv((1j * gw.w[iw] + gw.ef) * np.eye(nmo) - ham_hf_hfmo - sigma_w_hfmo[iw])
        gc_w_hfmo = g_w_hfmo - ghf_w_hfmo

        gw.g_w_ao = C_ao_hfmo @ g_w_hfmo @ C_ao_hfmo.T

        # get new density matrix
        dm_hfmo = 2.0 / np.pi * einsum('wij,w->ij', gc_w_hfmo, gw.w_wt).real
        dm_hfmo[:nocc, :nocc] += np.eye(nocc) * 2.0
        dm_hfmo = 0.5 * (dm_hfmo + dm_hfmo.T)
        gw.dm_ao = dm_ao = C_ao_hfmo @ dm_hfmo @ C_ao_hfmo.T
        # NOTE: natural orbitals are updated
        C_hfmo_gw = eigh(dm_hfmo)[1]
        gw.mo_coeff = C_ao_no = C_ao_hfmo @ C_hfmo_gw

        energy_diff = e_tot - e_tot_old
        norm_dm = np.linalg.norm(dm_ao - dm_ao_old, ord=1) / nmo / nmo
        logger.info(
            gw, 'scGW cycle=%d dE=%4.3g |ddm|=%4.3g Tr(dm)=%.6f', cycle + 1, energy_diff, norm_dm, np.trace(dm_hfmo)
        )
        if norm_dm < gw.conv_tol_dm and abs(energy_diff) < gw.conv_tol_energy:
            gw.conv = True
            break

        # Gc(it) = FT[Gc(iw)]
        gc_t_pos_hfmo = einsum('wpq,wt->tpq', gc_w_hfmo.real, np.cos(gw.w[:, None] * gw.t[None, :]) * gw.wt_w_to_t_cos)
        gc_t_pos_hfmo -= einsum('wpq,wt->tpq', gc_w_hfmo.imag, np.sin(gw.w[:, None] * gw.t[None, :]) * gw.wt_w_to_t_sin)
        gc_t_neg_hfmo = einsum('wpq,wt->tpq', gc_w_hfmo.real, np.cos(gw.w[:, None] * gw.t[None, :]) * gw.wt_w_to_t_cos)
        gc_t_neg_hfmo += einsum('wpq,wt->tpq', gc_w_hfmo.imag, np.sin(gw.w[:, None] * gw.t[None, :]) * gw.wt_w_to_t_sin)
        gw.FT_test_fermionic(gc_w_hfmo, 'Gc(w)<->Gc(t)')

        gc_t_pos_ao = C_ao_hfmo @ gc_t_pos_hfmo @ C_ao_hfmo.T
        gc_t_neg_ao = C_ao_hfmo @ gc_t_neg_hfmo @ C_ao_hfmo.T

        # NOTE: HF canonical orbitals are updated
        ham_hf_ao = rhf.get_fock(dm=dm_ao)
        mo_energy_hf, C_ao_hfmo = eigh(ham_hf_ao, ovlp)
        logger.info(gw, 'HF HOMO = %.6f , LUMO = %.6f', mo_energy_hf[nocc - 1], mo_energy_hf[nocc])
        # NOTE: Fermi level is updated
        if gw.fix_ef is None:
            gw.ef = ef = (mo_energy_hf[nocc - 1] + mo_energy_hf[nocc]) * 0.5
            logger.info(gw, 'updated ef = %.6f', ef)
        ghf_t_pos_hfmo, ghf_t_neg_hfmo = get_g0_t(nocc=nocc, mo_energy=mo_energy_hf, t=gw.t, ef=gw.ef)
        ghf_t_pos_ao = C_ao_hfmo @ ghf_t_pos_hfmo @ C_ao_hfmo.T
        ghf_t_neg_ao = C_ao_hfmo @ ghf_t_neg_hfmo @ C_ao_hfmo.T

        # DIIS update
        g_t_pos_ao_old, g_t_neg_ao_old = g_t_pos_ao.copy(), g_t_neg_ao.copy()
        g_t_pos_ao = ghf_t_pos_ao + gc_t_pos_ao
        g_t_neg_ao = ghf_t_neg_ao + gc_t_neg_ao
        if gw.damp > 1.0e-4 and (0 <= cycle < diis_start_cycle - 1 or gw.diis is False):
            g_t_pos_ao = gw.damp * g_t_pos_ao + (1 - gw.damp) * g_t_pos_ao_old
            g_t_neg_ao = gw.damp * g_t_neg_ao + (1 - gw.damp) * g_t_neg_ao_old

        def run_diis(g, cycle, adiis, diis_start_cycle):
            if adiis and cycle >= diis_start_cycle:
                g = adiis.update(g)
            return g

        g_t_pos_ao = run_diis(g_t_pos_ao, cycle, gw_diis_pos, diis_start_cycle)
        g_t_neg_ao = run_diis(g_t_neg_ao, cycle, gw_diis_neg, diis_start_cycle)

        # rotate G(it) to natural orbital for next cycle
        for it in range(ngrid):
            g_t_pos_no[it] = C_ao_no.T @ ovlp @ g_t_pos_ao[it] @ ovlp @ C_ao_no
            g_t_neg_no[it] = C_ao_no.T @ ovlp @ g_t_neg_ao[it] @ ovlp @ C_ao_no

        cycle += 1

    logger.debug(gw, 'SCGW %s in %-d cycles.', 'converged' if gw.conv else 'not converged', cycle + 1)

    # analytical continuation to get quasiparticle energy
    C_ao_mfmo = np.array(gw._scf.mo_coeff, copy=True)
    with temporary_env(rhf.mol, verbose=0):
        sigma_hx = C_ao_mfmo.T @ rhf.get_veff(dm=dm_ao) @ C_ao_mfmo
    with temporary_env(gw._scf, verbose=0):
        v_hxc = C_ao_mfmo.T @ gw._scf.get_veff() @ C_ao_mfmo

    C_no_mfmo = C_ao_no.T @ ovlp @ C_ao_mfmo
    sigma_w_mfmo = np.zeros(shape=[ngrid, nmo, nmo], dtype=np.complex128)
    for iw in range(ngrid):
        sigma_w_mfmo[iw] = C_no_mfmo.T @ sigma_w_no[iw] @ C_no_mfmo

    # analytical continuation: sigma(iw)->sigma(w)
    if gw.ac == 'twopole':
        acobj = TwoPoleAC(range(nmo), nocc)
    elif gw.ac == 'pade':
        acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
    acobj.ac_fit(sigma_w_mfmo, 1j * w, axis=0)

    # get GW quasiparticle energy
    diag_acobj = acobj.diagonal(axis1=0, axis2=1)
    mo_energy = np.zeros_like(gw._scf.mo_energy)
    for p in range(nmo):
        if gw.qpe_linearized is True:
            de = 1e-6
            sigmaR = diag_acobj[p].ac_eval(gw._scf.mo_energy[p] - ef).real
            dsigma = diag_acobj[p].ac_eval(gw._scf.mo_energy[p] - ef + de).real - sigmaR.real
            zn = 1.0 / (1.0 - dsigma / de)
            mo_energy[p] = gw._scf.mo_energy[p] + zn * (sigmaR.real + sigma_hx[p, p] - v_hxc[p, p])
        else:

            def quasiparticle(omega):
                sigmaR = diag_acobj[p].ac_eval(omega - ef).real
                return omega - gw._scf.mo_energy[p] - (sigmaR.real + sigma_hx[p, p] - v_hxc[p, p])

            try:
                mo_energy[p] = newton(quasiparticle, gw._scf.mo_energy[p], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter)
            except RuntimeError:
                logger.warn(gw, 'QPE for orbital=%d not converged!', p)

    with np.printoptions(threshold=len(mo_energy)):
        logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)
    logger.warn(gw, 'GW QP energies may not be sorted from min to max')

    # save results
    gw.acobj = acobj
    gw.e_tot, gw.e_hf, gw.e_c = e_tot, e_hf, e_c

    sigma_w_ao = np.zeros(shape=[ngrid, nmo, nmo], dtype=np.complex128)
    sigma_t_pos_ao = np.zeros(shape=[ngrid, nmo, nmo], dtype=np.double)
    sigma_t_neg_ao = np.zeros(shape=[ngrid, nmo, nmo], dtype=np.double)
    for i in range(ngrid):
        sigma_w_ao[i] = ovlp @ C_ao_no @ sigma_w_no[i] @ C_ao_no.T @ ovlp
        sigma_t_pos_ao[i] = ovlp @ C_ao_no @ sigma_t_pos_no[i] @ C_ao_no.T @ ovlp
        sigma_t_neg_ao[i] = ovlp @ C_ao_no @ sigma_t_neg_no[i] @ C_ao_no.T @ ovlp
    gw.ef = ef
    gw.mo_energy = mo_energy
    gw.mo_coeff = C_ao_no
    gw.C_ao_hfmo = C_ao_hfmo
    gw.dm_ao = dm_ao
    gw.sigma_w_ao = sigma_w_ao
    gw.sigma_t_pos_ao = sigma_t_pos_ao
    gw.sigma_t_neg_ao = sigma_t_neg_ao

    return


class SCGW(GWSpaceTime):
    def __init__(self, mf, auxbasis=None):
        GWSpaceTime.__init__(self, mf, auxbasis=auxbasis)
        self.max_cycle = 30  # max scGW cycle
        self.conv_tol_dm = 1.0e-5  # density matrix convergence tolerance
        self.conv_tol_energy = 1.0e-6  # density matrix convergence tolerance
        self.damp = 0.2  # damping factor
        self.diis = True  # use DIIS to update G in each cycle
        self.diis_start_cycle = 5  # DIIS start cycle
        self.diis_space = 5  # DIIS space

        # results
        self.conv = False  # if scGW converges
        self.e_c = None  # scGW correlation energy
        self.e_hf = None  # Hartree-Fock energy evaluated with scGW density matrix
        self.e_tot = None  # scGW total energy
        self.C_ao_hfmo = None  # coefficient from AO to Hartree-Fock canonical orbital
        self.dm_ao = None  # density matrix in AO
        self.g_w_ao = None  # Green's function in AO on imaginary frequency
        self.sigma_w_ao = None  # self-energy in AO on imaginary frequency
        self.sigma_t_neg_ao = None  # self-energy in AO on negative imaginary time
        self.sigma_t_pos_ao = None  # self-energy in AO on negative imaginary time

        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        log.info('nocc = %-d, nvir = %-d', self.nocc, self.nmo - self.nocc)
        log.info('number of minimax grids = %-d', self.ngrid)
        log.info('E_min scale = %-.4f', self.E_min_scale)
        log.info('E_max scale = %-.4f', self.E_max_scale)
        log.info('max cycle = %d', self.max_cycle)
        log.info('density matrix tolerance = %.3e', self.conv_tol_dm)
        log.info('energy tolerance = %.3e', self.conv_tol_energy)
        log.info('damping factor = %.3f', self.damp)
        log.info('use DIIS = %s', self.diis)
        if self.diis is True:
            log.info('DIIS start cycle = %s', self.diis_start_cycle)
            log.info('DIIS space = %s', self.diis_space)
        log.info('use perturbative linearized QP equation = %s', self.qpe_linearized)
        log.info('analytic continuation method = %s', self.ac)
        if self.FT_cycle > 0:
            log.info('FT accuracy test cycle = %d', self.FT_cycle)
            log.info('exit if FT error is large = %d', self.FT_error_stop)
        if self.fix_ef is not None:
            log.info('fixed Fermi level = %.6f', self.fix_ef)
        log.info('')
        return

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq_ao=None):
        """Do a fully self-consistent GW calculation.
        Minimax grids were taken from CP2K, see Refs.[1,2]
        Fourier transform see Refs.[1,3]
        scGW workflow see Ref.[4]

        Reference
        [1] 10.1103/PhysRevB.101.205145
        [2] 10.1103/PhysRevB.109.245101
        [3] 10.1103/PhysRevB.94.165109
        [4] 10.1103/PhysRevB.98.155143

        Parameters
        ----------
        mo_energy : double 1d array, optional
            orbital energy, by default None
        mo_coeff : double 2d array, optional
            coefficient from AO to MO, by default None
        Lpq_ao : double 3d array, optional
            3-center density-fitting matrix in AO space, by default None
        """
        if mo_energy is None:
            self.mo_energy = np.asarray(_mo_energy_without_core(self, self._scf.mo_energy))
        if mo_coeff is None:
            self.mo_coeff = np.asarray(_mo_without_core(self, self._scf.mo_coeff))

        if Lpq_ao is None:
            self.initialize_df(auxbasis=self.auxbasis)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self, Lpq_ao=Lpq_ao)
        logger.timer(self, 'scGW', *cput0)
        return

    def energy_tot(self):
        """A wrapper to get GW total energy calculated in kernel.

        Returns
        -------
        e_tot : double
            GW total energy
        e_hf : double
            HF total energy
        e_c : double
            GW correlation energy
        """
        logger.info(self, 'HF energy@GW density  = %.8f', self.e_hf)
        logger.info(self, 'GW correlation energy = %.8f', self.e_c)
        logger.info(self, 'GW total energy       = %.8f', self.e_tot)

        return self.e_tot, self.e_hf, self.e_c

    def make_rdm1(self, ao_repr=False):
        """A wrapper to get GW density matrix calculated in kernel.

        Parameters
        ----------
        ao_repr : bool, optional
            return density matrix in AO, by default False

        Returns
        -------
        rdm1 : double 2d array
            one-particle density matrix
        """
        rdm1_ao = self.dm_ao
        ovlp = self._scf.get_ovlp()
        rdm1_no = self.mo_coeff.T @ ovlp @ rdm1_ao @ ovlp @ self.mo_coeff
        logger.info(self, 'GW particle number = %s', np.trace(rdm1_no))

        if ao_repr is True:
            return rdm1_ao
        else:
            return rdm1_no


if __name__ == '__main__':
    from pyscf import gto, dft, scf

    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel()

    gw = SCGW(mf)
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.44630857) < 1e-4
    # assert abs(gw.mo_energy[5] - 0.15313415) < 1e-4

    # Galitskii-Migdal total energy
    e_tot, e_hf, e_c = gw.energy_tot()
    assert abs(e_c - -0.41049904) < 1.0e-7

    print('passed tests!')
