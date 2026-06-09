"""
Spin-restricted eigenvalue self-consistent GW method based on the analytic continuation scheme.

References:
    Phys. Rev. B 76, 165106 (2007)
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
"""

from functools import reduce
import numpy as np
import scipy
import time

from pyscf import dft, scf
from pyscf.lib import diis, einsum, logger, temporary_env

from fcdmft.ac.pade import PadeAC
from fcdmft.ac.two_pole import TwoPoleAC
from fcdmft.ac.grids import _get_scaled_legendre_roots
from fcdmft.gw.mol.gw_ac import GWAC, _mo_energy_without_core, _mo_without_core, set_frozen_orbs, get_sigma, \
    get_sigma_outcore, get_g0


def kernel(gw):
    # local variables for convenience
    mf = gw._scf
    nmo = gw.nmo
    nocc = gw.nocc
    mo_energy = gw.mo_energy

    # set frozen orbitals
    set_frozen_orbs(gw)
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz

    # get non-frozen quantities
    mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
    mf_mo_energy_frz = _mo_energy_without_core(gw, gw._scf.mo_energy)
    mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)

    if gw.Lpq is None and gw.outcore is False:
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            Lpq = gw.Lpq = gw.ao2mo(mo_coeff_frz)

    hcore = reduce(np.matmul, (mo_coeff_frz.T, mf.get_hcore(), mo_coeff_frz))
    if gw.vhf_df is False:
        dm = mf.make_rdm1()
        if (not isinstance(mf, dft.rks.RKS)) and isinstance(mf, scf.hf.RHF):
            rhf = mf
        else:
            rhf = scf.RHF(gw.mol)
            if hasattr(gw._scf, 'sigma'):
                rhf = scf.addons.smearing_(rhf, sigma=gw._scf.sigma, method=gw._scf.smearing_method)
        vjk = reduce(np.matmul, (mo_coeff_frz.T, rhf.get_veff(dm=dm), mo_coeff_frz))
        vk = vjk - reduce(np.matmul, (mo_coeff_frz.T, rhf.get_j(dm=dm), mo_coeff_frz))
    else:
        # TODO: smearing
        vj = 2.0 * einsum('Lii,Lpq->pq', Lpq[:, :nocc, :nocc], Lpq)
        vk = -einsum('Lpi,Liq->pq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
        vjk = vj + vk
    gw.vk = vk
    ham_hf = hcore + vjk  # HF Hamiltonian in MO space

    gw_diis = diis.DIIS(gw, gw.diis_file)
    gw_diis.space = gw.diis_space

    # set up Fermi level
    gw.ef = ef = gw.get_ef(mo_energy=mf.mo_energy)

    # grids for integration on imaginary axis
    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
    eval_freqs_with_zero = gw.setup_evaluation_grid(fallback_freqs=quad_freqs, fallback_wts=quad_wts)

    conv = False
    cycle = 0
    while conv is False and cycle < max(1, gw.max_cycle):
        mo_energy_prev = mo_energy.copy()
        mo_energy_frz_prev = mo_energy_frz.copy()

        # compute self-energy on imaginary
        mo_energy_w = mf_mo_energy_frz if gw.W0 is True else mo_energy_frz
        if gw.outcore:
            sigmaI, omega = get_sigma_outcore(
                gw, orbs_frz, quad_freqs, quad_wts, ef, mo_energy_frz, mo_coeff_frz, iw_cutoff=gw.ac_iw_cutoff,
                mo_energy_w=mo_energy_w, eval_freqs=eval_freqs_with_zero, fullsigma=False
            )
        else:
            sigmaI, omega = get_sigma(
                gw, orbs_frz, Lpq, quad_freqs, quad_wts, ef, mo_energy_frz, iw_cutoff=gw.ac_iw_cutoff, mo_energy_w=mo_energy_w,
                eval_freqs=eval_freqs_with_zero, fullsigma=False
            )

        # analytic continuation
        if gw.ac == 'twopole':
            acobj = TwoPoleAC(orbs_frz, nocc)
        elif gw.ac == 'pade':
            acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
        else:
            raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))
        acobj.ac_fit(sigmaI, omega, axis=-1)
        gw.acobj = acobj

        # follow the section 5.1.1 Method evGW in 10.1021/acs.jctc.5b01238
        # for each pole the QP-equation is solved self-consistently
        # only iterative quasiparticle equation is implemented here
        mo_energy = mo_energy_prev.copy()
        for ip, p in enumerate(orbs_frz):

            def quasiparticle(omega):
                sigmaR = acobj[ip].ac_eval(omega).real
                return omega - (ham_hf[p, p] + sigmaR)

            try:
                mo_energy[orbs[ip]] = scipy.optimize.newton(
                    quasiparticle, mo_energy_prev[orbs[ip]], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter
                )
            except RuntimeError:
                logger.warn(gw, 'QPE for orbital=%d not converged!', orbs[ip])

        # update quasiparticle energy through DIIS
        gw.mo_energy = gw_diis.update(mo_energy)

        # update non-frozen attribute and Fermi level
        mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
        gw.ef = ef = gw.get_ef(mo_energy=gw.mo_energy)

        diff = abs(np.sum(1.0 / mo_energy_frz - 1.0 / mo_energy_frz_prev)) / nmo / nmo
        if diff < gw.conv_tol:
            conv = True

        logger.info(gw, 'EVGW cycle= %d  |delta G|= %4.3g', cycle + 1, diff)
        cycle += 1

        with np.printoptions(threshold=len(gw._scf.mo_energy)):
            logger.debug(gw, '  GW mo_energy =\n%s', gw.mo_energy)

    logger.debug(gw, 'EVGW %s in %-d cycles.', 'converged' if conv else 'not converged', cycle + 1)

    with np.printoptions(threshold=len(gw._scf.mo_energy)):
        logger.debug(gw, '  GW mo_energy =\n%s', gw.mo_energy)

    return


class EVGW(GWAC):
    def __init__(self, mf, frozen=None, auxbasis=None):
        GWAC.__init__(self, mf, frozen=frozen, auxbasis=auxbasis)
        self.W0 = False  # evGW0
        self.max_cycle = 30  # max evGW cycle
        self.conv_tol = 1.0e-7  # convergence tolerance
        self.diis_space = 10  # DIIS space
        self.diis_file = None  # DIIS file name

        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
        log.info('density-fitting for exchange = %s', self.vhf_df)
        log.info('outcore for self-energy= %s', self.outcore)
        if self.outcore is True:
            log.info('outcore segment size = %d', self.segsize)
        log.info('broadening parameter = %.3e', self.eta)
        if self.nw2 is None:
            log.info('number of grids = %d', self.nw)
        else:
            log.info('number of grids for integration= %d', self.nw)
            log.info('number of grids to be integrated = %d', self.nw2)
        log.info('analytic continuation method = %s', self.ac)
        log.info('imaginary frequency cutoff = %.1f', self.ac_iw_cutoff)
        if self.ac == "pade":
            log.info('Pade points = %d', self.ac_pade_npts)
            log.info('Pade step ratio = %.3f', self.ac_pade_step_ratio)
        log.info('QPE max iter = %d', self.qpe_max_iter)
        log.info('QPE tolerance = %.1e', self.qpe_tol)
        # evGW parameters
        log.info('evGW0 = %s', self.W0)
        log.info('max cycle = %d', self.max_cycle)
        log.info('convergence tolerance = %.3e', self.conv_tol)
        log.info('DIIS space = %d', self.diis_space)
        log.info('DISS file = %s', self.diis_file)
        log.info('')
        return

    def kernel(self):
        """Run an eigenvalue-self-consistent GW calculation."""
        # smeared GW needs denser grids to be accurate
        if hasattr(self._scf, 'sigma'):
            assert self.frozen == 0 or self.frozen is None
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0

        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return

    def make_gf(self, omega, eta=0.0, mode='dyson'):
        """Get evGW Green's function by AC fitting.
        Self-energy is evaluated with diagonal approximation.

        Parameters
        ----------
        omega : complex 1d array
            frequency on which to evaluate the Green's function
        eta : double, optional
            broadening parameter. Defaults to 0.
        mode : str, optional
            mode for Dyson equation, 'linear' or 'dyson', by default 'dyson'

        Returns
        -------
        gf : complex 3d array
            GW Green's function
        gf0 : complex 3d array
            non-interacting Green's function
        sigma : complex 3d array
            self-energy
        """
        assert self.frozen is None or self.frozen == 0
        mf_mo_energy = np.array(self._scf.mo_energy, copy=True)
        nmo = len(mf_mo_energy)

        with temporary_env(self._scf, verbose=0):
            vxc = reduce(np.matmul, (self.mo_coeff.T, self._scf.get_veff() - self._scf.get_j(), self.mo_coeff))

        gf0 = get_g0(omega, mf_mo_energy, eta)
        sigma = np.zeros_like(gf0)
        sigma_diff = np.zeros_like(gf0)
        vk_minus_vxc_diag = np.diagonal(self.vk) - np.diagonal(vxc)
        for iw, w in enumerate(omega):
            np.fill_diagonal(sigma[:, :, iw], self.acobj.ac_eval(w + 1j * eta))
            np.fill_diagonal(sigma_diff[:, :, iw], np.diagonal(sigma[:, :, iw]) + vk_minus_vxc_diag)

        gf = np.zeros_like(gf0)
        if mode == 'linear':
            for iw in range(len(omega)):
                gf[:, :, iw] = gf0[:, :, iw] + reduce(np.matmul, (gf0[:, :, iw], sigma_diff[:, :, iw], gf0[:, :, iw]))
        elif mode == 'dyson':
            for iw in range(len(omega)):
                gf[:, :, iw] = np.linalg.inv(np.linalg.inv(gf0[:, :, iw]) - sigma_diff[:, :, iw])

        return gf, gf0, sigma


if __name__ == '__main__':
    from pyscf import gto, dft

    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    # evGW
    gw = EVGW(mf)
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.44302542) < 1e-4
    assert abs(gw.mo_energy[5] - 0.1681912) < 1e-4

    # evGW outcore
    gw = EVGW(mf)
    gw.outcore = True
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.44302542) < 1e-4
    assert abs(gw.mo_energy[5] - 0.1681912) < 1e-4

    # evGW frozen core
    gw = EVGW(mf)
    gw.frozen = 1
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.44322502) < 1e-4
    assert abs(gw.mo_energy[5] - 0.16812461) < 1e-4

    # evGW0
    gw = EVGW(mf)
    gw.W0 = True
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.43324833) < 1e-4
    assert abs(gw.mo_energy[5] - 0.16626621) < 1e-4

    # generate Green's function on real axis and print density of states
    gw = EVGW(mf)
    gw.kernel()
    omega = np.linspace(-0.5, 0.5, 101)
    gf, gf0, _ = gw.make_gf(omega=omega, eta=0.01, mode='dyson')
    print('\nDOS: KS, GW')
    for iw in range(len(omega)):
        print(omega[iw], -np.trace(gf0[:, :, iw].imag) / np.pi, -np.trace(gf[:, :, iw].imag) / np.pi)

    print('passed tests!')
