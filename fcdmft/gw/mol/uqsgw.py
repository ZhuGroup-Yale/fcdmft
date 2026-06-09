"""
Spin-unrestricted quasiparticle self-consistent GW method based on the analytic continuation scheme.

References:
    J. Lei and T. Zhu, J. Chem. Phys. 157, 214114 (2022)
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    Phys. Rev. B 76, 165106 (2007)
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
"""

import h5py
from functools import reduce
import numpy as np
import scipy

from pyscf.lib import einsum, logger, temporary_env
from pyscf import dft, scf

from fcdmft.ac.pade import PadeAC
from fcdmft.ac.two_pole import TwoPoleAC
from fcdmft.ac.grids import _get_scaled_legendre_roots
from fcdmft.gw.mol.ugw_ac import UGWAC, _mo_energy_without_core, _mo_without_core, set_frozen_orbs, get_sigma, \
    get_sigma_outcore, get_g0
from fcdmft.gw.mol.qsgw import qs_sigmaR_mode_a, qs_sigmaR_mode_b


def kernel(gw):
    # local variables for convenience
    mf = gw._scf
    nmo = gw.nmo[0]
    nocc = gw.nocc
    eta = gw.eta

    if gw.load_chkfile:
        with h5py.File(gw.chkfile, 'r') as fh5:
            gw.mo_energy = np.array(fh5['gw/mo_energy'])
            gw.mo_coeff = np.array(fh5['gw/mo_coeff'])

    # set frozen orbitals
    set_frozen_orbs(gw)
    orbs_frz = gw.orbs_frz

    # get non-frozen quantities
    mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
    mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)
    mo_occ_frz = _mo_energy_without_core(gw, gw.mo_occ)

    # grids for integration on imaginary axis
    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
    eval_freqs_with_zero = gw.setup_evaluation_grid(fallback_freqs=quad_freqs, fallback_wts=quad_wts)

    # get hcore and ovlp
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()

    # keep this condition for embedding calculations
    if (not isinstance(gw._scf, dft.uks.UKS)) and isinstance(gw._scf, scf.uhf.UHF):
        uhf = gw._scf
    else:
        uhf = scf.UHF(gw.mol.copy(deep=True))
        if hasattr(gw._scf, 'sigma'):
            uhf = scf.addons.smearing_(uhf, sigma=gw._scf.sigma, method=gw._scf.smearing_method)
        uhf.verbose = uhf.mol.verbose = 0
        uhf.mo_occ = uhf.get_occ(mo_energy=gw.mo_energy, mo_coeff=gw.mo_coeff)
        if mf._eri is not None:
            uhf._eri = mf._eri
    dm_iter = uhf.make_rdm1(mo_energy=gw.mo_energy, mo_coeff=gw.mo_coeff)

    gw_diis = scf.diis.DIIS(gw, gw.diis_file)
    gw_diis.space = gw.diis_space

    # set up Fermi level
    gw.ef = ef = gw.get_ef(mo_energy=gw.mo_energy)

    conv = False
    cycle = 0
    while conv is False and cycle < max(1, gw.max_cycle):
        # update Lpq
        if gw.Lpq_ao is None:
            with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
                gw.Lpq = gw.ao2mo(gw.mo_coeff)
        else:
            gw.Lpq = einsum('sLmn,smp,snq->sLpq', gw.Lpq_ao, gw.mo_coeff, gw.mo_coeff)

        # compute full self-energy on imaginary axis
        if gw.outcore:
            sigmaI, omega = get_sigma_outcore(
                gw, orbs_frz, quad_freqs, quad_wts, ef, mo_energy=mo_energy_frz, mo_coeff=mo_coeff_frz, mo_occ=mo_occ_frz,
                iw_cutoff=gw.ac_iw_cutoff, eval_freqs=eval_freqs_with_zero, fullsigma=True
            )
        else:
            sigmaI, omega = get_sigma(
                gw, orbs_frz, gw.Lpq, quad_freqs, quad_wts, ef, mo_energy=mo_energy_frz, mo_occ=mo_occ_frz,
                iw_cutoff=gw.ac_iw_cutoff, eval_freqs=eval_freqs_with_zero, fullsigma=True
            )

        # analytic continuation
        if gw.ac == 'twopole':
            acobj = TwoPoleAC(list(range(nmo)), nocc)
        elif gw.ac == 'pade':
            acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
        else:
            raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))
        acobj.ac_fit(sigmaI, omega, axis=-1)

        sigma = np.zeros(shape=[2, nmo, nmo], dtype=np.complex128)
        for s in range(2):
            for ip, p in enumerate(orbs_frz):
                for iq, q in enumerate(orbs_frz):
                    if gw.mode == 'b':
                        if p == q:
                            sigma[s, p, q] += 0.5 * acobj[s, ip, iq].ac_eval(mo_energy_frz[s][p] + 1j * eta)
                            sigma[s, p, q] += 0.5 * acobj[s, ip, iq].ac_eval(mo_energy_frz[s][p] + 1j * eta).conj()
                        else:
                            sigma[s, p, q] += 0.5 * acobj[s, ip, iq].ac_eval(1j * eta + ef)
                            sigma[s, p, q] += 0.5 * acobj[s, iq, ip].ac_eval(1j * eta + ef).conj()
                    elif gw.mode == 'a':
                        sigma[s, p, q] += 0.25 * acobj[s, ip, iq].ac_eval(mo_energy_frz[s][p] + 1j * eta)
                        sigma[s, p, q] += 0.25 * acobj[s, iq, ip].ac_eval(mo_energy_frz[s][p] + 1j * eta).conj()
                        sigma[s, p, q] += 0.25 * acobj[s, ip, iq].ac_eval(mo_energy_frz[s][q] + 1j * eta)
                        sigma[s, p, q] += 0.25 * acobj[s, iq, ip].ac_eval(mo_energy_frz[s][q] + 1j * eta).conj()
                    elif gw.mode == 'c':
                        sigma[s, p, q] += 0.5 * acobj[s, ip, iq].ac_eval(1j * eta + ef)
                        sigma[s, p, q] += 0.5 * acobj[s, iq, ip].ac_eval(1j * eta + ef).conj()
                    else:
                        raise ValueError('Unknown QSGW mode %s' % gw.mode)

        # get static correlation self-energy in AO basis
        vsig = np.zeros_like(dm_iter)
        for s in range(2):
            CS = np.matmul(mo_coeff_frz[s].T, ovlp)
            vsig[s] = reduce(np.matmul, (CS.T, sigma[s].real, CS))
        gw.vsig = vsig

        # update veff
        if gw.vhf_df is False:
            veff = uhf.get_veff(dm=dm_iter)
        else:
            veff = np.zeros_like(dm_iter)
            for s1 in range(2):
                for s2 in range(2):
                    veff[s1] = einsum('tLii,Lpq->pq', gw.Lpq[s2, :, : nocc[s2], : nocc[s2]], gw.Lpq[s1])
                    if s1 == s2:
                        veff[s1] -= einsum('sLpi,sLiq->spq', gw.Lpq[s2, :, :, : nocc[s2]], gw.Lpq[s1, :, : nocc[s1], :])
            for s in range(2):
                veff[s] = reduce(np.matmul, (ovlp, mo_coeff_frz[s], veff[s], mo_coeff_frz[s].T, ovlp))

        # complete Hamiltonian through DIIS
        ham = hcore[None, :, :] + veff + vsig
        ham = gw_diis.update(ovlp, dm_iter, ham)

        # update mo_energy, mo_coeff, mo_occ and ef
        for s in range(2):
            gw.mo_energy[s], gw.mo_coeff[s] = scipy.linalg.eigh(ham[s], ovlp)
        gw.mo_occ = uhf.get_occ(mo_energy=gw.mo_energy, mo_coeff=gw.mo_coeff)
        gw.ef = ef = gw.get_ef()
        gw.acobj = acobj

        # update non-frozen quantities
        mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
        mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)
        mo_occ_frz = _mo_energy_without_core(gw, gw.mo_occ)

        # check density matrix convergence
        dm_old = dm_iter.copy()
        # update QSGW density matrix
        dm_iter = uhf.make_rdm1(mo_coeff=gw.mo_coeff, mo_occ=gw.mo_occ)
        norm_dm = np.linalg.norm(dm_iter - dm_old) / nmo / 2.0
        if norm_dm < gw.conv_tol:
            conv = True

        logger.info(gw, 'UQSGW cycle= %d  |ddm|= %4.3g', cycle + 1, norm_dm)
        cycle += 1
        with np.printoptions(threshold=len(gw.mo_energy[0])):
            logger.debug(gw, '  GW mo_energy spin-up   =\n%s', gw.mo_energy[0])
            logger.debug(gw, '  GW mo_energy spin-down =\n%s', gw.mo_energy[1])

        if gw.chkfile:
            gw.dump_chk()

    logger.debug(gw, 'UQSGW %s in %-d cycles.', 'converged' if conv else 'not converged', cycle + 1)

    with np.printoptions(threshold=len(gw.mo_energy[0])):
        logger.debug(gw, '  GW mo_energy spin-up   =\n%s', gw.mo_energy[0])
        logger.debug(gw, '  GW mo_energy spin-down =\n%s', gw.mo_energy[1])

    return


class UQSGW(UGWAC):
    def __init__(self, mf, frozen=None, auxbasis=None):
        UGWAC.__init__(self, mf, frozen=frozen, auxbasis=auxbasis)
        # options
        self.mode = 'b'  # mode to evaluate off-diagonal self-energy
        self.max_cycle = 30  # max qsGW cycle
        self.conv_tol = 1.0e-6  # convergence tolerance
        self.diis_space = 10  # DIIS space
        self.diis_file = None  # DIIS file name
        self.chkfile = None  # name of check file
        self.load_chkfile = False  # load check file

        # matrices
        self.Lpq_ao = None  # three-center density-fitting matrix in AO, used for impurity solver
        self.vsig = None  # static correlation self-energy in AO

        return

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        log.info('GW nmo = %s', self.nmo[0])
        log.info('GW nocc = %s, nvir = %s', self.nocc, (self.nmo[s] - self.nocc[s] for s in range(2)))
        log.info('frozen orbitals = %s', self.frozen)
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
        if self.ac == 'pade':
            log.info('Pade points = %d', self.ac_pade_npts)
            log.info('Pade step ratio = %.3f', self.ac_pade_step_ratio)
        # qsGW parameters
        log.info('off-diagonal mode = %s' % self.mode)
        log.info('max cycle = %d', self.max_cycle)
        log.info('convergence tolerance = %.3e', self.conv_tol)
        log.info('DIIS space = %d', self.diis_space)
        log.info('DIIS file = %s', self.diis_file)
        log.info('')
        return

    def dump_chk(self):
        if self.chkfile:
            with h5py.File(self.chkfile, 'w') as fh5:
                fh5['gw/mo_energy'] = self.mo_energy
                fh5['gw/mo_coeff'] = self.mo_coeff
                fh5['gw/mo_occ'] = self.mo_occ
        return

    def kernel(self):
        """Run a spin-unrestricted quasiparticle self-consistent GW calculation."""
        if hasattr(self._scf, 'sigma'):
            assert self.frozen == 0 or self.frozen is None
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0

        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        if isinstance(self.frozen, list) and (not isinstance(self.frozen[0], list)):
            # make sure self.frozen is a list of lists if not frozen core
            self.frozen = [self.frozen, self.frozen]
        else:
            assert self.frozen is None or isinstance(self.frozen, (int, np.int64))

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return

    def make_gf(self, omega, eta=0.0, mode='dyson'):
        """Get qsGW Green's function by AC fitting.

        Parameters
        ----------
        omega : complex 1d array
            frequency on which to evaluate the Green's function
        eta : double, optional
            broadening parameter, by default 0.0
        mode : str, one of dyson or linear
            mode for Dyson equation, 'linear' or 'dyson', by default 'dyson'

        Returns
        -------
        gf: complex 4d array
            qsGW Green's function in mean-field MO space.
        gf0 : complex 4d array
            non-interacting Green's function.
        sigma : complex 4d array
            self-energy (including 1e correction) in mean-field MO space.
        """
        assert mode in ['dyson', 'linear']
        assert self.frozen is None or self.frozen == 0

        mf = self._scf
        sigma = self.acobj.ac_eval(omega + 1.0j * eta)

        # get self-energy difference in the one-electron Hamiltonian in mean-field MO space
        if (not isinstance(self._scf, dft.uks.UKS)) and isinstance(self._scf, scf.hf.UHF):
            uhf = self._scf
        else:
            uhf = scf.UHF(self.mol)
            if hasattr(self._scf, 'sigma'):
                uhf = scf.addons.smearing_(uhf, sigma=self._scf.sigma, method=self._scf.smearing_method)
        with temporary_env(mf, verbose=0):
            ovlp = mf.get_ovlp()
            hcore = mf.get_hcore()
            dm_mf = mf.make_rdm1()
            dm_qp = mf.make_rdm1(mo_coeff=self.mo_coeff)
            veff_mf = mf.get_veff(dm=dm_mf)

        veff_qp = uhf.get_veff(dm=dm_qp)

        gf0 = get_g0(omega=omega, mo_energy=mf.mo_energy, eta=eta)
        gf = np.zeros_like(gf0)
        for s in range(2):
            C_qp_mf = reduce(np.matmul, (self.mo_coeff[s].T, ovlp, mf.mo_coeff[s]))
            ham_mf = reduce(np.matmul, (mf.mo_coeff[s].T, hcore + veff_mf[s], mf.mo_coeff[s]))
            ham_qp = reduce(np.matmul, (mf.mo_coeff[s].T, hcore + veff_qp[s], mf.mo_coeff[s]))

            for iw in range(len(omega)):
                sigma[s, :, :, iw] = reduce(np.matmul, (C_qp_mf.T, sigma[s, :, :, iw], C_qp_mf))
            sigma[s] += ham_qp[:, :, np.newaxis] - ham_mf[:, :, np.newaxis]

        for s in range(2):
            for iw in range(len(omega)):
                if mode == 'dyson':
                    gf[s, :, :, iw] = np.linalg.inv(np.linalg.inv(gf0[s, :, :, iw]) - sigma[s, :, :, iw])
                elif mode == 'linear':
                    gf[s, :, :, iw] = gf0[s, :, :, iw] + reduce(
                        np.matmul, (gf0[s, :, :, iw], sigma[s, :, :, iw], gf0[s, :, :, iw])
                    )

        return gf, gf0, sigma

    def make_rdm1(self, ao_repr=False, mode='dyson'):
        """make qsGW 1-particle density matrix

        Parameters
        ----------
        ao_repr : bool, optional
            return dm in AO space instead of MO space, by default False
        mode : str, optional
            mode for Dyson equation, 'linear' or 'dyson', by default 'dyson'
        Returns
        -------
        rdm1 : double 3d array
            one-particle density matrix
        """
        assert self.frozen is None or self.frozen == 0
        # make_gf returns qsGW Green's function in mean-field MO basis
        gf = self.make_gf(omega=1j * self.freqs + self.ef, eta=0, mode=mode)[0]

        # qsGW density matrix in qsGW MO basis
        rdm1 = 1.0 / np.pi * einsum('sijw,w->sij', gf, self.wts).real + np.eye(self.nmo[0]) * 0.5
        # symmetrize density matrix
        rdm1 = 0.5 * (rdm1 + rdm1.transpose((0, 2, 1)))
        nelec = np.trace(rdm1, axis1=1, axis2=2)
        logger.info(self, 'GW particle number up = %s, dn = %s, total = %s', nelec[0], nelec[1], nelec[0] + nelec[1])

        for s in range(2):
            if ao_repr is True:
                rdm1[s] = reduce(np.matmul, (self._scf.mo_coeff[s], rdm1[s], self._scf.mo_coeff[s].T))

        return rdm1


if __name__ == '__main__':
    from pyscf import gto, dft, scf

    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.charge = 1
    mol.spin = 1
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    gw = UQSGW(mf)
    gw.kernel()
    assert abs(gw.mo_energy[0][4] - -1.06510471) < 1e-4
    assert abs(gw.mo_energy[0][5] - -0.15976838) < 1e-4
    assert abs(gw.mo_energy[1][3] - -1.02368349) < 1e-4
    assert abs(gw.mo_energy[1][4] - -0.42241723) < 1e-4

    # An example for using chkfile
    gw = UQSGW(mf)
    # set chkfile for saving qsgw results
    gw.chkfile = 'h2o_qsgw.chk'
    gw.max_cycle = 5
    gw.kernel()

    gw = UQSGW(mf)
    # load qsgw chkfile
    gw.load_chkfile = True
    gw.chkfile = 'h2o_qsgw.chk'
    gw.kernel()
    assert abs(gw.mo_energy[0][4] - -1.06510471) < 1e-4
    assert abs(gw.mo_energy[0][5] - -0.15976838) < 1e-4
    assert abs(gw.mo_energy[1][3] - -1.02368349) < 1e-4
    assert abs(gw.mo_energy[1][4] - -0.42241723) < 1e-4

    # generate Green's function on real axis and print density of states
    gw = UQSGW(mf)
    gw.kernel()
    omega = np.linspace(-0.5, 0.5, 101)
    gf, gf0, _ = gw.make_gf(omega=omega, eta=0.01, mode='dyson')
    print('\nalpha DOS: KS, GW')
    for iw in range(len(omega)):
        print(omega[iw], -np.trace(gf0[0, :, :, iw].imag) / np.pi, -np.trace(gf[0, :, :, iw].imag) / np.pi)

    print('passed tests!')
