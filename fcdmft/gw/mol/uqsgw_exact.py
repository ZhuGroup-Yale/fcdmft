"""
Spin-unrestricted quasiparticle self-consistent GW method based on the fully analytic scheme.

References:
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
    J. Chem. Theory Comput. 9, 1, 232-246 (2013)
"""

from functools import reduce
import numpy as np
import scipy

from pyscf import dft, scf
from pyscf.lib import einsum, logger, temporary_env

from fcdmft.ac.grids import _get_scaled_legendre_roots
from fcdmft.gw.mol.ugw_ac import UGWAC, get_g0
from fcdmft.gw.mol.ugw_exact import diagonalize_phrpa, get_transition_density, get_sigma


def kernel(gw):
    # local variables for convenience
    mf = gw._scf
    nmo = gw.nmo[0]
    nocc = gw.nocc
    mo_energy = gw.mo_energy
    mo_coeff = gw.mo_coeff

    # get hcore and ovlp
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()

    # keep this condition for embedding calculations
    if (not isinstance(gw._scf, dft.uks.UKS)) and isinstance(gw._scf, scf.uhf.UHF):
        uhf = gw._scf
    else:
        uhf = scf.UHF(gw.mol.copy(deep=True))
        uhf.verbose = uhf.mol.verbose = 0
        uhf.mo_energy = np.array(mo_energy, copy=True)
        uhf.mo_coeff = np.array(mo_coeff, copy=True)
        uhf.mo_occ = uhf.get_occ()
        if mf._eri is not None:
            uhf._eri = mf._eri
    dm_iter = uhf.make_rdm1()

    gw_diis = scf.diis.DIIS(gw, gw.diis_file)
    gw_diis.space = gw.diis_space

    conv = False
    cycle = 0
    while conv is False and cycle < max(1, gw.max_cycle):
        # update Lpq
        if gw.Lpq_ao is None:
            with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
                gw.Lpq = gw.ao2mo(mo_coeff)
        else:
            gw.Lpq = einsum('sLmn,smp,snq->sLpq', gw.Lpq_ao, mo_coeff, mo_coeff)

        # diagonalize the RPA matrix
        gw.exci, _ = exci, xpy = diagonalize_phrpa(nocc=nocc, mo_energy=mo_energy, Lpq=gw.Lpq, RPAE=gw.RPAE)

        # calculate the transition density
        gw.rho = rho = get_transition_density(nocc=nocc, xpy=xpy, Lpq=gw.Lpq)

        # calculate the self-energy
        sigma = get_sigma(
            nocc=nocc, mo_energy=mo_energy, mo_energy_prev=mo_energy, exci=exci, rho=rho, eta=gw.eta, fullsigma=True,
            mode=gw.mode
        )

        # obtain static correlation energy in AO basis
        vsig = np.asarray([reduce(np.matmul, (ovlp, mo_coeff[s], sigma[s], mo_coeff[s].T, ovlp)) for s in range(2)])

        # update veff
        if gw.vhf_df is False:
            veff = uhf.get_veff(dm=dm_iter)
        else:
            veff = np.zeros_like(dm)
            for s1 in range(2):
                for s2 in range(2):
                    veff[s1] = einsum('tLii,Lpq->pq', gw.Lpq[s2, :, : nocc[s2], : nocc[s2]], gw.Lpq[s1])
                    if s1 == s2:
                        veff[s1] -= einsum('sLpi,sLiq->spq', gw.Lpq[s2, :, :, : nocc[s2]], gw.Lpq[s1, :, : nocc[1], :])
            for s in range(2):
                veff[s] = reduce(np.matmul, (ovlp, mo_coeff[s], veff[s], mo_coeff[s].T, ovlp))

        # complete Hamiltonian through DIIS
        ham = hcore[None, :, :] + veff + vsig
        ham = gw_diis.update(ovlp, dm_iter, ham)

        # diagonalize
        for s in range(2):
            gw.mo_energy[s], gw.mo_coeff[s] = mo_energy[s], mo_coeff[s] = scipy.linalg.eigh(ham[s], ovlp)

        # check density matrix convergence
        dm_old = dm_iter.copy()
        # update QSGW density matrix
        dm_iter = uhf.make_rdm1(mo_coeff=mo_coeff)
        norm_dm = np.linalg.norm(dm_iter[s] - dm_old[s]) / nmo / 2.0
        if norm_dm < gw.conv_tol:
            conv = True

        logger.info(gw, 'UQSGW cycle= %d  |ddm|= %4.3g', cycle + 1, norm_dm)
        with np.printoptions(threshold=nmo):
            logger.debug(gw, '  GW mo_energy spin-up   =\n%s', mo_energy[0])
            logger.debug(gw, '  GW mo_energy spin-down =\n%s', mo_energy[1])
        cycle += 1

    logger.debug(gw, 'UQSGWExact %s in %-d cycles.', 'converged' if conv else 'not converged', cycle + 1)

    with np.printoptions(threshold=nmo):
        logger.debug(gw, '  GW mo_energy spin-up   =\n%s', mo_energy[0])
        logger.debug(gw, '  GW mo_energy spin-down =\n%s', mo_energy[1])
        cycle += 1

    return


class UQSGWExact(UGWAC):
    def __init__(self, mf, auxbasis=None):
        UGWAC.__init__(self, mf, frozen=None, auxbasis=auxbasis)
        # options
        self.max_cycle = 20  # max cycle
        self.conv_tol = 1.0e-6  # convergence tolerance
        self.diis_space = 10  # DIIS space
        self.diis_file = None  # DIIS file
        self.mode = 'b'  # mode for off-diagonal elements of static correlation self-energy
        self.RPAE = False  # exchange in RPA response

        # matrices
        self.Lpq_ao = None  # three-center density-fitting matrix in AO, used for impurity solver

        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        log.info('GW nmo = %s', self.nmo[0])
        log.info('GW nocc = %s, nvir = %s', self.nocc, (self.nmo[s] - self.nocc[s] for s in range(2)))
        log.info('density-fitting for exchange = %s', self.vhf_df)
        log.info('broadening parameter = %.3e', self.eta)
        log.info('static sigma mode = %s', self.mode)
        # qsGW parameters
        log.info('max cycle = %d', self.max_cycle)
        log.info('convergence tolerance = %.3e', self.conv_tol)
        log.info('DIIS space = %d', self.diis_space)
        log.info('DIIS file = %s', self.diis_file)
        log.info('')
        return

    def kernel(self):
        assert self.frozen is None or self.frozen == 0
        assert self.orbs is None
        assert self.outcore is False
        assert hasattr(self._scf, 'sigma') is False
        assert self.nmo[0] == self.nmo[1]

        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return

    def make_gf(self, omega, eta=0.0):
        """Get qsGW Green's function.

        Parameters
        ----------
        omega : complex 1d array
             frequency grids
        eta : double, optional
            broadening parameter, by default 0

        Returns
        -------
        gf: complex 4d array
            qsGW Green's function in mean-field MO space.
        gf0 : complex 4d array
            non-interacting Green's function.
        sigma : complex 4d array
            self-energy (including 1e correction) in mean-field MO space.
        """
        mf = self._scf
        nmo = self.nmo[0]
        nocc = self.nocc
        exci = self.exci
        rho = self.rho

        mo_energy = np.asarray(self.mo_energy)

        # get self-energy
        # NOTE: this is for G0W0 Green's function, so mo_energy is from the SCF calculation
        sigma = np.zeros(shape=[2, nmo, nmo, len(omega)], dtype=np.complex128)
        for s in range(2):
            e_occ = omega[:, None, None] - mo_energy[s, None, :nocc[s], None] + (exci[None, None, :] - 3.0 * 1j * eta)
            e_vir = omega[:, None, None] - mo_energy[s, None, nocc[s]:, None] - (exci[None, None, :] - 3.0 * 1j * eta)
            energy = np.concatenate([e_occ, e_vir], axis=1)
            energy = 1.0 / energy
            sigma[s] = einsum('mpr,mqr,wrm->pqw', rho[s], rho[s], energy)

        # get self-energy difference in the one-electron Hamiltonian in mean-field MO space
        if (not isinstance(self._scf, dft.uks.UKS)) and isinstance(self._scf, scf.uhf.UHF):
            uhf = self._scf
        else:
            uhf = scf.UHF(self.mol)
        with temporary_env(mf, verbose=0):
            ovlp = mf.get_ovlp()
            hcore = mf.get_hcore()
            dm_mf = mf.make_rdm1()
            dm_qp = mf.make_rdm1(mo_coeff=self.mo_coeff)
            veff_mf = mf.get_veff(dm=dm_mf)

        veff_qp = uhf.get_veff(dm=dm_qp)
        C_qp_mf = np.array([reduce(np.matmul, (self.mo_coeff[s].T, ovlp, mf.mo_coeff[s])) for s in range(2)])
        ham_mf = np.array([reduce(np.matmul, (mf.mo_coeff[s].T, hcore + veff_mf[s], mf.mo_coeff[s])) for s in range(2)])
        ham_qp = np.array([reduce(np.matmul, (mf.mo_coeff[s].T, hcore + veff_qp[s], mf.mo_coeff[s])) for s in range(2)])

        gf0 = get_g0(omega=omega, mo_energy=mf.mo_energy, eta=eta)

        gf = np.zeros_like(gf0)
        for s in range(2):
            for iw in range(len(omega)):
                sigma[s, ..., iw] = reduce(np.matmul, (C_qp_mf[s].T, sigma[s, ..., iw], C_qp_mf[s]))
                sigma[s, ..., iw] += ham_qp[s] - ham_mf[s]
                gf[s, ..., iw] = np.linalg.inv(np.linalg.inv(gf0[s, ..., iw]) - sigma[s, ..., iw])

        return gf, gf0, sigma

    def make_rdm1(self, nw=60, ao_repr=False):
        r"""Get GW one-particle density matrix.
        rdm1 = G(it=0) = \int dw G(iw)

        Parameters
        ----------
        nw : int, optional
            number for imaginary frequency grids for integration, by default 60
        ao_repr : bool, optional
            return density matrix in AO, by default False

        Returns
        -------
        rdm1 : double ndarray
            GW one-particle density matrix
        """
        # get qsGW G
        homo = max(self.mo_energy[0][self.nocc[0] - 1], self.mo_energy[1][self.nocc[1] - 1])
        lumo = min(self.mo_energy[0][self.nocc[0]], self.mo_energy[1][self.nocc[1]])
        ef = (homo + lumo) / 2.0
        freqs, wts = _get_scaled_legendre_roots(nw)
        gf = self.make_gf(omega=1j * freqs + ef, eta=0)[0]

        # qsGW density matrix in mean-field MO basis
        rdm1 = 1.0 / np.pi * einsum('sijw,w->sij', gf, wts).real + np.eye(self.nmo[0])[None, :, :] * 0.5

        # symmetrize density matrix
        for s in range(2):
            rdm1[s] = 0.5 * (rdm1[s] + rdm1[s].T)
        nelec = np.trace(rdm1, axis1=1, axis2=2)
        logger.info(self, 'GW particle number up = %s, dn = %s, total = %s', nelec[0], nelec[1], nelec[0] + nelec[1])

        if ao_repr is True:
            for s in range(2):
                rdm1[s] = reduce(np.matmul, (self._scf.mo_coeff[s], rdm1[s], self._scf.mo_coeff[s].T))

        return rdm1

    def energy_tot(self, nw=60):
        """Calculate GW total energy using Galitskii-Migdal formula.
        V. M. Galitskii and A. B. Migdal, Zh. E ksp. Teor. Fiz. 34, 139~1958! @Sov. Phys. JETP 139, 96 ~1958!#
        Working equation: equation A5 in Phys. Rev. B 88, 075105

        Parameters
        ----------
        nw : int, optional
            number for imaginary frequency grids for integration, by default 60

        Returns
        -------
        e_tot : double
            GW total energy
        e_hf : double
            HF total energy
        e_c : double
            GW correlation energy
        """
        homo = max(self.mo_energy[0][self.nocc[0] - 1], self.mo_energy[1][self.nocc[1] - 1])
        lumo = min(self.mo_energy[0][self.nocc[0]], self.mo_energy[1][self.nocc[1]])
        ef = (homo + lumo) / 2.0
        freqs, wts = _get_scaled_legendre_roots(nw)
        gf = self.make_gf(omega=1j * freqs + ef, eta=0)[0]

        # gf0 = 1 / (iw + ef - Fock), not from make_gf
        dm = self.make_rdm1(nw=nw, ao_repr=True)
        if (not isinstance(self._scf, dft.uks.UKS)) and isinstance(self._scf, scf.uhf.UHF):
            uhf = self._scf
        else:
            uhf = scf.UHF(self.mol)
        fock = uhf.get_fock(dm=dm)
        for s in range(2):
            fock[s] = reduce(np.matmul, (self._scf.mo_coeff[s].T, fock[s], self._scf.mo_coeff[s]))
        gf0 = np.zeros_like(gf)
        for s in range(2):
            for iw in range(nw):
                gf0[s, :, :, iw] = np.linalg.inv((ef + 1j * freqs[iw]) * np.eye(self.nmo[0]) - fock[s])

        # sigma = G0^-1 - G^-1, not from make_gf
        sigma = np.zeros_like(gf0)
        for s in range(2):
            for iw in range(nw):
                sigma[s, :, :, iw] = np.linalg.inv(gf0[s, :, :, iw]) - np.linalg.inv(gf[s, :, :, iw])

        # evaluate HF energy with qsGW density matrix
        with temporary_env(uhf, verbose=0):
            e_hf = uhf.energy_tot(dm=dm)
        e_c = einsum('sijw,sjiw,w->', gf, sigma, wts).real / np.pi / 2.0
        e_tot = e_c + e_hf

        logger.info(self, 'HF energy@GW density  = %.8f', e_hf)
        logger.info(self, 'GW correlation energy = %.8f', e_c)
        logger.info(self, 'GW total energy       = %.8f', e_tot)

        return e_tot, e_hf, e_c


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

    gw = UQSGWExact(mf)
    gw.kernel()
    assert abs(gw.mo_energy[0][4] - -1.06508525) < 1e-4
    assert abs(gw.mo_energy[0][5] - -0.15976937) < 1e-4
    assert abs(gw.mo_energy[1][3] - -1.02365107) < 1e-4
    assert abs(gw.mo_energy[1][4] - -0.42253155) < 1e-4

    # Galitskii-Migdal total energy
    # numerically not stable because it's highly sensitive to final quasiparticle energy and orbital
    e_tot, e_hf, e_c = gw.energy_tot()

    # density matrix
    dm = gw.make_rdm1()

    # generate Green's function on real axis and print density of states
    omega = np.linspace(-0.5, 0.5, 101)
    gf, gf0, _ = gw.make_gf(omega=omega, eta=0.01)
    print('\nalpha DOS: KS, GW')
    for iw in range(len(omega)):
        print(omega[iw], -np.trace(gf0[0, :, :, iw].imag) / np.pi, -np.trace(gf[0, :, :, iw].imag) / np.pi)

    print('passed tests!')
