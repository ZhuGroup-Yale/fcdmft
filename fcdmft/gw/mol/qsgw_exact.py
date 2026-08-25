"""
Spin-restricted quasiparticle self-consistent GW method based on the fully analytic scheme.

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
from fcdmft.gw.mol.gw_ac import GWAC, get_g0
from fcdmft.gw.mol.gw_exact import diagonalize_phrpa, get_transition_density, get_sigma


def kernel(gw):
    # local variables for convenience
    mf = gw._scf
    nmo = gw.nmo
    nocc = gw.nocc
    mo_energy = gw.mo_energy
    mo_coeff = gw.mo_coeff

    # get hcore and ovlp
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()

    # keep this condition for embedding calculations
    if (not isinstance(gw._scf, dft.rks.RKS)) and isinstance(gw._scf, scf.hf.RHF):
        rhf = gw._scf
    else:
        rhf = scf.RHF(gw.mol.copy(deep=True))
        rhf.verbose = rhf.mol.verbose = 0
        rhf.mo_energy = np.array(mo_energy, copy=True)
        rhf.mo_coeff = np.array(mo_coeff, copy=True)
        rhf.mo_occ = rhf.get_occ()
        if mf._eri is not None:
            rhf._eri = mf._eri
    dm_iter = rhf.make_rdm1()

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
            gw.Lpq = einsum('Lmn,mp,nq->Lpq', gw.Lpq_ao, mo_coeff, mo_coeff)

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
        CS = np.matmul(mo_coeff.T, ovlp)
        vsig = reduce(np.matmul, (CS.T, sigma, CS))

        # update veff
        if gw.vhf_df is False:
            veff = rhf.get_veff(dm=dm_iter)
        else:
            vj = 2.0 * einsum('Lii,Lpq->pq', gw.Lpq[:, :nocc, :nocc], gw.Lpq)
            vk = -einsum('Lpi,Liq->pq', gw.Lpq[:, :, :nocc], gw.Lpq[:, :nocc, :])
            veff = reduce(np.matmul, (CS.T, vj + vk, CS))

        # complete Hamiltonian through DIIS
        ham = hcore + veff + vsig
        ham = gw_diis.update(ovlp, dm_iter, ham)

        # diagonalize
        gw.mo_energy, gw.mo_coeff = mo_energy, mo_coeff = scipy.linalg.eigh(ham, ovlp)

        # check density matrix convergence
        dm_old = dm_iter.copy()
        # update QSGW density matrix
        dm_iter = rhf.make_rdm1(mo_coeff=mo_coeff)
        norm_dm = np.linalg.norm(dm_iter - dm_old) / nmo
        if norm_dm < gw.conv_tol:
            conv = True

        logger.info(gw, 'QSGW cycle= %d  |ddm|= %4.3g', cycle + 1, norm_dm)
        with np.printoptions(threshold=nmo):
            logger.debug(gw, '  mo_energy =\n%s', mo_energy)
        cycle += 1

    logger.debug(gw, 'QSGWExact %s in %-d cycles.', 'converged' if conv else 'not converged', cycle + 1)

    with np.printoptions(threshold=nmo):
        logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)

    return


class QSGWExact(GWAC):
    def __init__(self, mf, auxbasis=None):
        GWAC.__init__(self, mf, frozen=None, auxbasis=auxbasis)
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
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
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
        gf: complex 3d array
            qsGW Green's function in mean-field MO space.
        gf0 : complex 3d array
            non-interacting Green's function.
        sigma : complex 3d array
            self-energy (including 1e correction) in mean-field MO space.
        """
        mf = self._scf
        nocc = self.nocc
        exci = self.exci
        rho = self.rho

        mo_energy = np.asarray(self.mo_energy)

        # get self-energy
        # NOTE: this is for G0W0 Green's function, so mo_energy is from the SCF calculation
        energy_occ = omega[:, None, None] - mo_energy[None, :nocc, None] + (exci[None, None, :] - 3.0 * 1j * eta)
        energy_vir = omega[:, None, None] - mo_energy[None, nocc:, None] - (exci[None, None, :] - 3.0 * 1j * eta)
        energy = np.concatenate([energy_occ, energy_vir], axis=1)
        energy = 1.0 / energy

        sigma = einsum('mpr,mqr,wrm->pqw', rho, rho, energy)
        sigma *= 2.0  # for two spin channels

        # get self-energy difference in the one-electron Hamiltonian in mean-field MO space
        if (not isinstance(self._scf, dft.rks.RKS)) and isinstance(self._scf, scf.hf.RHF):
            rhf = self._scf
        else:
            rhf = scf.RHF(self.mol)
        with temporary_env(mf, verbose=0):
            ovlp = mf.get_ovlp()
            hcore = mf.get_hcore()
            dm_mf = mf.make_rdm1()
            dm_qp = mf.make_rdm1(mo_coeff=self.mo_coeff)
            veff_mf = mf.get_veff(dm=dm_mf)

        veff_qp = rhf.get_veff(dm=dm_qp)
        C_qp_mf = reduce(np.matmul, (self.mo_coeff.T, ovlp, mf.mo_coeff))
        ham_mf = reduce(np.matmul, (mf.mo_coeff.T, hcore + veff_mf, mf.mo_coeff))
        ham_qp = reduce(np.matmul, (mf.mo_coeff.T, hcore + veff_qp, mf.mo_coeff))

        gf0 = get_g0(omega=omega, mo_energy=mf.mo_energy, eta=eta)

        gf = np.zeros_like(gf0)
        for iw in range(len(omega)):
            sigma[..., iw] = reduce(np.matmul, (C_qp_mf.T, sigma[..., iw], C_qp_mf))
            sigma[..., iw] += ham_qp - ham_mf
            gf[..., iw] = np.linalg.inv(np.linalg.inv(gf0[..., iw]) - sigma[..., iw])

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
            one-particle density matrix
        """
        # get qsGW G
        ef = (self.mo_energy[self.nocc - 1] + self.mo_energy[self.nocc]) * 0.5
        freqs, wts = _get_scaled_legendre_roots(nw)
        gf = self.make_gf(omega=1j * freqs + ef, eta=0)[0]

        # qsGW density matrix in mean-field MO basis
        rdm1 = 2.0 / np.pi * einsum('ijw,w->ij', gf, wts).real + np.eye(self.nmo)

        # symmetrize density matrix
        rdm1 = 0.5 * (rdm1 + rdm1.T)
        logger.info(self, 'GW particle number = %s', np.trace(rdm1))

        if ao_repr is True:
            rdm1 = reduce(np.matmul, (self._scf.mo_coeff, rdm1, self._scf.mo_coeff.T))

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
        ef = (self.mo_energy[self.nocc - 1] + self.mo_energy[self.nocc]) * 0.5
        freqs, wts = _get_scaled_legendre_roots(nw)
        gf = self.make_gf(omega=1j * freqs + ef, eta=0)[0]

        # gf0 = 1 / (iw + ef - Fock), not from make_gf
        dm = self.make_rdm1(nw=nw, ao_repr=True)
        if (not isinstance(self._scf, dft.rks.RKS)) and isinstance(self._scf, scf.hf.RHF):
            rhf = self._scf
        else:
            rhf = scf.RHF(self.mol)
        fock = reduce(np.matmul, (self._scf.mo_coeff.T, rhf.get_fock(dm=dm), self._scf.mo_coeff))
        gf0 = np.zeros_like(gf)
        for iw in range(nw):
            gf0[:, :, iw] = np.linalg.inv((ef + 1j * freqs[iw]) * np.eye(self.nmo) - fock)

        # sigma = G0^-1 - G^-1, not from make_gf
        sigma = np.zeros_like(gf0)
        for iw in range(nw):
            sigma[:, :, iw] = np.linalg.inv(gf0[:, :, iw]) - np.linalg.inv(gf[:, :, iw])

        # evaluate HF energy with qsGW density matrix
        with temporary_env(rhf, verbose=0):
            e_hf = rhf.energy_tot(dm=dm)
        e_c = einsum('ijw,jiw,w->', gf, sigma, wts).real / np.pi
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
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    gw = QSGWExact(mf)
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.45766166) < 1e-4
    assert abs(gw.mo_energy[5] - 0.16255341) < 1e-4

    # Galitskii-Migdal total energy
    # numerically not stable because it's highly sensitive to final quasiparticle energy and orbital
    e_tot, e_hf, e_c = gw.energy_tot()

    # density matrix
    dm = gw.make_rdm1()

    # generate Green's function on real axis and print density of states
    omega = np.linspace(-0.5, 0.5, 101)
    gf, gf0, _ = gw.make_gf(omega=omega, eta=0.01)
    print('\nDOS: KS, GW')
    for iw in range(len(omega)):
        print(omega[iw], -np.trace(gf0[:, :, iw].imag) / np.pi, -np.trace(gf[:, :, iw].imag) / np.pi)

    print('passed tests!')
