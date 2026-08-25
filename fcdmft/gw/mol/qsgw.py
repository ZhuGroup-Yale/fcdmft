"""
Spin-restricted quasiparticle self-consistent GW method based on the analytic continuation scheme.

References:
    J. Lei and T. Zhu, J. Chem. Phys. 157, 214114 (2022)
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    Phys. Rev. B 76, 165106 (2007)
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
"""

from functools import reduce
import h5py
import numpy as np
import os
import scipy

from pyscf import dft, scf
from pyscf.lib import einsum, logger, temporary_env

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
    eta = gw.eta
    nocc_full = int(np.sum(gw._scf.mo_occ)) // 2

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
    gw.hf_obj = rhf = gw.set_hf_object()
    dm_iter = rhf.make_rdm1(mo_energy=gw.mo_energy, mo_coeff=gw.mo_coeff)

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
            gw.Lpq = einsum('Lmn,mp,nq->Lpq', gw.Lpq_ao, gw.mo_coeff, gw.mo_coeff)

        # compute full self-energy on imaginary axis
        if gw.outcore:
            sigmaI, omega = get_sigma_outcore(
                gw, orbs_frz, quad_freqs, quad_wts, ef, mo_energy_frz, mo_coeff_frz, mo_occ=mo_occ_frz, iw_cutoff=gw.ac_iw_cutoff,
                eval_freqs=eval_freqs_with_zero, fullsigma=True
            )
        else:
            sigmaI, omega = get_sigma(
                gw, orbs_frz, gw.Lpq, quad_freqs, quad_wts, ef, mo_energy_frz, mo_occ=mo_occ_frz, iw_cutoff=gw.ac_iw_cutoff,
                eval_freqs=eval_freqs_with_zero, fullsigma=True
            )

        # analytic continuation
        if gw.ac == 'twopole':
            acobj = TwoPoleAC(list(range(nmo)), nocc)
        elif gw.ac == 'pade':
            acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
            if gw.ac_idx is not None:
                acobj.idx = gw.ac_idx
        else:
            raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))
        acobj.ac_fit(sigmaI, omega, axis=-1)

        if gw.mode == 'b':
            sigma_at_fermi = acobj.ac_eval(1j * eta + ef)
            nfrz_sigma = 0.5 * (sigma_at_fermi + sigma_at_fermi.T.conj())
            acobj_diag = acobj.diagonal()
            for ip, p in enumerate(orbs_frz):
                nfrz_sigma[ip, ip] = acobj_diag[ip].ac_eval(mo_energy_frz[p] + 1j * eta).real
        elif gw.mode == 'a':
            nfrz_sigma = np.zeros(shape=[len(orbs_frz), len(orbs_frz)], dtype=np.complex128)
            for ip, p in enumerate(orbs_frz):
                    for iq, q in enumerate(orbs_frz):
                        nfrz_sigma[ip, iq] += 0.25 * acobj[ip, iq].ac_eval(mo_energy_frz[p] + 1j * eta)
                        nfrz_sigma[ip, iq] += 0.25 * acobj[iq, ip].ac_eval(mo_energy_frz[p] + 1j * eta).conj()
                        nfrz_sigma[ip, iq] += 0.25 * acobj[ip, iq].ac_eval(mo_energy_frz[q] + 1j * eta)
                        nfrz_sigma[ip, iq] += 0.25 * acobj[iq, ip].ac_eval(mo_energy_frz[q] + 1j * eta).conj()
        elif gw.mode == 'c':
            sigma_at_fermi = acobj.ac_eval(1j * eta + ef)
            nfrz_sigma = 0.5 * (sigma_at_fermi + sigma_at_fermi.T.conj())
        else:
            raise ValueError('Unknown QSGW mode %s' % gw.mode)

        sigma = np.zeros(shape=[nmo, nmo], dtype=np.complex128)
        sigma[np.ix_(orbs_frz, orbs_frz)] = nfrz_sigma

        # get static correlation self-energy in AO basis
        CS = mo_coeff_frz.T @ ovlp
        vsig = CS.T @ sigma.real @ CS

        # update veff
        if gw.vhf_df is False:
            veff = rhf.get_veff(dm=dm_iter)
        else:
            vj = 2.0 * einsum('Lii,Lpq->pq', gw.Lpq[:, :nocc, :nocc], gw.Lpq)
            vk = -einsum('Lpi,Liq->pq', gw.Lpq[:, :, :nocc], gw.Lpq[:, :nocc, :])
            veff = CS.T @ (vj + vk) @ CS

        if gw.fc:
            vk_corr = -2.0 / np.pi * (6.0 * np.pi**2 / gw.mol.vol) ** (1.0 / 3.0)
            veff = gw.mo_coeff.T @ veff @ gw.mo_coeff

            for i in range(nocc_full):
                veff[i, i] = veff[i, i] + vk_corr

            CS = gw.mo_coeff.T @ ovlp
            veff = CS.T @ veff @ CS

        # complete Hamiltonian through DIIS
        ham = hcore + veff + vsig
        ham = gw_diis.update(ovlp, dm_iter, ham)

        # update mo_energy, mo_coeff, mo_occ and ef
        gw.mo_energy, gw.mo_coeff = scipy.linalg.eigh(ham, ovlp)
        gw.mo_occ = rhf.get_occ(mo_energy=gw.mo_energy, mo_coeff=gw.mo_coeff)
        gw.ef = ef = gw.get_ef()
        gw.acobj = acobj

        # update non-frozen quantities
        mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
        mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)
        mo_occ_frz = _mo_energy_without_core(gw, gw.mo_occ)

        # check density matrix convergence
        dm_old = dm_iter.copy()
        # update QSGW density matrix
        dm_iter = rhf.make_rdm1(mo_coeff=gw.mo_coeff, mo_occ=gw.mo_occ)
        norm_dm = np.linalg.norm(dm_iter - dm_old) / nmo
        if norm_dm < gw.conv_tol:
            conv = True

        logger.info(gw, 'QSGW cycle= %d  |ddm|= %4.3g', cycle + 1, norm_dm)
        cycle += 1
        with np.printoptions(threshold=len(gw.mo_energy)):
            logger.debug(gw, '  GW mo_energy =\n%s', gw.mo_energy)

        if gw.chkfile:
            gw.dump_chk()

    logger.debug(gw, 'QSGW %s in %-d cycles.', 'converged' if conv else 'not converged', cycle + 1)

    with np.printoptions(threshold=len(gw.mo_energy)):
        logger.debug(gw, '  GW mo_energy =\n%s', gw.mo_energy)

    return


def qs_sigmaR_mode_a(qp_energy, acobj):
    """Mode A to evaluate static qsGW self-energy.
    Equation 15 in doi.org/10.1063/5.0125756
    Equation 10 in doi.org/10.1103/PhysRevB.76.165106

    Parameters
    ----------
    qp_energy : double array
        quasiparticle energy
    acobj : fcdmft.ac.interface.AC_method
        analytical continuation object

    Returns
    -------
    sigmaR: double ndarray
        full static correlation self-energy matrix
    """
    norbs = qp_energy.shape[0]
    sigmaR = acobj.ac_eval(qp_energy).real  # ncoeff x ncoeff x nQP

    p_indices = np.arange(norbs).reshape(-1, 1)  # Column vector
    q_indices = np.arange(norbs).reshape(1, -1)  # Row vector

    # sigmaR_{pq} = 0.5*(sigmaR_{pq}(eps_{p}) + sigmaR_{pq}(eps_{q}))
    sigmaR = 0.5 * (sigmaR[p_indices, q_indices, p_indices] + sigmaR[p_indices, q_indices, q_indices])

    # slow but correct version with loops
    # sigmaR = np.zeros((norbs, norbs))
    # for p in range(norbs):
    #   for q in range(norbs):
    #       sigmaR[p, q] = 0.5*(acobj.ac_eval(qp_energy[p]).real[p,q] + acobj.ac_eval(qp_energy[q]).real)[p,q]

    # diff = sigmaR_a - sigmaR
    # print(np.linalg.norm(diff))

    return sigmaR


def qs_sigmaR_mode_b(qp_energy, acobj, ef):
    """Mode B to evaluate static qsGW self-energy.
    Equation 16 in doi.org/10.1063/5.0125756
    Equation 11 in doi.org/10.1103/PhysRevB.76.165106

    Parameters
    ----------
    qp_energy : double array
        quasiparticle energy
    acobj : fcdmft.ac.interface.AC_method
        analytical continuation object
    ef : double
        Fermi level

    Returns
    -------
    sigmaR: double ndarray
        full static correlation self-energy matrix
    """
    norbs = qp_energy.shape[0]
    mask = np.ones((norbs, norbs)) - np.eye(norbs)

    ii = np.diagonal(acobj.ac_eval(qp_energy).real, axis1=0, axis2=1)
    ij = acobj.ac_eval(ef).real[:, :, 0]
    sigmaR = (1.0 - mask) * ii + mask * ij

    # slow but correct version with loops
    # sigmaR = np.zeros((norbs, norbs))
    # for p in range(norbs):
    #     for q in range(norbs):
    #         if p != q:
    #             sigmaR[p, q] = acobj.ac_eval(ef).real[p,q]
    #         else:
    #             sigmaR[p, q] = acobj.ac_eval(qp_energy[p]).real[p,p]

    # diff = mask*(sigmaR - ij)
    # print(np.linalg.norm(diff))
    # diff = (1.0 - mask)*(ii - sigmaR)
    # print(np.linalg.norm(diff))
    # diff = sigmaR - ((1.0 - mask)*ii + mask*ij)
    # print(np.linalg.norm(diff))

    return sigmaR


class QSGW(GWAC):
    def __init__(self, mf, frozen=None, auxbasis=None):
        GWAC.__init__(self, mf, frozen=frozen, auxbasis=auxbasis)
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
        self.hf_obj = None  # Hartree-Fock object for hcore, J and K

        return

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
        log.info('frozen orbitals = %s', self.frozen)
        log.info('GW density matrix = %s', self.rdm)
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
        log.info('imaginary frequency cutoff = %s', str(self.ac_iw_cutoff))
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
        """Run a quasiparticle self-consistent GW calculation."""
        # smeared GW needs denser grids to be accurate
        if hasattr(self._scf, 'sigma'):
            assert self.frozen == 0 or self.frozen is None
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0

        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return

    def make_gf(self, omega, eta=0.0, mode = 'dyson'):
        """Get qsGW Green's function by AC fitting.

        Parameters
        ----------
        omega : complex 1d array
            frequency on which to evaluate the Green's function
        eta : double, optional
            broadening parameter. Defaults to 0.
        mode : str, one of dyson or linear
            mode for Dyson equation, 'linear' or 'dyson', by default 'dyson'

        Returns
        -------
        gf: complex 3d array
            qsGW Green's function in mean-field MO space.
        gf0 : complex 3d array
            non-interacting Green's function.
        sigma : complex 3d array
            self-energy (including 1e correction) in mean-field MO space.
        """
        assert mode in ['dyson', 'linear']
        assert self.frozen is None or self.frozen == 0

        mf = self._scf
        sigma = self.acobj.ac_eval(omega + 1.0j * eta)

        # get self-energy difference in the one-electron Hamiltonian in mean-field MO space
        with temporary_env(mf, verbose=0):
            ovlp = mf.get_ovlp()
            hcore = mf.get_hcore()
            dm_mf = mf.make_rdm1()
            dm_qp = mf.make_rdm1(mo_coeff=self.mo_coeff)
            veff_mf = mf.get_veff(dm=dm_mf)

        veff_qp = self.hf_obj.get_veff(dm=dm_qp)
        C_qp_mf = reduce(np.matmul, (self.mo_coeff.T, ovlp, mf.mo_coeff))
        ham_mf = reduce(np.matmul, (mf.mo_coeff.T, hcore + veff_mf, mf.mo_coeff))
        ham_qp = reduce(np.matmul, (mf.mo_coeff.T, hcore + veff_qp, mf.mo_coeff))

        gf0 = get_g0(omega=omega, mo_energy=mf.mo_energy, eta=eta)

        sigma = reduce(np.matmul, (C_qp_mf.T[np.newaxis, :, :], sigma.T, C_qp_mf[np.newaxis, :, :])).T
        sigma += ham_qp[:, :, np.newaxis] - ham_mf[:, :, np.newaxis]

        if mode == 'dyson':
            gf = np.linalg.inv(np.linalg.inv(gf0.T) - sigma.T).T
        elif mode == 'linear':
            gf = gf0 + reduce(np.matmul, (gf0.T, sigma.T, gf0.T)).T

        return gf, gf0, sigma

    def make_rdm1(self, ao_repr=False, mode='dyson'):
        """make qsGW 1-particle density matrix in either MO or AO space

        Parameters
        ----------
        ao_repr : bool, optional
            return dm in AO space instead of MO space, by default False
        mode : str
            mode for Dyson equation, 'linear' or 'dyson', by default 'dyson'
        Returns
        -------
        rdm1 : double 2d array
            one-particle density matrix
        """

        # get qsGW G in MO space
        ef = self.ef
        gf = self.make_gf(omega=1j * self.freqs + ef, eta=0, mode=mode)[0]

        # density matrix in MO space
        rdm1 = 2.0 / np.pi * einsum('ijw,w->ij', gf, self.wts).real + np.eye(self.nmo)

        logger.info(self, 'GW particle number = %s', np.trace(rdm1))

        # Symmetrize density matrix
        rdm1 = 0.5 * (rdm1 + rdm1.T)

        if ao_repr is True:
            rdm1 = reduce(np.matmul, (self._scf.mo_coeff, rdm1, self._scf.mo_coeff.T))

        return rdm1

    def energy_tot(self, mode='dyson'):
        """ "Get qsGW total energy using Galitskii-Migdal formula.
        V. M. Galitskii and A. B. Migdal, Zh. E ksp. Teor. Fiz. 34, 139~1958! @Sov. Phys. JETP 139, 96 ~1958!#
        working equation: equation.A5 in Phys. Rev. B 88, 075105

        Hartree-Fock energy is evaluated with qsGW density matrix.
        G = [G0^-1 - sigma]^-1
        where G0 = [w - Fock]^-1
        and sigma = sigma_c + (ham_qp - ham_mf) = sigma_c + 1e correction

        Parameters
        mode : str
            mode for Dyson equation, 'linear' or 'dyson', by default 'dyson'
        ----------

        Returns
        -------
        etot : double
            total energy
        ehf : double
            Hartree-Fock energy
        ec : double
            correlation energy
        """
        ef = self.ef
        gf = self.make_gf(omega=1j * self.freqs + ef, eta=0, mode = mode)[0]

        dm_qs_mo = 2.0 / np.pi * einsum('ijw,w->ij', gf, self.wts).real + np.eye(self.nmo)
        dm_ao = reduce(np.matmul, (self._scf.mo_coeff, dm_qs_mo, self._scf.mo_coeff.T))
        dm_ao = 0.5 * (dm_ao + dm_ao.T)

        fock = reduce(np.matmul, (self._scf.mo_coeff.T, self.hf_obj.get_fock(dm=dm_ao), self._scf.mo_coeff))
        gf0 = (ef + 1j * self.freqs[:, None, None]) * np.eye(fock.shape[0])[None, :, :] - fock[None, :, :]
        gf0 = np.linalg.inv(gf0).T

        sigma = (np.linalg.inv(gf0.T) - np.linalg.inv(gf.T)).T

        ehf = self.hf_obj.energy_tot(dm=dm_ao)
        ec = einsum('ijw,jiw,w->', gf, sigma, self.wts).real / np.pi
        etot = ec + ehf

        return etot, ehf, ec

    def set_hf_object(self):
        """Set Hartree-Fock object for hcore, J and K.
        This function should be overwritten for supercell.

        Returns
        -------
        rhf : pyscf.scf.rhf.RHF
            RHF object
        """
        if (not isinstance(self._scf, dft.rks.RKS)) and isinstance(self._scf, scf.hf.RHF):
            rhf = self._scf
        else:
            rhf = self._scf.to_hf()
            if self._scf._eri is not None:
                rhf._eri = self._scf._eri
        return rhf


if __name__ == '__main__':
    from pyscf import gto, dft, scf

    mol_test = gto.Mole()
    mol_test.verbose = 5
    mol_test.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol_test.basis = 'def2-svp'
    mol_test.build()

    mf_test = dft.RKS(mol_test)
    mf_test.xc = 'pbe0'
    mf_test.kernel()

    gw_test = QSGW(mf_test)
    gw_test.kernel()
    assert abs(gw_test.mo_energy[4] - -0.45770292) < 1e-4
    assert abs(gw_test.mo_energy[5] - 0.16254796) < 1e-4

    # use fewer points in integration
    gw_test = QSGW(mf_test)
    gw_test.nw2 = 30
    gw_test.kernel()
    assert abs(gw_test.mo_energy[4] - -0.45770292) < 1e-4
    assert abs(gw_test.mo_energy[5] - 0.16254796) < 1e-4

    # outcore, low-memory mode
    gw_test = QSGW(mf_test)
    gw_test.outcore = True
    gw_test.kernel()
    assert abs(gw_test.mo_energy[4] - -0.45770292) < 1e-4
    assert abs(gw_test.mo_energy[5] - 0.16254796) < 1e-4

    # An example for using chkfile
    gw_test = QSGW(mf_test)
    # set chkfile for saving qsgw results
    gw_test.chkfile = 'h2o_qsgw.chk'
    gw_test.max_cycle = 5
    gw_test.kernel()

    gw_test = QSGW(mf_test)
    # load qsgw chkfile
    gw_test.load_chkfile = True
    gw_test.chkfile = 'h2o_qsgw.chk'
    gw_test.kernel()
    assert abs(gw_test.mo_energy[4] - -0.45770292) < 1e-4
    assert abs(gw_test.mo_energy[5] - 0.16254796) < 1e-4
    os.remove('h2o_qsgw.chk')

    # make density matrix and density of states
    gw_test = QSGW(mf_test)
    gw_test.rdm = True
    gw_test.kernel()
    dm_test = gw_test.make_rdm1(ao_repr=False)
    print('QSGW density matrix in MO\n', dm_test)
    omega_test = np.linspace(-0.5, 0.5, 101)
    gf_test, gf0_test, sigma_test = gw_test.make_gf(omega=omega_test, eta=0.01)
    print('\nDOS: KS, GW')
    for iw in range(len(omega_test)):
        print(omega_test[iw], -np.trace(gf0_test[:, :, iw].imag) / np.pi, -np.trace(gf_test[:, :, iw].imag) / np.pi)

    print('passed tests!')
