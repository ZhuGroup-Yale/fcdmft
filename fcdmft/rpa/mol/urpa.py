"""
Spin-unrestricted random phase approximation (direct RPA) with N^4 scaling.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14, 053020 (2012)
"""

import numpy as np
import scipy
import time

from pyscf import dft, lib, scf
from pyscf.lib import logger, temporary_env
from pyscf.ao2mo._ao2mo import nr_e2
from pyscf.mp.ump2 import get_nocc, get_nmo, get_frozen_mask

from fcdmft.rpa.mol.rpa import RPA
from fcdmft.gw.minimax.minimax_rpa import get_frequency_minimax_grid
from fcdmft.ac.grids import _get_scaled_legendre_roots
from fcdmft.gw.mol.ugw_ac import get_rho_response, _mo_energy_without_core, _mo_without_core


def kernel(rpa, mo_energy, mo_coeff, Lpq=None, nw=None):
    """RPA correlation and total energy

    Parameters
    ----------
    rpa : RPA
        rpa object
    mo_energy : double 2d array
        molecular orbital energies
    mo_coeff : double 3d array
        molecular orbital coefficients
    Lpq : double array, optional
        density fitting 3-center integral in MO basis, by default None
    nw : int, optional
        number of frequency point on imaginary axis, by default None

    Returns
    -------
    e_tot : float
        RPA total energy
    e_hf : float
        HF energy (exact exchange for given mo_coeff)
    e_corr : float
        RPA correlation energy
    """
    # only support frozen core
    if rpa.frozen is not None:
        assert isinstance(rpa.frozen, int)

    if Lpq is None:
        rpa.initialize_df(auxbasis=rpa.auxbasis)
        if rpa.outcore is False:
            with temporary_env(rpa.with_df, verbose=0), temporary_env(rpa.mol, verbose=0):
                Lpq = rpa.ao2mo(mo_coeff)

    # Grids for integration on imaginary axis
    freqs, wts = rpa.get_grids(nw=nw, mo_energy=mo_energy)

    # Compute HF exchange energy (EXX)
    e_hf = rpa.get_ehf()

    # Compute RPA correlation energy
    e_corr = get_rpa_ecorr(rpa, Lpq, freqs, wts)

    # Compute total energy
    e_tot = e_hf + e_corr

    logger.debug(rpa, '  RPA total energy = %s', e_tot)
    logger.debug(rpa, '  EXX energy = %s, RPA corr energy = %s', e_hf, e_corr)

    return e_tot, e_hf, e_corr


def get_rpa_ecorr(rpa, Lpq, freqs, wts):
    """Compute RPA correlation energy.

    Parameters
    ----------
    rpa : URPA
        rpa object
    Lpq : double 4d array
        density fitting 3-center integral in MO basis
    freqs : double 1d array
        frequency grid
    wts : double 1d array
        weight of grids

    Returns
    -------
    e_corr : double
        RPA correlation energy
    """
    mo_energy = _mo_energy_without_core(rpa, rpa._scf.mo_energy)
    nocca, noccb = rpa.nocc
    nw = len(freqs)
    naux = Lpq[0].shape[0]

    homo = max(mo_energy[0][nocca - 1], mo_energy[1][noccb - 1])
    lumo = min(mo_energy[0][nocca], mo_energy[1][noccb])
    if (lumo - homo) < 1e-3:
        logger.warn(rpa, 'Current RPA code not well-defined for degeneracy!')

    is_metal = hasattr(rpa._scf, 'sigma')

    e_corr = 0.0
    for w in range(nw):
        if is_metal:
            raise NotImplementedError('Metal calculations not implemented with urpa')
        else:
            Pi = get_rho_response(freqs[w], mo_energy, Lpq[0, :, :nocca, nocca:], Lpq[1, :, :noccb, noccb:])
        ec_w = np.linalg.slogdet((np.eye(naux) - Pi))[1]
        ec_w += np.trace(Pi)
        e_corr += 1.0 / (2.0 * np.pi) * ec_w * wts[w]

    return e_corr


def get_idx_metal(mo_occ, threshold=1.0e-6):
    """Get index of occupied/virtual/fractional orbitals of metals.

    Parameters
    ----------
    mo_occ : double 1d array
        occupation number
    threshold : double, optional
        threshold to determine fractionally occupied orbitals, by default 1.0e-6

    Returns
    -------
    idx_occ : list
        list of occupied orbital indexes
    idx_frac : list
        list of fractionally occupied orbital indexes
    idx_vir : list
        list of virtual orbital indexes
    """
    idx_occ = np.where(mo_occ > 1.0 - threshold)[0]
    idx_vir = np.where(mo_occ < threshold)[0]
    idx_frac = list(range(idx_occ[-1] + 1, idx_vir[0]))

    return idx_occ, idx_frac, idx_vir


def get_rpa_exx(rpa, acfd=False, correction_only=False):
    """Calculate RPA exchange energy.
    For gapped systems, Hartree-Fock and adiabatic connection fluctuation dissipation exchange energies are the same.
    For metallic systems, they are different.
    The ACFD exchange energy is given by equation 12 in doi.org/10.1103/PhysRevB.81.115126

    Parameters
    ----------
    rpa : RPA
        rpa object
    acfd : bool, optional
        calculate ACFD exchange energy, by default False
    correction_only : bool, optional
        only calculate the correction term, by default False

    Returns
    -------
    ex : double
        exchange energy
    """
    mo_energy = np.array(rpa._scf.mo_energy)
    mo_coeff = np.array(rpa._scf.mo_coeff)
    mo_occ = np.array(rpa._scf.mo_occ)

    if not hasattr(rpa._scf, 'sigma'):
        if correction_only:
            return 0.0

        dm = np.asarray(rpa._scf.make_rdm1())
        with temporary_env(rpa.mol, verbose=0):
            uhf = scf.UHF(rpa.mol)
            uhf.verbose = 0
            vk = np.asarray(uhf.get_k(uhf.mol, dm))
        return -0.5 * lib.einsum('sij,sji->', dm, vk)

    if getattr(rpa, 'with_df', None) is None:
        rpa.initialize_df(auxbasis=rpa.auxbasis)

    ex = 0.0
    for spin in range(2):
        idx_occ, idx_frac, _ = get_idx_metal(mo_occ[spin])
        nocc = len(idx_occ) + len(idx_frac)
        mo = np.asarray(mo_coeff[spin], order='F')
        Lij = nr_e2(rpa.with_df._cderi, mo, (0, nocc, 0, nocc), aosym='s2')
        Lij = Lij.reshape(-1, nocc, nocc)

        if acfd is True:
            if correction_only is True:
                mo_occ_ij = np.minimum(mo_occ[spin][:nocc, None], mo_occ[spin][None, :nocc])
                mo_occ_ij -= mo_occ[spin][:nocc, None] * mo_occ[spin][None, :nocc]
            else:
                # The following line is equivalent to the frequency integration in equation 12 in
                # doi.org/10.1103/PhysRevB.81.115126
                # TODO: add a detailed note
                eij = mo_energy[spin][:nocc, None] - mo_energy[spin][None, :nocc]
                integrand = np.zeros((nocc, nocc), dtype=np.double)
                integrand[eij > 1e-6] = 1
                integrand[eij < -1e-6] = -1
                mo_occ_ij = 1.0 - integrand
                mo_occ_ij = mo_occ_ij * mo_occ[spin][:nocc, None]
        else:
            mo_occ_ij = mo_occ[spin][:nocc, None] * mo_occ[spin][None, :nocc]
        Lij_occ = Lij * mo_occ_ij[None]
        ex -= scipy.linalg.blas.ddot(Lij_occ.ravel(), Lij.ravel())

    ex *= 0.5
    return ex


def as_scanner(rpa):
    """Generating a scanner/solver for RPA PES."""
    if isinstance(rpa, lib.SinglePointScanner):
        return rpa

    logger.info(rpa, 'Set %s as a scanner', rpa.__class__)

    class RPA_Scanner(rpa.__class__, lib.SinglePointScanner):
        def __init__(self, rpa):
            self.__dict__.update(rpa.__dict__)
            self._scf = rpa._scf.as_scanner()

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

    return RPA_Scanner(rpa)


class URPA(RPA):
    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira = nmoa - nocca
        nvirb = nmob - noccb
        log.info('RPA (nocca, noccb) = (%d, %d), (nvira, nvirb) = (%d, %d)', nocca, noccb, nvira, nvirb)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', str(self.frozen))
        log.info('grid type = %s', self.grids_alg)
        # TODO: add outcore
        # log.info('outcore mode = %s', self.outcore)
        # if self.outcore is True:
        #     log.info('outcore segment size = %d', self.segsize)
        log.info('')
        return

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    as_scanner = as_scanner

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, nw=None):
        """RPA correlation and total energy

        Calculated total energy, HF energy and RPA correlation energy
        are stored in self.e_tot, self.e_hf, self.e_corr

        Parameters
        ----------
        mo_energy : double 2d array
            molecular orbital energies
        mo_coeff : double 3d array
            molecular orbital coefficients
        Lpq : double 4d array, optional
            density fitting 3-center integral in MO basis, by default None
        nw : int, optional
            number of frequency point on imaginary axis, by default None

        Returns
        -------
        self.e_corr : float
            RPA correlation energy
        """
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self._scf.mo_energy)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.e_tot, self.e_hf, self.e_corr = kernel(self, mo_energy, mo_coeff, Lpq=Lpq, nw=nw)

        logger.timer(self, 'RPA', *cput0)
        return self.e_corr

    def ao2mo(self, mo_coeff=None):
        """Transform density-fitting integral from AO to MO.

        Parameters
        ----------
        mo_coeff : double 3d array, optional
            coefficient from AO to MO, by default None

        Returns
        -------
        Lpq : double 4d array
            three-center density-fitting matrix in MO
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmoa = mo_coeff[0].shape[1]
        nmob = mo_coeff[1].shape[1]
        nao = self.mo_coeff[0].shape[0]
        naux = self.with_df.get_naoaux()
        mem_incore = (nmoa**2 * naux + nmob**2 * naux + nao**2 * naux) * 8 / 1e6
        mem_now = lib.current_memory()[0]

        moa = np.asarray(mo_coeff[0], order='F')
        mob = np.asarray(mo_coeff[1], order='F')
        ijslicea = (0, nmoa, 0, nmoa)
        ijsliceb = (0, nmob, 0, nmob)
        Lpqa = None
        Lpqb = None
        if (mem_incore + mem_now < 0.99 * self.max_memory) or self.mol.incore_anyway:
            Lpqa = nr_e2(self.with_df._cderi, moa, ijslicea, aosym='s2', out=Lpqa)
            Lpqb = nr_e2(self.with_df._cderi, mob, ijsliceb, aosym='s2', out=Lpqb)
            return np.asarray((Lpqa.reshape(naux, nmoa, nmoa), Lpqb.reshape(naux, nmob, nmob)))
        else:
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError

    def get_grids(self, alg=None, nw=None, mo_energy=None):
        """Generate grids for integration.

        Parameters
        ----------
        alg : str, optional
            algorithm for generating grids, by default None
        nw : int, optional
            number of grids, by default None
        mo_energy : double 2d array, optional
            orbital energy, used for minimax grids, by default None

        Returns
        -------
        freqs : double 1d array
            frequency grid
        wts : double 1d array
            weight of grids
        """
        if alg is None:
            alg = self.grids_alg
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self._scf.mo_energy)
        if alg == 'legendre':
            nw = 40 if nw is None else nw
            freqs, wts = _get_scaled_legendre_roots(nw)
        elif alg == 'minimax':
            nw = 30 if nw is None else nw
            E_max = max(mo_energy[0, -1] - mo_energy[0, 0], mo_energy[1, -1] - mo_energy[1, 0])
            E_min = min(
                mo_energy[0, self.nocc[0]] - mo_energy[0, self.nocc[0] - 1],
                mo_energy[1, self.nocc[1]] - mo_energy[1, self.nocc[1] - 1],
            )
            E_range = E_max / E_min
            freqs, wts = get_frequency_minimax_grid(nw, E_range)
        else:
            raise NotImplementedError('Grids algorithm not implemented!')

        return freqs, wts

    def get_ehf(self):
        """Get Hartree-Fock energy.

        Returns
        -------
        e_hf : double
            Hartree-Fock energy
        """
        with temporary_env(self.with_df, verbose=0), temporary_env(self.mol, verbose=0):
            uhf = scf.UHF(self.mol)
            uhf.verbose = 0
            if hasattr(self._scf, 'sigma'):
                uhf = scf.addons.smearing_(uhf, sigma=self._scf.sigma, method=self._scf.smearing_method)
            with temporary_env(self.with_df, verbose=0), temporary_env(uhf.mol, verbose=0):
                dm = np.asarray(self._scf.make_rdm1())
                dm_tot = dm[0] + dm[1]
                hcore = uhf.get_hcore()
                vj = np.asarray(uhf.get_j(uhf.mol, dm))
                e_1e = lib.einsum('sij,ji->', dm, hcore)
                e_j = 0.5 * lib.einsum('ij,ji', dm_tot, vj[0] + vj[1])
                e_x = get_rpa_exx(self, acfd=self.acfd_exx, correction_only=False)
                e_nuc = self._scf.energy_nuc()
                e_hf = e_1e + e_j + e_x + e_nuc
        return e_hf

    def get_acfd_exx(self, correction_only=False):
        """Calculate ACFD exchange energy.

        Parameters
        ----------
        correction_only : bool
            only return the correction term

        Returns
        -------
        ex_acfd : double
            ACFD exchange energy
        """
        ex_acfd = get_rpa_exx(self, acfd=True, correction_only=correction_only)
        return ex_acfd


if __name__ == '__main__':
    from pyscf import gto, dft, scf

    mol_test = gto.Mole()
    mol_test.verbose = 5
    mol_test.atom = 'F 0 0 0'
    mol_test.basis = 'def2-svp'
    mol_test.spin = 1
    mol_test.build()

    mf_test = dft.UKS(mol_test)
    mf_test.xc = 'pbe0'
    mf_test.kernel()

    rpa_test = URPA(mf_test)
    rpa_test.kernel()
    print('RPA e_tot, e_hf, e_corr = ', rpa_test.e_tot, rpa_test.e_hf, rpa_test.e_corr)
    assert abs(rpa_test.e_corr - -0.20980646878974454) < 1e-6
    assert abs(rpa_test.e_tot - -99.49292565821425) < 1e-6

    rpa_test = URPA(mf_test)
    rpa_test.frozen = 1
    rpa_test.kernel()
    print('RPA e_tot, e_hf, e_corr = ', rpa_test.e_tot, rpa_test.e_hf, rpa_test.e_corr)
    assert abs(rpa_test.e_corr - -0.20755152016910608) < 1e-6
    assert abs(rpa_test.e_tot - -99.49067072777731) < 1e-6

    rpa_test = URPA(mf_test)
    rpa_test.grids_alg = 'minimax'
    rpa_test.frozen = 1
    rpa_test.kernel()
    print('RPA e_tot, e_hf, e_corr = ', rpa_test.e_tot, rpa_test.e_hf, rpa_test.e_corr)
    assert abs(rpa_test.e_corr - -0.20755152016910608) < 1e-6
    assert abs(rpa_test.e_tot - -99.49067072777731) < 1e-6

    print('passed tests!')
