"""
Spin-restricted random phase approximation (direct RPA) with N^4 scaling.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14, 053020 (2012)
"""

import numpy as np
import scipy
import time

from pyscf import df, dft, lib, scf
from pyscf.ao2mo._ao2mo import nr_e2
from pyscf.lib import logger, temporary_env
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask

from fcdmft.gw.minimax.minimax_rpa import get_frequency_minimax_grid
from fcdmft.ac.grids import _get_scaled_legendre_roots
from fcdmft.gw.mol.gw_ac import get_rho_response, _mo_energy_without_core, _mo_without_core


def kernel(rpa, mo_energy, mo_coeff, Lpq=None, nw=None):
    """RPA correlation and total energy.

    Parameters
    ----------
    rpa : RPA
        rpa object
    mo_energy : double 1d array
        molecular orbital energies
    mo_coeff : double 2d array
        molecular orbital coefficients
    Lpq : double 3d array, optional
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
    if rpa.outcore:
        e_corr = get_rpa_ecorr_outcore(rpa, mo_coeff, freqs, wts)
    else:
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
    rpa : RPA
        rpa object
    Lpq : double 3d array
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
    nw = len(freqs)
    naux = Lpq.shape[0]
    mo_occ = rpa.mo_occ

    is_metal = hasattr(rpa._scf, 'sigma')

    e_corr = 0.0
    for w in range(nw):
        if is_metal:
            Pi = get_rho_response_metal(freqs[w], mo_energy, Lpq, mo_occ)
        else:
            Pi = get_rho_response(freqs[w], mo_energy, Lpq)
        ec_w = np.linalg.slogdet((np.eye(naux) - Pi))[1]
        ec_w += np.trace(Pi)
        e_corr += 1.0 / (2.0 * np.pi) * ec_w * wts[w]

    return e_corr


def get_rpa_ecorr_outcore(rpa, mo_coeff, freqs, wts):
    """Low-memory routine to compute RPA correlation energy.

    Parameters
    ----------
    rpa : RPA
        rpa object
    mo_coeff : double 2d array
        orbital coefficient
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
    nocc = rpa.nocc
    nw = len(freqs)
    nmo = rpa.nmo
    naux = rpa.with_df.get_naoaux()
    mo_occ = rpa.mo_occ

    is_metal = False
    mo_occ_1d = np.array(mo_occ).reshape(-1)
    if np.linalg.norm(np.abs(mo_occ_1d - 1.0) - 1.0) > 1e-5:
        is_metal = True

    e_corr = 0.0
    Pi = np.zeros(shape=[nw, naux, naux], dtype=np.double)
    if is_metal:
        idx_occ, idx_frac, idx_vir = get_idx_metal(mo_occ_1d)

        # Pi from frac to all orbs
        orb_start = idx_frac[0]
        orb_end = idx_frac[-1] + 1
        ijslice = (orb_start, orb_end, 0, nmo)
        Lpq = rpa.loop_ao2mo(mo_coeff=mo_coeff, ijslice=ijslice)
        for w in range(nw):
            eia = mo_energy[idx_frac][:, None] - mo_energy[None, :]
            fia = (mo_occ[idx_frac][:, None] - mo_occ[None, :]) / 2.0
            freqs_w = freqs[w]
            try:
                import numexpr as ne

                eia = ne.evaluate('4.0 * eia * fia / (freqs_w**2 + eia**2)')
                Pia = ne.evaluate('Lpq * eia')
            except ImportError:
                eia = 4.0 * eia * fia / (freqs_w**2 + eia**2)
                Pia = Lpq * eia
            # This line computes Pi[:,:,w] += einsum('Pia, Qia -> PQ', Pia, Lpq)
            Pi[w] += Pia.reshape(naux, -1) @ Lpq.reshape(naux, -1).T

        # Pi double counting from frac
        orb_start = idx_frac[0]
        orb_end = idx_frac[-1] + 1
        ijslice = (orb_start, orb_end, orb_start, orb_end)
        Lpq = rpa.loop_ao2mo(mo_coeff=mo_coeff, ijslice=ijslice)
        for w in range(nw):
            eia = mo_energy[idx_frac][:, None] - mo_energy[idx_frac][None, :]
            fia = (mo_occ[idx_frac][:, None] - mo_occ[idx_frac][None, :]) / 2.0
            freqs_w = freqs[w]
            try:
                import numexpr as ne

                eia = ne.evaluate('2.0 * eia * fia / (freqs_w**2 + eia**2)')
                Pia = ne.evaluate('Lpq * eia')
            except ImportError:
                eia = 2.0 * eia * fia / (freqs_w**2 + eia**2)
                Pia = Lpq * eia
            # This line computes Pi[:,:,w] -= einsum('Pia, Qia -> PQ', Pia, Lpq)
            Pi[w] -= Pia.reshape(naux, -1) @ Lpq.reshape(naux, -1).T

        # Pi from occ to vir orbs
        nocc_metal = len(idx_occ)
        nseg = nocc_metal // rpa.segsize + 1
        for i in range(nseg):
            orb_start = i * rpa.segsize
            orb_end = min((i + 1) * rpa.segsize, nocc_metal)
            if orb_end == orb_start:
                continue
            ijslice = (orb_start, orb_end, idx_vir[0], nmo)
            Lpq = rpa.loop_ao2mo(mo_coeff=mo_coeff, ijslice=ijslice)
            for w in range(nw):
                eia = mo_energy[orb_start:orb_end, None] - mo_energy[None, idx_vir[0] :]
                freqs_w = freqs[w]
                try:
                    import numexpr as ne

                    eia = ne.evaluate('4.0 * eia / (freqs_w**2 + eia**2)')
                    Pia = ne.evaluate('Lpq * eia')
                except ImportError:
                    eia = 4.0 * eia / (freqs_w**2 + eia**2)
                    Pia = Lpq * eia
                # This line computes Pi[:,:,w] += einsum('Pia, Qia -> PQ', Pia, Lpq)
                Pi[w] += Pia.reshape(naux, -1) @ Lpq.reshape(naux, -1).T
    else:
        nseg = nocc // rpa.segsize + 1
        for i in range(nseg):
            orb_start = i * rpa.segsize
            orb_end = min((i + 1) * rpa.segsize, nocc)
            if orb_end == orb_start:
                continue
            ijslice = (orb_start, orb_end, nocc, nmo)
            Lpq = rpa.loop_ao2mo(mo_coeff=mo_coeff, ijslice=ijslice)
            for w in range(nw):
                eia = mo_energy[orb_start:orb_end, None] - mo_energy[None, nocc:]
                freqs_w = freqs[w]
                try:
                    import numexpr as ne

                    eia = ne.evaluate('4.0 * eia / (freqs_w**2 + eia**2)')
                    Pia = ne.evaluate('Lpq * eia')
                except ImportError:
                    eia = 4.0 * eia / (freqs_w**2 + eia**2)
                    Pia = Lpq * eia
                # This line computes Pi[:,:,w] += einsum('Pia, Qia -> PQ', Pia, Lpq)
                Pi[w] += Pia.reshape(naux, -1) @ Lpq.reshape(naux, -1).T

    for w in range(nw):
        ec_w = np.linalg.slogdet((np.eye(naux) - Pi[w]))[1]
        ec_w += np.trace(Pi[w])
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
    idx_occ = np.where(mo_occ > 2.0 - threshold)[0]
    idx_vir = np.where(mo_occ < threshold)[0]
    idx_frac = list(range(idx_occ[-1] + 1, idx_vir[0]))

    return idx_occ, idx_frac, idx_vir


def get_rho_response_metal(omega, mo_energy, Lpq, mo_occ, out=None):
    """Compute density response function in auxiliary basis at freq iw.

    Parameters
    ----------
    omega : double
        imaginary part of a frequency point
    mo_energy : double 1d array
        orbital energy
    Lpq : double 3d array
        three-center density-fitting matrix in MO
    mo_occ : double 1d array
        occupation number
    out : double ndarray, optional
        a location into which the result is stored, by default None

    Returns
    -------
    Pi : double 2d array
        density-density response function in auxiliary basis at freq iw
    """
    idx_occ, idx_frac, _ = get_idx_metal(mo_occ)
    nocc = len(idx_occ)
    nfrac = len(idx_frac)
    naux = Lpq.shape[0]
    eia = mo_energy[: nocc + nfrac, None] - mo_energy[None, nocc:]
    fia = (mo_occ[: nocc + nfrac, None] - mo_occ[None, nocc:]) / 2.0
    # Add a factor of 0.5 for double counting
    fia[nocc:, :nfrac] *= 0.5

    try:
        import numexpr as ne

        eia = ne.evaluate('4.0 * eia * fia / (omega**2 + eia**2)')
        Pia = ne.evaluate('Lpq * eia')
    except ImportError:
        # plain numpy, numexpr not available
        eia = 4.0 * eia * fia / (omega**2 + eia**2)
        Pia = Lpq * eia

    # This line computes Pi = einsum('Pia, Qia -> PQ', Pia, Lpq)
    Pi = np.matmul(Pia.reshape(naux, -1), Lpq.reshape(naux, -1).T, out=out)
    return Pi


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

        dm = rpa._scf.make_rdm1()
        with temporary_env(rpa.mol, verbose=0):
            rhf = scf.RHF(rpa.mol)
            rhf.verbose = 0
            vk = rhf.get_k(rhf.mol, dm)
        return -0.25 * lib.einsum('ij,ji', dm, vk)

    if getattr(rpa, 'with_df', None) is None:
        rpa.initialize_df(auxbasis=rpa.auxbasis)

    ex = 0.0
    if hasattr(rpa._scf, 'sigma'):
        idx_occ, idx_frac, _ = get_idx_metal(mo_occ)
        nocc = len(idx_occ) + len(idx_frac)
        Lij = rpa.loop_ao2mo(mo_coeff=mo_coeff, ijslice=(0, nocc, 0, nocc))

        if acfd is True:
            if correction_only is True:
                mo_occ_ij = np.minimum(mo_occ[:nocc, None], mo_occ[None, :nocc]) / 2.0
                mo_occ_ij -= mo_occ[:nocc, None] * mo_occ[None, :nocc] / 4.0
            else:
                # The following line is equivalent to the frequency integration in equation 12 in
                # doi.org/10.1103/PhysRevB.81.115126
                # TODO: add a detailed note
                eij = mo_energy[:nocc, None] - mo_energy[None, :nocc]
                integrand = np.zeros((nocc, nocc), dtype=np.double)
                integrand[eij > 1e-6] = 1
                integrand[eij < -1e-6] = -1
                mo_occ_ij = 1.0 - integrand
                # spin-restricted mo_occ should be divided by 2
                mo_occ_ij = mo_occ_ij * mo_occ[:nocc, None] / 2.0
        else:
                mo_occ_ij = mo_occ[:nocc, None] * mo_occ[None, :nocc] / 4.0
        Lij_occ = Lij * mo_occ_ij[None]
        # ex -= np.einsum('Lij,Lij->', Lij_occ.reshape(-1, nocc, nocc), Lij.reshape(-1, nocc, nocc))
        ex -= scipy.linalg.blas.ddot(Lij_occ.ravel(), Lij.ravel())
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


class RPA(lib.StreamObject):
    def __init__(self, mf, frozen=None, auxbasis=None):
        self.mol = mf.mol  # mol object
        self._scf = mf  # mean-field object
        self.verbose = self.mol.verbose  # verbose level
        self.stdout = self.mol.stdout  # standard output
        self.max_memory = mf.max_memory  # max memory in MB
        self.auxbasis = auxbasis  # auxiliary basis set for density fitting

        # options
        self.frozen = frozen  # frozen orbital options
        self.grids_alg = 'legendre'  # algorithm to generate grids
        self.acfd_exx = False  # calculate ACFD exchange energy
        self.outcore = False  # low-memory routine
        self.segsize = 60  # number of orbitals in one segment for outcore

        # don't modify the following attributes, they are not input options
        self._nocc = None  # number of occupied orbitals
        self._nmo = None  # number of orbitals (exclude frozen orbitals)
        self.mo_energy = np.array(mf.mo_energy, copy=True)  # orbital energy
        self.mo_coeff = np.array(mf.mo_coeff, copy=True)  # orbital coefficient
        self.mo_occ = np.array(mf.mo_occ, copy=True)  # occupation number
        self.e_corr = None  # correlation energy
        self.e_hf = None  # Hartree-Fock energy
        self.e_tot = None  # total energy

        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('RPA nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', str(self.frozen))
        log.info('grid type = %s', self.grids_alg)
        log.info('ACFD exchange energy = %s', self.acfd_exx)
        log.info('outcore mode = %s', self.outcore)
        if self.outcore is True:
            log.info('outcore segment size = %d', self.segsize)
        log.info('')
        return

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

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, nw=None):
        """RPA correlation and total energy

        Calculated total energy, HF energy and RPA correlation energy
        are stored in self.e_tot, self.e_hf, self.e_corr

        Parameters
        ----------
        mo_energy : double 1d array
            molecular orbital energies
        mo_coeff : double 2d ndarray
            molecular orbital coefficients
        Lpq : double 3d array, optional
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

    def initialize_df(self, auxbasis=None):
        """Initialize density fitting.

        Parameters
        ----------
        auxbasis : str, optional
            name of auxiliary basis set, by default None
        """
        if getattr(self._scf, 'with_df', None):
            self.with_df = self._scf.with_df
        else:
            self.with_df = df.DF(self._scf.mol)
            if auxbasis is not None:
                self.with_df.auxbasis = auxbasis
            else:
                try:
                    self.with_df.auxbasis = df.make_auxbasis(self._scf.mol, mp2fit=True)
                except RuntimeError:
                    self.with_df.auxbasis = df.make_auxbasis(self._scf.mol, mp2fit=False)
        self._keys.update(['with_df'])
        return

    def ao2mo(self, mo_coeff=None, mo_occ=None):
        """Transform density-fitting integral from AO to MO.

        Parameters
        ----------
        mo_coeff : double 2d array, optional
            coefficient from AO to MO, by default None
        mo_occ : double 1d array, optional
            occupation number, by default None

        Returns
        -------
        Lpq : double 3d array
            three-center density-fitting matrix in MO
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        nmo = mo_coeff.shape[1]
        if mo_occ is None:
            nocc = self.nocc
        else:
            nocc = len(np.where(mo_occ > 0.0)[0])
        naux = self.with_df.get_naoaux()
        mem_incore = (2 * nmo**2 * naux) * 8 / 1e6
        mem_now = lib.current_memory()[0]

        mo = np.asarray(mo_coeff, order='F')
        ijslice = (0, nocc, nocc, nmo)
        Lpq = None
        if (mem_incore + mem_now < 0.99 * self.max_memory) or self.mol.incore_anyway:
            Lpq = nr_e2(self.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
            return Lpq.reshape(naux, nocc, nmo - nocc)
        else:
            logger.warn(self, 'Memory is not enough for incore algorithm! Changing to mol.rpa.loop_ao2mo.')
            return self.loop_ao2mo(mo_coeff, ijslice)

    def loop_ao2mo(self, mo_coeff=None, ijslice=None):
        """Transform density-fitting integral from AO to MO by block.

        Parameters
        ----------
        mo_coeff : double 2d array, optional
            coefficient from AO to MO, by default None
        ijslice : tuple, optional
            tuples for (1st idx start, 1st idx end, 2nd idx start, 2nd idx end), by default None

        Returns
        -------
        eri_3d : double 3d array
            three-center density-fitting matrix in MO in a block
        """
        nmo = self.nmo
        naux = self.with_df.get_naoaux()

        mo = np.asarray(mo_coeff, order='F')
        if ijslice is None:
            ijslice = (0, nmo, 0, nmo)
        nislice = ijslice[1] - ijslice[0]
        njslice = ijslice[3] - ijslice[2]

        with_df = self.with_df
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, (self.max_memory - mem_now) * 0.3)
        blksize = int(min(naux, max(with_df.blockdim, (max_memory * 1e6 / 8) / (nmo * nmo))))
        eri_3d = []
        for eri1 in with_df.loop(blksize=blksize):
            Lpq = None
            Lpq = nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
            eri_3d.append(Lpq)
        del eri1
        del Lpq

        eri_3d = np.vstack(eri_3d).reshape(naux, nislice, njslice)
        return eri_3d

    def get_grids(self, alg=None, nw=None, mo_energy=None):
        """Generate grids for integration.

        Parameters
        ----------
        alg : str, optional
            algorithm for generating grids, by default None
        nw : int, optional
            number of grids, by default None
        mo_energy : double 1d array, optional
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
            E_min, E_max = mo_energy[self.nocc] - mo_energy[self.nocc - 1], mo_energy[-1] - mo_energy[0]
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
            rhf = scf.RHF(self.mol)
            rhf.verbose = 0
            if hasattr(self._scf, 'sigma'):
                rhf = scf.addons.smearing_(rhf, sigma=self._scf.sigma, method=self._scf.smearing_method)
            if hasattr(self._scf, 'with_df'):
                rhf.with_df = self._scf.with_df
            with temporary_env(self.with_df, verbose=0), temporary_env(rhf.mol, verbose=0):
                dm = self._scf.make_rdm1()
                e_1e = lib.einsum('ij,ji', dm, rhf.get_hcore())
                e_j = 0.5 * lib.einsum('ij,ji', dm, rhf.get_j(rhf.mol, dm))
                e_x = self.get_rpa_exx(acfd=self.acfd_exx, correction_only=False)
                e_nuc = self._scf.energy_nuc()
                e_hf = e_1e + e_j + e_x + e_nuc
        return e_hf

    def get_rpa_exx(self, acfd=False, correction_only=False):
        """Calculate RPA exchange energy.

        Parameters
        ----------
        acfd : bool
            whether to include ACFD correction, by default False
        correction_only : bool
            only return the correction term, by default False

        Returns
        -------
        ex : double
            exchange energy
        """
        ex = get_rpa_exx(self, acfd=acfd, correction_only=correction_only)
        return ex


if __name__ == '__main__':
    from pyscf import gto, dft, scf

    mol_test = gto.Mole()
    mol_test.verbose = 4
    mol_test.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol_test.basis = 'def2-svp'
    mol_test.build()

    mf_test = dft.RKS(mol_test)
    mf_test.xc = 'pbe'
    mf_test.kernel()

    rpa_test = RPA(mf_test)
    rpa_test.kernel()
    print('RPA e_tot, e_hf, e_corr = ', rpa_test.e_tot, rpa_test.e_hf, rpa_test.e_corr)
    assert abs(rpa_test.e_corr - -0.30783004035780076) < 1e-6
    assert abs(rpa_test.e_tot - -76.26428191794182) < 1e-6

    # outcore
    rpa_test = RPA(mf_test)
    rpa_test.outcore = True
    rpa_test.segsize = 2
    rpa_test.kernel()
    print('RPA e_tot, e_hf, e_corr = ', rpa_test.e_tot, rpa_test.e_hf, rpa_test.e_corr)
    assert abs(rpa_test.e_corr - -0.30783004035780076) < 1e-6
    assert abs(rpa_test.e_tot - -76.26428191794182) < 1e-6

    # frozen orbital
    rpa_test = RPA(mf_test)
    rpa_test.frozen = 1
    rpa_test.kernel()
    print('RPA e_tot, e_hf, e_corr = ', rpa_test.e_tot, rpa_test.e_hf, rpa_test.e_corr)
    assert abs(rpa_test.e_corr - -0.3046092046689639) < 1e-6
    assert abs(rpa_test.e_tot - -76.26106101984152) < 1e-6

    # frozen orbital, outcore
    rpa = RPA(mf_test)
    rpa.frozen = 1
    rpa.outcore = True
    rpa.segsize = 2
    rpa.kernel()
    print('RPA e_tot, e_hf, e_corr = ', rpa.e_tot, rpa.e_hf, rpa.e_corr)
    assert abs(rpa.e_corr - -0.3046092046689639) < 1e-6
    assert abs(rpa.e_tot - -76.26106101984152) < 1e-6

    rpa_test = RPA(mf_test)
    rpa_test.frozen = 1
    rpa_test.grids_alg = 'minimax'
    rpa_test.kernel()
    print('RPA e_tot, e_hf, e_corr = ', rpa_test.e_tot, rpa_test.e_hf, rpa_test.e_corr)
    assert abs(rpa_test.e_corr - -0.3046092046689639) < 1e-6
    assert abs(rpa_test.e_tot - -76.26106101984152) < 1e-6

    print('passed tests!')
