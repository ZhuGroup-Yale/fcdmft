"""
Periodic spin-restricted eigenvalue self-consistent GW method based on the analytic continuation scheme.

References:
    Phys. Rev. B 76, 165106 (2007)
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
"""

from functools import reduce
import os
import scipy
import numpy as np

from pyscf import lib
from pyscf.lib import logger, temporary_env
from pyscf.pbc import df, dft, scf
from pyscf.pbc.lib import chkfile

from fcdmft.ac.grids import _get_scaled_legendre_roots
from fcdmft.ac.two_pole import TwoPoleAC
from fcdmft.ac.pade import PadeAC
from fcdmft.gw.pbc.krgw_ac import KRGWAC, _mo_energy_frozen, get_sigma, get_ef

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD


def kernel(gw):
    mf = gw._scf
    nkpts = gw.nkpts
    nmo = gw.nmo

    # set frozen orbitals
    gw.set_frozen_orbs()
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz
    kptlist = gw.kptlist
    if kptlist is None:
        gw.kptlist = kptlist = range(gw.nkpts)
    mo_energy = gw.mo_energy
    mo_energy_frz = _mo_energy_frozen(gw, gw.mo_energy)

    # grids for integration on imaginary axis
    gw.freqs, gw.wts = freqs, wts = _get_scaled_legendre_roots(gw.nw)

    # set up Fermi level
    ef = gw.ef = get_ef(kmf=mf, mo_energy=gw.mo_energy)

    # set up RHF object
    if isinstance(mf.with_df, df.GDF):
        rhf = scf.KRHF(gw.mol.copy(deep=True), gw.kpts, exxdiv=None).density_fit()
    elif isinstance(mf.with_df, df.RSDF):
        rhf = scf.KRHF(gw.mol.copy(deep=True), gw.kpts, exxdiv=None).rs_density_fit()
    if hasattr(mf, "sigma"):
        rhf = scf.addons.smearing_(rhf, sigma=mf.sigma, method=mf.smearing_method)
    rhf.with_df = gw.with_df
    rhf.mo_energy = np.array(gw.mo_energy, copy=True)
    rhf.mo_coeff = np.array(gw.mo_coeff, copy=True)
    rhf.mo_occ = np.array(gw.mo_occ, copy=True)
    if rank > 0:
        rhf.verbose = rhf.mol.verbose = 0

    # initialize DIIS
    gw_diis = lib.diis.DIIS(gw, gw.diis_file)
    gw_diis.space = gw.diis_space
    # get hcore and veff matrix
    with temporary_env(mf.mol, verbose=0):
        hcore = mf.get_hcore()
        dm = mf.make_rdm1()
    with temporary_env(rhf, verbose=0), temporary_env(rhf.with_df, verbose=0):
        veff = rhf.get_veff(dm_kpts=dm)
    for k in range(nkpts):
        hcore[k] = reduce(np.matmul, (mf.mo_coeff[k].T.conj(), hcore[k], mf.mo_coeff[k]))
        veff[k] = reduce(np.matmul, (mf.mo_coeff[k].T.conj(), veff[k], mf.mo_coeff[k]))

    # finite size correction for exchange self-energy
    if gw.fc:
        vk_corr = -2.0 / np.pi * (6.0 * np.pi**2 / gw.mol.vol / nkpts) ** (1.0 / 3.0)
        nocc_full = int(np.sum(gw._scf.mo_occ[0])) // 2
        for k in range(nkpts):
            for i in range(nocc_full):
                veff[k][i, i] = veff[k][i, i] + vk_corr
    ham_hf = hcore + veff


    cycle = 0
    while gw.converged is False and cycle < max(1, gw.max_cycle):
        # calculate self-energy on imaginary axis
        gw.sigmaI, gw.omega = sigmaI, omega = get_sigma(
            gw, freqs, wts, ef=ef, mo_energy=mo_energy_frz, orbs=orbs_frz, kptlist=kptlist, iw_cutoff=gw.ac_iw_cutoff,
            fullsigma=False)

        # analytic continuation
        if gw.ac == 'twopole':
            acobj = TwoPoleAC(list(range(nmo)), gw.nocc)
        elif gw.ac == 'pade':
            acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
        elif gw.ac == 'pes':
            raise NotImplementedError
        else:
            raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))

        acobj.ac_fit(sigmaI, omega, axis=-1)
        acobj.coeff = comm.bcast(acobj.coeff, root=0)

        # follow the section 5.1.1 Method evGW in 10.1021/acs.jctc.5b01238
        # for each pole the QP-equation is solved self-consistently
        # only iterative quasiparticle equation is implemented here
        mo_energy_old = np.array(mo_energy, copy=True)
        for ik, k in enumerate(kptlist):
            for ip, p in enumerate(orbs):

                def quasiparticle(omega):
                    sigmaR = acobj[ik, ip].ac_eval(omega)
                    return omega - (ham_hf[k, p, p] + sigmaR).real

                try:
                    mo_energy[k, p] = scipy.optimize.newton(
                        quasiparticle, mo_energy_old[k, p], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter
                    )
                except RuntimeError:
                    if rank == 0:
                        logger.warn(gw, 'QPE for k=%d orbital=%d not converged!', k, p)

        # update quasiparticle energy through DIIS
        if rank == 0:
            if cycle >= gw.diis_start_cycle:
                mo_energy = gw_diis.update(mo_energy)
        mo_energy = comm.bcast(mo_energy, root=0)

        # update attributes in the GW object
        gw.mo_energy = np.array(mo_energy, copy=True)
        ef = gw.ef = get_ef(mf, mo_energy=mo_energy)
        gw.acobj = acobj

        # update NON-FROZEN quantities
        mo_energy_frz = _mo_energy_frozen(gw, mo_energy)

        # check density matrix convergence
        diff = 0
        for k in range(nkpts):
            diff += abs(np.sum(1.0 / mo_energy[k] - 1.0 / mo_energy_old[k]))
        diff /= nkpts * nmo * nmo
        if diff < gw.conv_tol:
            gw.converged = True

        if rank == 0:
            logger.info(gw, 'EVGW cycle= %d  |delta G|= %-.4e', cycle + 1, diff)
        cycle += 1

        if rank == 0 and gw.verbose >= logger.DEBUG:
            with np.printoptions(threshold=len(mf.mo_energy[0])):
                for k in range(nkpts):
                    logger.debug(gw, 'cycle %d GW mo_energy @ k%d =\n%s', cycle + 1, k, mo_energy[k])

    if rank == 0 and gw.verbose >= logger.DEBUG:
        logger.debug(gw, '')
        with np.printoptions(threshold=len(mf.mo_energy[0])):
            for k in range(nkpts):
                logger.debug(gw, '  GW mo_energy @ k%d =\n%s', k, mo_energy[k])
        logger.warn(gw, 'GW QP energies may not be sorted from min to max')

    return


class KREVGW(KRGWAC):
    def __init__(self, mf, frozen=None):
        KRGWAC.__init__(self, mf, frozen=frozen)

        # options
        self.max_cycle = 30  # max cycle
        self.conv_tol = 1e-5  # convergence tolerance
        self.diis_space = 10  # DIIS space
        self.diis_start_cycle = 0  # DIIS start cycle
        self.diis_file = None  # DIIS file

        # results
        self.converged = False  # qsGW convergence

        return

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        log.info('GW nocc = %d, nvir = %d, nkpts = %d', nocc, nvir, nkpts)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', str(self.frozen))
        if self.kptlist is not None:
            log.info('k-point list = %s', str(self.kptlist))
        if self.orbs is not None:
            log.info('orbital list = %s', str(self.orbs))
        log.info('density-fitting for exchange = %s', self.vhf_df)
        log.info('finite size corrections = %s', self.fc)
        if self.fc_grid is not None:
            log.info('grids for finite size corrections = %s', self.fc_grid)
        log.info('broadening parameter = %.3e', self.eta)
        log.info('number of grids = %d', self.nw)
        log.info('analytic continuation method = %s', self.ac)
        log.info('imaginary frequency cutoff = %s', str(self.ac_iw_cutoff))
        if self.ac == 'pade':
            log.info('Pade points = %d', self.ac_pade_npts)
            log.info('Pade step ratio = %.3f', self.ac_pade_step_ratio)
        # evGW settings
        log.info('max cycle = %d', self.max_cycle)
        log.info('density matrix convergence tolerance = %.2e', self.conv_tol)
        log.info('DIIS space = %d', self.diis_space)
        log.info('DIIS start cycle = %d', self.diis_start_cycle)
        log.info('DIIS file = %s', self.diis_file)
        log.info('')
        return

    def _finalize(self):
        """Hook for dumping results and clearing up the object."""
        if self.converged:
            logger.note(self, 'EVGW converged.')
        else:
            logger.note(self, 'EVGW not converged.')
        return

    def kernel(self, orbs=None, kptlist=None):
        """Run evGW calculation.

        Parameters
        ----------
        orbs : list, optional
            orbital list to calculate self-energy, by default None
        kptlist : list, optional
            k-point list to calculate self-energy, by default None
        """
        if self.mo_energy is None:
            self.mo_energy = np.array(self._scf.mo_energy, copy=True)
        if self.mo_coeff is None:
            self.mo_coeff = np.array(self._scf.mo_coeff, copy=True)
        if self.mo_occ is None:
            self.mo_occ = np.array(self._scf.mo_occ, copy=True)

        self.orbs = orbs
        self.kptlist = kptlist

        if hasattr(self._scf, 'sigma'):
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0
            self.fc = False

        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (2 * nkpts * nmo**2 * naux) * 16 / 1e6
        mem_now = lib.current_memory()[0]
        if mem_incore + mem_now > 0.99 * self.max_memory:
            if rank == 0:
                logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError

        cput0 = (logger.process_clock(), logger.perf_counter())
        if rank == 0:
            self.dump_flags()
        kernel(self)

        if rank == 0:
            logger.timer(self, 'KREVGW', *cput0)
            self._finalize()
        return


if __name__ == '__main__':
    import os
    from pyscf.pbc import gto, dft, scf
    from pyscf.pbc.lib import chkfile

    # Test on diamond
    cell = gto.Cell()
    cell.build(
        unit='angstrom',
        a="""
                0.000000     1.783500     1.783500
                1.783500     0.000000     1.783500
                1.783500     1.783500     0.000000
            """,
        atom='C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
        dimension=3,
        max_memory=8000,
        verbose=5,
        pseudo='gth-pade',
        basis='gth-szv',
        precision=1e-10,
    )

    kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    gdf = df.RSDF(cell, kpts)
    gdf_fname = 'gdf_ints_311.h5'
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'diamond_311.chk'
    if os.path.isfile(chkfname):
        kmf = dft.KRKS(cell, kpts)
        kmf.xc = 'pbe'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = dft.KRKS(cell, kpts)
        kmf.xc = 'pbe'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    gw = KREVGW(kmf)
    gw.fc = False
    gw.kernel()
    assert abs(gw.mo_energy[0][3] - 0.63904658) < 1e-4
    assert abs(gw.mo_energy[0][4] - 1.00135539) < 1e-4
    assert abs(gw.mo_energy[1][3] - 0.54374428) < 1e-4
    assert abs(gw.mo_energy[1][4] - 1.13826106) < 1e-4

    # frozen core
    gw = KREVGW(kmf)
    gw.fc = False
    gw.frozen = 1
    gw.kernel()
    assert abs(gw.mo_energy[0][3] - 0.63970059) < 1e-4
    assert abs(gw.mo_energy[0][4] - 0.99731846) < 1e-4

    # Test on Na (metallic)
    cell = gto.Cell()
    cell.build(
        unit='angstrom',
        a="""
           -2.11250000000000   2.11250000000000   2.11250000000000
           2.11250000000000  -2.11250000000000   2.11250000000000
           2.11250000000000   2.11250000000000  -2.11250000000000
           """,
        atom="""
           Na   0.00000   0.00000   0.00000
           """,
        dimension=3,
        max_memory=126000,
        verbose=5,
        pseudo='gth-pade',
        basis='gth-dzvp-molopt-sr',
        precision=1e-10,
    )

    kpts = cell.make_kpts([2, 2, 1], scaled_center=[0, 0, 0])
    gdf = df.RSDF(cell, kpts)
    gdf_fname = 'gdf_ints_221.h5'
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'na_221.chk'
    if os.path.isfile(chkfname):
        kmf = dft.KRKS(cell, kpts)
        kmf = scf.addons.smearing_(kmf, sigma=5e-3, method='fermi')
        kmf.xc = 'lda'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = dft.KRKS(cell, kpts)
        kmf = scf.addons.smearing_(kmf, sigma=5e-3, method='fermi')
        kmf.xc = 'lda'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    gw = KREVGW(kmf)
    gw.conv_tol = 1e-4
    gw.kernel()

    # frozen core
    gw = KREVGW(kmf)
    gw.conv_tol = 1e-4
    gw.frozen = 1
    gw.kernel()

    if rank == 0:
        print('passed tests!')
