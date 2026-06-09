"""
Periodic spin-restricted quasiparticle self-consistent GW method based on the analytic continuation scheme.

References:
    J. Lei and T. Zhu, J. Chem. Phys. 157, 214114 (2022)
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    Phys. Rev. B 76, 165106 (2007)
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
"""

from functools import reduce
import os
import scipy
import numpy as np
import h5py
import scipy.linalg

from pyscf import lib
from pyscf.lib import logger, einsum, temporary_env
from pyscf.pbc import df, dft, scf
from pyscf.pbc.lib import chkfile
from pyscf import scf as mol_scf

from fcdmft.ac import load_ac
from fcdmft.ac.grids import _get_scaled_legendre_roots
from fcdmft.ac.two_pole import TwoPoleAC
from fcdmft.ac.pade import PadeAC
from fcdmft.gw.pbc.krgw_ac import KRGWAC, _mo_frozen, _mo_energy_frozen, _mo_occ_frozen, get_sigma, get_g0_k, get_ef, \
    set_frozen_orbs

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD


def kernel(gw):
    mf = gw._scf
    eta = gw.eta
    nkpts = gw.nkpts
    nmo = gw.nmo
    nocc_full = int(np.sum(gw._scf.mo_occ[0])) // 2

    # set frozen orbital list
    gw.set_frozen_orbs()
    orbs_frz = gw.orbs_frz

    if gw.load_chkfile:
        if rank == 0:
            logger.info(gw, 'Load chkfile from %s, QSGW previous cycle: ', gw.chkfile)
        with h5py.File(gw.chkfile, 'r') as fh5:
            gw.mo_energy = mo_energy_full = np.array(fh5['gw/mo_energy'])
            gw.mo_coeff = mo_coeff_full = np.array(fh5['gw/mo_coeff'])
            gw.mo_occ = mo_occ_full = np.array(fh5['gw/mo_occ'])
    else:
        mo_energy_full = np.array(gw.mo_energy, copy=True)
        mo_coeff_full = np.array(gw.mo_coeff, copy=True)
        mo_occ_full = np.array(gw.mo_occ, copy=True)

    mo_energy_frz = _mo_energy_frozen(gw, mo_energy_full)
    mo_coeff_frz = _mo_frozen(gw, mo_coeff_full)
    mo_occ_frz = _mo_occ_frozen(gw, mo_occ_full)

    # grids for integration on imaginary axis
    gw.freqs, gw.wts = freqs, wts = _get_scaled_legendre_roots(gw.nw)

    # set up Fermi level
    ef = gw.ef = get_ef(kmf=mf, mo_energy=mo_energy_full)

    # get hcore and overlap matrix
    with temporary_env(mf.mol, verbose=0):
        hcore = mf.get_hcore()
        ovlp = mf.get_ovlp()

    # set up RHF object
    if isinstance(mf.with_df, df.GDF):
        rhf = scf.KRHF(gw.mol.copy(deep=True), gw.kpts, exxdiv=None).density_fit()
    elif isinstance(mf.with_df, df.RSDF):
        rhf = scf.KRHF(gw.mol.copy(deep=True), gw.kpts, exxdiv=None).rs_density_fit()
    if hasattr(mf, 'sigma'):
        rhf = scf.addons.smearing_(rhf, sigma=mf.sigma, method=mf.smearing_method)
    rhf.with_df = gw.with_df
    rhf.mo_energy = np.array(mo_energy_full, copy=True)
    rhf.mo_coeff = np.array(mo_coeff_full, copy=True)
    rhf.mo_occ = np.array(mo_occ_full, copy=True)
    if rank > 0:
        rhf.verbose = rhf.mol.verbose = 0
    dm_iter = np.array(rhf.make_rdm1(), copy=True)
    dm_old = dm_iter.copy()

    # initialize DIIS
    gw_diis = mol_scf.diis.DIIS(gw, gw.diis_file)
    gw_diis.space = gw.diis_space

    cycle = 0
    while gw.converged is False and cycle < max(1, gw.max_cycle):
        # calculate self-energy on imaginary axis
        gw.sigmaI, gw.omega = sigmaI, omega = get_sigma(
            gw, freqs, wts, ef=ef, mo_energy=mo_energy_frz, orbs=orbs_frz, mo_coeff=mo_coeff_frz, mo_occ=mo_occ_frz,
            iw_cutoff=gw.ac_iw_cutoff, fullsigma=True)

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

        # real-axis sigma
        mode = gw.mode.strip().lower()
        sigma = np.zeros(shape=[nkpts, nmo, nmo], dtype=np.complex128)
        for k in range(nkpts):
            for ip, p in enumerate(orbs_frz):
                for iq, q in enumerate(orbs_frz):
                    if mode == 'b':
                        if p == q:
                            sigma[k, p, q] += 0.5 * acobj[k, ip, iq].ac_eval(mo_energy_frz[k][p] + 1j * eta)
                            sigma[k, p, q] += 0.5 * acobj[k, ip, iq].ac_eval(mo_energy_frz[k][p] + 1j * eta).conj()
                        else:
                            sigma[k, p, q] += 0.5 * acobj[k, ip, iq].ac_eval(ef + 1j * eta)
                            sigma[k, p, q] += 0.5 * acobj[k, iq, ip].ac_eval(ef + 1j * eta).conj()
                    elif mode == 'a':
                        sigma[k, p, q] += 0.25 * acobj[k, ip, iq].ac_eval(mo_energy_frz[k][p] + 1j * eta)
                        sigma[k, p, q] += 0.25 * acobj[k, iq, ip].ac_eval(mo_energy_frz[k][p] + 1j * eta).conj()
                        sigma[k, p, q] += 0.25 * acobj[k, ip, iq].ac_eval(mo_energy_frz[k][q] + 1j * eta)
                        sigma[k, p, q] += 0.25 * acobj[k, iq, ip].ac_eval(mo_energy_frz[k][q] + 1j * eta).conj()
                    elif mode == 'c':
                        sigma[k, p, q] += 0.5 * acobj[k, ip, iq].ac_eval(ef + 1j * eta)
                        sigma[k, p, q] += 0.5 * acobj[k, iq, ip].ac_eval(ef + 1j * eta).conj()
                    else:
                        raise ValueError("Unknown QSGW mode %s" % gw.mode)

        # obtain static correlation self-energy in AO basis
        vsig = np.zeros_like(dm_iter, dtype=np.result_type(dm_iter, sigma))
        for k in range(nkpts):
            CS = np.matmul(mo_coeff_frz[k].T.conj(), ovlp[k])
            vsig[k] = reduce(np.matmul, (CS.T.conj(), sigma[k], CS))
        gw.vsig = vsig

        # update veff
        with temporary_env(rhf, verbose=0), temporary_env(rhf.with_df, verbose=0):
            veff = rhf.get_veff(dm_kpts=dm_iter)

        # finite size correction for exchange self-energy
        if gw.fc:
            vk_corr = -2.0 / np.pi * (6.0 * np.pi**2 / gw.mol.vol / nkpts) ** (1.0 / 3.0)
            for k in range(nkpts):
                veff[k] = reduce(np.matmul, (mo_coeff_full[k].T.conj(), veff[k], mo_coeff_full[k]))

            for k in range(nkpts):
                for i in range(nocc_full):
                    veff[k][i, i] = veff[k][i, i] + vk_corr

            for k in range(nkpts):
                CS = np.matmul(mo_coeff_full[k].T.conj(), ovlp[k])
                veff[k] = reduce(np.matmul, (CS.T.conj(), veff[k], CS))
        gw.veff = veff

        # complete Hamiltonian through DIIS
        ham = hcore + veff + vsig
        mo_energy_full = np.zeros_like(mf.mo_energy)
        mo_coeff_full = np.zeros_like(mf.mo_coeff)
        if rank == 0:
            if cycle >= gw.diis_start_cycle:
                ham = gw_diis.update(ovlp, dm_iter, ham)
            # diagonalize
            for k in range(nkpts):
                mo_energy_full[k], mo_coeff_full[k] = scipy.linalg.eigh(ham[k], ovlp[k])
        comm.Barrier()
        mo_energy_full = comm.bcast(mo_energy_full, root=0)
        mo_coeff_full = comm.bcast(mo_coeff_full, root=0)
        ham = comm.bcast(ham, root=0)

        # update QSGW mean-field object
        rhf.mo_energy = np.array(mo_energy_full, copy=True)
        rhf.mo_coeff = np.array(mo_coeff_full, copy=True)
        mo_occ_full = rhf.get_occ(mo_energy_kpts=mo_energy_full, mo_coeff_kpts=mo_coeff_full)
        rhf.mo_occ = np.array(mo_occ_full, copy=True)

        # update attributes in the GW object
        gw.mo_energy = np.array(mo_energy_full, copy=True)
        gw.mo_coeff = np.array(mo_coeff_full, copy=True)
        gw.mo_occ = np.array(mo_occ_full, copy=True)
        ef = gw.ef = get_ef(mf, mo_energy=mo_energy_full)
        gw.acobj = acobj

        # update NON-FROZEN quantities
        mo_energy_frz = _mo_energy_frozen(gw, mo_energy_full)
        mo_coeff_frz = _mo_frozen(gw, mo_coeff_full)
        mo_occ_frz = _mo_occ_frozen(gw, mo_occ_full)

        # check density matrix convergence
        if cycle == 0:
            dm_old2 = dm_iter
        else:
            dm_old2 = dm_old
        dm_old = dm_iter
        # density matrix from updated QSGW density
        dm_iter = rhf.make_rdm1()
        norm_dm = np.linalg.norm(dm_iter - dm_old) / (nmo * nkpts)
        norm_dm2 = np.linalg.norm(dm_old - dm_old2) / (nmo * nkpts)
        if norm_dm < gw.conv_tol and norm_dm2 < gw.conv_tol and cycle > 0:
            gw.converged = True

        if rank == 0:
            logger.info(gw, 'QSGW cycle= %d  |ddm|= %4.3g', cycle + 1, norm_dm)
        cycle += 1

        if gw.chkfile and rank == 0:
            gw.dump_chk()
        comm.Barrier()

    if gw.writefile > 0 and rank == 0:
        with h5py.File(name='vxc.h5', mode='w') as feri:
            feri['hcore'] = np.asarray(hcore)
            feri['veff'] = np.asarray(gw.veff)
            feri['vsig'] = np.asarray(gw.vsig)
            feri['mo_energy'] = np.asarray(gw.mo_energy)
            feri['mo_coeff'] = np.asarray(gw.mo_coeff)
            feri['mo_occ'] = np.asarray(gw.mo_occ)

        acobj.save('ac_coeff.h5')

    return


def make_gf(gw, omega, eta):
    """Get qsGW Green's function.
    G = G0 + G0 (sigma - vc) G, where G0 is non-interacting Green's function evaluated at quasiparticle energy,
    vc is the static correlation self-energy.
    TODO: this definition might be incorrect, because it doesn't consider the change in hcore and J.

    Parameters
    ----------
    gw : KRQSGW
        GW object, provides attributes: mo_energy, mo_coeff, ef, ac_coeff, omega_fit, vsig
    omega : double or complex array
        frequency grids
    eta : double
        broadening parameter

    Returns
    -------
    gf : complex ndarray
        qsGW Green's function
    """
    if gw.ac_coeff is None:
        if gw.chkfile:
            with h5py.File(gw.chkfile, 'r') as fh5:
                gw.mo_energy = np.array(fh5['gw/mo_energy'])
                gw.mo_coeff = np.array(fh5['gw/mo_coeff'])
                gw.ef = np.array(fh5['gw/ef'])
                gw.vsig = np.array(fh5['gw/vsig'])

            gw.acobj = load_ac('ac_coeff.h5')
        else:
            raise NotImplementedError

    mo_energy = gw.mo_energy
    mo_coeff = gw.mo_coeff
    acobj = gw.acobj

    nomega = len(omega)
    nkpts = gw.nkpts
    nmo = gw.nmo
    orbs = range(nmo)

    sigma = np.zeros(shape=[nkpts, nmo, nmo, nomega], dtype=np.complex128)
    for k in range(nkpts):
        for ip, p in enumerate(orbs):
            for iq, q in enumerate(orbs):
                sigma[k, p, q] = acobj[k, ip, iq].ac_eval(omega + 1j * eta)

    # QSGW static correlation self-energy
    v_c = np.zeros_like(gw.vsig)
    for k in range(nkpts):
        v_c[k] = reduce(np.matmul, (mo_coeff[k].T.conj(), gw.vsig[k], mo_coeff[k]))

    gf0 = get_g0_k(omega, np.asarray(mo_energy), eta)
    gf = np.zeros_like(gf0)
    for k in range(nkpts):
        for iw in range(nomega):
            gf[k, :, :, iw] = np.linalg.inv(np.linalg.inv(gf0[k, :, :, iw]) - (sigma[k, :, :, iw] - v_c[k]))

    return gf


def make_rdm1_dyson(gw, ao_repr=False):
    """Get GW density matrix from Green's function G(it=0).
    G is from Dyson equation G = G0 + G0 Sigma G

    Parameters
    ----------
    gw : KRQSGW
        GW object, provides attributes: mo_energy, mo_coeff, vsig, sigmaI, _scf, mol
    ao_repr : bool, optional
        return density matrix in AO, by default False

    Returns
    -------
    rdm1 : double ndarray
        density matrix
    """
    assert gw.sigmaI is not None
    assert gw.rdm is True and gw.fullsigma is True
    assert gw.frozen is None or gw.frozen == 0
    sigmaI = gw.sigmaI[:, :, :, 1:]
    freqs = 1j * gw.freqs
    wts = gw.wts
    nmo = gw.nmo
    nkpts = gw.nkpts
    if len(gw.orbs) != nmo:
        sigma = np.zeros(shape=[nkpts, nmo, nmo, len(freqs)], dtype=sigmaI.dtype)
        for k in range(nkpts):
            for ia, a in enumerate(gw.orbs):
                for ib, b in enumerate(gw.orbs):
                    sigma[k, a, b, :] = sigmaI[k, ia, ib, :]
    else:
        sigma = sigmaI

    # QSGW static correlation self-energy
    v_c = np.zeros_like(gw.vsig)
    for k in range(nkpts):
        v_c[k] = reduce(np.matmul, (gw.mo_coeff[k].T.conj(), gw.vsig[k], gw.mo_coeff[k]))

    # Compute GW Green's function on imag freq
    gf0 = get_g0_k(freqs, np.asarray(gw.mo_energy) - gw.ef, eta=0)
    gf = np.zeros_like(gf0)
    for k in range(nkpts):
        for iw in range(len(freqs)):
            gf[k, :, :, iw] = np.linalg.inv(np.linalg.inv(gf0[k, :, :, iw]) - (sigma[k, :, :, iw] - v_c[k]))

    # GW density matrix
    rdm1 = np.zeros(shape=[nkpts, nmo, nmo], dtype=gf.dtype)
    for k in range(nkpts):
        rdm1[k] = 2.0 / np.pi * einsum("ijw,w->ij", gf[k], wts).real + np.eye(nmo)

    # GW density matrix in original MO basis
    ovlp = gw._scf.get_ovlp()
    for k in range(nkpts):
        CSC = reduce(np.matmul, (gw._scf.mo_coeff[k].T.conj(), ovlp[k], gw.mo_coeff[k]))
        rdm1[k] = reduce(np.matmul, (CSC, rdm1[k], CSC.T.conj()))
        if rank == 0:
            logger.info(gw, "GW particle number @ k%d = %s", k, np.trace(rdm1[k]).real)

    if ao_repr is True:
        for k in range(nkpts):
            CS = np.matmul(ovlp[k], gw._scf.mo_coeff[k])
            rdm1[k] = reduce(np.matmul, (CS, rdm1[k], CS.conj().T))

    return rdm1


class KRQSGW(KRGWAC):
    def __init__(self, mf, frozen=None):
        KRGWAC.__init__(self, mf, frozen=frozen)

        # options
        self.mode = 'b'  # mode for off-diagonal self-energy, mode a, b in PRB 76, 165106, c for all elements at ef
        self.max_cycle = 20  # max cycle
        self.conv_tol = 1e-5  # convergence tolerance for density matrix
        self.diis_space = 10  # DIIS space
        self.diis_start_cycle = 0  # DIIS start cycle
        self.diis_file = None  # DIIS file
        self.chkfile = None  # check point file
        self.load_chkfile = False  # load check point file

        # results
        self.converged = False  # qsGW convergence
        self.vsig = None  # static correlation self-energy in AO space
        self.veff = None  # Hartree-Fock potential with qsGW density in AO space

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
        log.info('GW density matrix = %s', self.rdm)
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
        # qsGW settings
        log.info('off-diagonal self-energy mode = %s', self.mode)
        log.info('max cycle = %d', self.max_cycle)
        log.info('density matrix convergence tolerance = %.2e', self.conv_tol)
        log.info('DIIS space = %d', self.diis_space)
        log.info('DIIS start cycle = %d', self.diis_start_cycle)
        log.info('DIIS file = %s', self.diis_file)
        log.info('load chkfile = %s', self.load_chkfile)
        if self.load_chkfile:
            log.info('chkfile path = %s', self.chkfile)
        log.info('')
        return

    def _finalize(self):
        """Hook for dumping results and clearing up the object."""
        if self.converged:
            logger.note(self, 'QSGW converged.')
        else:
            logger.note(self, 'QSGW not converged.')
        return

    def dump_chk(self):
        """Dump qsGW check files to disk."""
        if self.chkfile:
            with h5py.File(self.chkfile, 'w') as fh5:
                fh5['gw/mo_energy'] = self.mo_energy
                fh5['gw/mo_coeff'] = self.mo_coeff
                fh5['gw/mo_occ'] = self.mo_occ
                fh5['gw/ef'] = self.ef
                fh5['gw/veff'] = self.veff
                fh5['gw/vsig'] = self.vsig

            self.acobj.save('ac_coeff.h5')
        return

    def kernel(self):
        """Run qsGW calculation."""
        if self.mo_energy is None:
            self.mo_energy = np.array(self._scf.mo_energy, copy=True)
        if self.mo_coeff is None:
            self.mo_coeff = np.array(self._scf.mo_coeff, copy=True)
        if self.mo_occ is None:
            self.mo_occ = np.array(self._scf.mo_occ, copy=True)

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
            logger.timer(self, 'KRQSGW', *cput0)
            self._finalize()
        return

    set_frozen_orbs = set_frozen_orbs
    make_rdm1 = make_rdm1_dyson
    make_gf = make_gf


if __name__ == '__main__':
    import os
    from pyscf.pbc import gto, dft, scf
    from pyscf.pbc.lib import chkfile

    # This test takes a few minutes
    # Test on diamond
    cell_test = gto.Cell()
    cell_test.build(
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

    kpts_test = cell_test.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    gdf_test = df.RSDF(cell_test, kpts_test)
    gdf_fname_test = 'gdf_ints_311.h5'
    gdf_test._cderi_to_save = gdf_fname_test
    if not os.path.isfile(gdf_fname_test):
        gdf_test.build()

    chkfname_test = 'diamond_311.chk'
    if os.path.isfile(chkfname_test):
        kmf_test = dft.KRKS(cell_test, kpts_test).rs_density_fit()
        kmf_test.xc = 'pbe'
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        data_test = chkfile.load(chkfname_test, 'scf')
        kmf_test.__dict__.update(data_test)
    else:
        kmf_test = dft.KRKS(cell_test, kpts_test).rs_density_fit()
        kmf_test.xc = 'pbe'
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        kmf_test.conv_tol = 1e-12
        kmf_test.chkfile = chkfname_test
        kmf_test.kernel()

    gw_test = KRQSGW(kmf_test)
    gw_test.fc = False
    gw_test.kernel()
    assert abs(gw_test.mo_energy[0][3] - 0.71535002) < 1e-4
    assert abs(gw_test.mo_energy[0][4] - 0.99042025) < 1e-4
    assert abs(gw_test.mo_energy[1][3] - 0.53587079) < 1e-4
    assert abs(gw_test.mo_energy[1][4] - 1.14822789) < 1e-4

    gw_test = KRQSGW(kmf_test)
    gw_test.fc = True
    gw_test.kernel()
    assert abs(gw_test.mo_energy[0][3] - 0.43221512) < 1e-4
    assert abs(gw_test.mo_energy[0][4] - 0.8736619) < 1e-4
    assert abs(gw_test.mo_energy[1][3] - 0.23905907) < 1e-4
    assert abs(gw_test.mo_energy[1][4] - 1.0414254) < 1e-4

    gw_test = KRQSGW(kmf_test)
    gw_test.fc = True
    # frozen core
    gw_test.frozen = 1
    gw_test.kernel()
    assert abs(gw_test.mo_energy[0][3] - 0.43451826) < 1e-4
    assert abs(gw_test.mo_energy[0][4] - 0.86603013) < 1e-4

    # An example for using chkfile
    gw_test = KRQSGW(kmf_test)
    gw_test.fc = True
    # set chkfile for saving qsgw results
    gw_test.chkfile = 'c_qsgw.chk'
    # set diis_file for saving DIIS vectors
    gw_test.diis_file = 'c_diis.h5'
    # here we force qsgw to quit after 3 cycles
    gw_test.max_cycle = 3
    gw_test.kernel()

    gw_test = KRQSGW(kmf_test)
    gw_test.fc = True
    # load qsgw chkfile
    gw_test.load_chkfile = True
    gw_test.chkfile = 'c_qsgw.chk'
    # load DIIS vectors
    if rank == 0:
        gw_test.diis = mol_scf.diis.DIIS().restore('c_diis.h5')
    comm.Barrier()
    gw_test.kernel()

    # An example for QSGW Green's function
    # NOTE: this example assumes QSGW already converges and
    #       there is a chkfile without ac_coeff
    gw_test = KRQSGW(kmf_test)
    gw_test.fc = True
    gw_test.load_chkfile = True
    gw_test.chkfile = 'c_qsgw.chk'
    if rank == 0:
        gw_test.diis = mol_scf.diis.DIIS().restore('c_diis.h5')
    comm.Barrier()
    gw_test.max_cycle = 1
    gw_test.conv_tol = 1e-8
    gw_test.kernel()

    nkpts_test = len(gw_test._scf.mo_energy)
    # gfomega: freq points for computing GF, eta: broadening
    gfomega_test = np.linspace(0.0 / 27.211386, 30.0 / 27.211386, 301)
    eta_test = 0.2 / 27.211386
    gf_test = gw_test.make_gf(gfomega_test, eta_test)
    if rank == 0:
        for i in range(len(gfomega_test)):
            print(gfomega_test[i], -np.trace((gf_test.sum(axis=0) / nkpts_test)[:, :, i].imag) / np.pi)
    comm.Barrier()
    if rank == 0:
        os.remove('c_diis.h5')
        os.remove('c_qsgw.chk')

    # Test on Na (metallic)
    cell_test = gto.Cell()
    cell_test.build(
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

    kpts_test = cell_test.make_kpts([2, 2, 1], scaled_center=[0, 0, 0])
    gdf_test = df.RSDF(cell_test, kpts_test)
    gdf_fname_test = 'gdf_ints_221.h5'
    gdf_test._cderi_to_save = gdf_fname_test
    if not os.path.isfile(gdf_fname_test):
        gdf_test.build()

    chkfname_test = 'na_221.chk'
    if os.path.isfile(chkfname_test):
        kmf_test = dft.KRKS(cell_test, kpts_test).rs_density_fit()
        kmf_test = scf.addons.smearing_(kmf_test, sigma=5e-3, method='fermi')
        kmf_test.xc = 'lda'
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        data_test = chkfile.load(chkfname_test, 'scf')
        kmf_test.__dict__.update(data_test)
    else:
        kmf_test = dft.KRKS(cell_test, kpts_test).rs_density_fit()
        kmf_test = scf.addons.smearing_(kmf_test, sigma=5e-3, method='fermi')
        kmf_test.xc = 'lda'
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        kmf_test.conv_tol = 1e-12
        kmf_test.chkfile = chkfname_test
        kmf_test.kernel()

    gw_test = KRQSGW(kmf_test)
    gw_test.conv_tol = 1e-4
    gw_test.kernel()
    assert abs(gw_test.mo_energy[0][4] - 0.057319) < 5e-4
    assert abs(gw_test.mo_energy[1][4] - 0.198266) < 5e-4

    # frozen core
    gw_test = KRQSGW(kmf_test)
    gw_test.conv_tol = 1e-4
    gw_test.frozen = 1
    gw_test.kernel()
    assert abs(gw_test.mo_energy[0][4] - 0.056628) < 5e-4
    assert abs(gw_test.mo_energy[1][4] - 0.197958) < 5e-4

    if rank == 0:
        print('passed tests!')
