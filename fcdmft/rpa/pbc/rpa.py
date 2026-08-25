"""
Periodic gamma-point spin-restricted random phase approximation (direct RPA) with N^4 scaling.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14, 053020 (2012)
"""

import numpy as np
import os

from pyscf.ao2mo import _ao2mo
from pyscf.lib import current_memory, temporary_env, einsum
from pyscf.pbc import df, dft, scf, tools

from fcdmft.rpa.mol.rpa import RPA, get_idx_metal, _mo_energy_without_core


def get_rpa_exx(rpa, acfd=False, correction_only=False):
    """Calculate periodic gamma-point RPA exchange energy."""
    mo_energy = np.array(rpa._scf.mo_energy)
    mo_coeff = np.array(rpa._scf.mo_coeff)
    mo_occ = np.array(rpa._scf.mo_occ)

    if getattr(rpa, 'with_df', None) is None:
        rpa.initialize_df(auxbasis=rpa.auxbasis)

    if not acfd and not correction_only:
        dm = rpa._scf.make_rdm1()
        with temporary_env(rpa.with_df, verbose=0), temporary_env(rpa.mol, verbose=0):
            rhf = scf.RHF(rpa.mol, exxdiv=rpa._scf.exxdiv)
            rhf.verbose = 0
            if hasattr(rpa._scf, 'sigma'):
                rhf = scf.addons.smearing_(rhf, sigma=rpa._scf.sigma, method=rpa._scf.smearing_method)
            rhf.with_df = rpa._scf.with_df
            vk = rhf.get_k(rhf.cell, dm)
        return -0.25 * einsum('ij,ji', dm, vk)

    if not hasattr(rpa._scf, 'sigma'):
        return 0.0

    idx_occ, idx_frac, _ = get_idx_metal(mo_occ)
    nocc = len(idx_occ) + len(idx_frac)
    Lij = rpa.loop_ao2mo(mo_coeff=mo_coeff, ijslice=(0, nocc, 0, nocc))

    if acfd:
        if correction_only:
            mo_occ_ij = np.minimum(mo_occ[:nocc, None], mo_occ[None, :nocc]) / 2.0
            mo_occ_ij -= mo_occ[:nocc, None] * mo_occ[None, :nocc] / 4.0
        else:
            eij = mo_energy[:nocc, None] - mo_energy[None, :nocc]
            integrand = np.zeros((nocc, nocc), dtype=np.double)
            integrand[eij > 1e-6] = 1
            integrand[eij < -1e-6] = -1
            mo_occ_ij = 1.0 - integrand
            mo_occ_ij = mo_occ_ij * mo_occ[:nocc, None] / 2.0
    else:
        mo_occ_ij = mo_occ[:nocc, None] * mo_occ[None, :nocc] / 4.0

    ex = -np.dot((Lij * mo_occ_ij[None]).ravel(), Lij.ravel())

    if rpa._scf.exxdiv == 'ewald' and rpa._scf.cell.dimension != 0:
        madelung = tools.pbc.madelung(rpa._scf.cell, [np.zeros(3)])
        ex -= 0.5 * madelung * np.sum(mo_occ**2)
        if acfd:
            f_i = mo_occ[:nocc]
            ex -= 0.5 * madelung * np.sum(f_i - f_i * f_i)

    return ex


class RPA(RPA):
    def get_ehf(self):
        """Get Hartree-Fock energy.

        Returns
        -------
        e_hf : double
            Hartree-Fock energy
        """
        with temporary_env(self.with_df, verbose=0), temporary_env(self.mol, verbose=0):
            dm = self._scf.make_rdm1()
            rhf = scf.RHF(self.mol, exxdiv=self._scf.exxdiv)
            rhf.verbose = 0
            if hasattr(self._scf, 'sigma'):
                rhf = scf.addons.smearing_(rhf, sigma=self._scf.sigma, method='fermi')
            rhf.with_df = self._scf.with_df
            with temporary_env(self.with_df, verbose=0), temporary_env(rhf.mol, verbose=0):
                dm = self._scf.make_rdm1()
                e_1e = einsum('ij,ji', dm, rhf.get_hcore())
                e_j = 0.5 * einsum('ij,ji', dm, rhf.get_j(rhf.mol, dm))
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
        return get_rpa_exx(self, acfd=acfd, correction_only=correction_only)

    def ao2mo(self, mo_coeff=None, mo_occ=None):
        """Transform density-fitting integral from AO to MO.

        Parameters
        ----------
        mo_coeff : double 1d array, optional
            orbital energy, by default None
        mo_occ : double 1d array, optional
            occupation number, by default None

        Returns
        -------
        eri_3d : double 3d array
            three-center density-fitting matrix in MO
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = mo_coeff.shape[1]
        nao = self.mo_coeff.shape[0]
        naux = self.with_df.get_naoaux()
        kpts = self._scf.with_df.kpts
        max_memory = max(2000, self._scf.max_memory - current_memory()[0] - nao**2 * naux * 8 / 1e6)

        is_metal = False
        if mo_occ is None:
            mo_occ_1d = _mo_energy_without_core(self, self._scf.mo_occ).reshape(-1)
        else:
            mo_occ_1d = np.array(mo_occ).reshape(-1)
        if np.linalg.norm(np.abs(mo_occ_1d - 1.0) - 1.0) > 1e-5:
            is_metal = True

        mo = np.asarray(mo_coeff, order='F')
        if is_metal:
            idx_occ, idx_frac, _ = get_idx_metal(mo_occ_1d)
            nocc = len(idx_occ)
            nfrac = len(idx_frac)
            ijslice = (0, nocc + nfrac, nocc, nmo)
        else:
            idx_occ, idx_frac, _ = get_idx_metal(mo_occ_1d)
            nocc = len(idx_occ)
            ijslice = (0, nocc, nocc, nmo)
        nislice = ijslice[1] - ijslice[0]
        njslice = ijslice[3] - ijslice[2]

        kptijkl = df.fft_ao2mo._format_kpts(kpts)
        eri_3d = []
        for LpqR, _, _ in self._scf.with_df.sr_loop(kptijkl[:2], max_memory=0.3 * max_memory, compact=False):
            Lpq = None
            Lpq = _ao2mo.nr_e2(LpqR.reshape(-1, nao, nao), mo, ijslice, aosym='s1', mosym='s1', out=Lpq)
            eri_3d.append(Lpq)
        eri_3d = np.vstack(eri_3d).reshape(-1, nislice, njslice)

        return eri_3d

    def loop_ao2mo(self, mo_coeff=None, ijslice=None):
        """ransform density-fitting integral from AO to MO by block.

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
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = mo_coeff.shape[1]
        nao = self.mo_coeff.shape[0]
        self.with_df.get_naoaux()
        kpts = self._scf.with_df.kpts

        is_metal = False
        mo_occ_1d = np.array(self._scf.mo_occ).reshape(-1)
        if np.linalg.norm(np.abs(mo_occ_1d - 1.0) - 1.0) > 1e-5:
            is_metal = True

        mo = np.asarray(mo_coeff, order='F')
        if ijslice is None:
            if is_metal:
                # Assume the ordering of mo_coeff is
                # occ, frac, vir
                idx_occ, idx_frac, idx_vir = get_idx_metal(mo_occ_1d)
                for i in idx_occ:
                    assert i < idx_frac[0]
                for i in idx_frac:
                    assert i < idx_vir[0]
                nocc = len(idx_occ)
                nfrac = len(idx_frac)
                ijslice = (0, nocc + nfrac, nocc, nmo)
            else:
                idx_occ, idx_frac, _ = get_idx_metal(mo_occ_1d)
                nocc = len(idx_occ)
                ijslice = (0, nocc, nocc, nmo)
        nislice = ijslice[1] - ijslice[0]
        njslice = ijslice[3] - ijslice[2]

        kptijkl = df.fft_ao2mo._format_kpts(kpts)
        eri_3d = []
        for LpqR, _, _ in self._scf.with_df.sr_loop(
            kptijkl[:2], max_memory=0.1 * self._scf.max_memory, compact=False
        ):
            Lpq = None
            Lpq = _ao2mo.nr_e2(LpqR.reshape(-1, nao, nao), mo, ijslice, aosym='s1', mosym='s1', out=Lpq)
            eri_3d.append(Lpq)
        eri_3d = np.vstack(eri_3d).reshape(-1, nislice, njslice)

        return eri_3d


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, dft, df, tools
    from pyscf.pbc.lib import chkfile

    # test on diamond
    ucell_test = gto.Cell()
    ucell_test.build(
        unit='angstrom',
        a="""
                0.000000     1.783500     1.783500
                1.783500     0.000000     1.783500
                1.783500     1.783500     0.000000
            """,
        atom='C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
        dimension=3,
        max_memory=16000,
        verbose=5,
        pseudo='gth-pbe',
        basis='gth-dzv',
        precision=1e-12,
    )

    kmesh_test = [3, 1, 1]
    cell_test = tools.super_cell(ucell_test, kmesh_test)
    cell_test.verbose = 5

    gdf_test = df.RSDF(cell_test)
    gdf_fname_test = 'gdf_ints.h5'
    gdf_test._cderi_to_save = gdf_fname_test
    if not os.path.isfile(gdf_fname_test):
        gdf_test.build()

    chkfname_test = 'diamond_hf.chk'
    if os.path.isfile(chkfname_test):
        kmf_test = scf.RHF(cell_test).rs_density_fit()
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        kmf_test.conv_tol = 1e-12
        data_test = chkfile.load(chkfname_test, 'scf')
        kmf_test.__dict__.update(data_test)
    else:
        kmf_test = scf.RHF(cell_test).rs_density_fit()
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        kmf_test.conv_tol = 1e-12
        kmf_test.chkfile = chkfname_test
        kmf_test.kernel()

    rpa_test = RPA(kmf_test)
    rpa_test.kernel()
    assert abs(rpa_test.e_corr - -0.5558316165999143) < 1e-6
    assert abs(rpa_test.e_tot - -32.08317615664809) < 1e-6

    # Test on Na (metallic)
    ucell_test = gto.Cell()
    ucell_test.build(
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
        max_memory=16000,
        verbose=5,
        pseudo='gth-pade',
        basis='gth-dzvp-molopt-sr',
        precision=1e-10,
    )

    kmesh_test = [2, 2, 1]
    cell_test = tools.super_cell(ucell_test, kmesh_test)
    cell_test.verbose = 5

    gdf_test = df.RSDF(cell_test)
    gdf_fname_test = 'gdf_ints_na.h5'
    gdf_test._cderi_to_save = gdf_fname_test
    if not os.path.isfile(gdf_fname_test):
        gdf_test.build()

    chkfname_test = 'na_lda.chk'
    if os.path.isfile(chkfname_test):
        kmf_test = dft.RKS(cell_test).rs_density_fit()
        kmf_test = scf.addons.smearing_(kmf_test, sigma=5e-3, method='fermi')
        kmf_test.xc = 'lda'
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        kmf_test.conv_tol = 1e-12
        data = chkfile.load(chkfname_test, 'scf')
        kmf_test.__dict__.update(data)
    else:
        kmf_test = dft.RKS(cell_test).rs_density_fit()
        kmf_test = scf.addons.smearing_(kmf_test, sigma=5e-3, method='fermi')
        kmf_test.xc = 'lda'
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        kmf_test.conv_tol = 1e-12
        kmf_test.chkfile = chkfname_test
        kmf_test.kernel()

    rpa_test = RPA(kmf_test)
    rpa_test.kernel()

    assert abs(rpa_test.e_corr - -0.1612718859071952) < 1e-6  # old value: -0.1612721103400592
    assert abs(rpa_test.e_tot - -190.42773568569532) < 1e-6  # old value: -190.4277327357537

    print('passed tests!')
