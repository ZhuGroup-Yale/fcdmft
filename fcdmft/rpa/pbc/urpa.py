"""
Periodic gamma-point spin-unrestricted random phase approximation (direct RPA) with N^4 scaling.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14, 053020 (2012)
"""

import os
import numpy as np

from pyscf import df, scf
from pyscf.ao2mo import _ao2mo
from pyscf.lib import current_memory, temporary_env
from fcdmft.rpa.mol.urpa import URPA


class URPA(URPA):
    def get_ehf(self):
        """Get Hartree-Fock energy.

        Returns
        -------
        e_hf : double
            Hartree-Fock energy
        """
        with temporary_env(self.with_df, verbose=0), temporary_env(self.mol, verbose=0):
            dm = self._scf.make_rdm1()
            uhf = scf.UHF(self.mol, exxdiv=self._scf.exxdiv)
            uhf.verbose = 0
            uhf.with_df = self._scf.with_df
            e_hf = uhf.energy_elec(dm=dm)[0]
            e_hf += self._scf.energy_nuc()
        return e_hf

    def ao2mo(self, mo_coeff=None):
        """Transform density-fitting integral from AO to MO.

        Parameters
        ----------
        mo_coeff : double 3d array, optional
            coefficient from AO to MO, by default None

        Returns
        -------
        double 4d array
            three-center density-fitting matrix in MO
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmoa = mo_coeff[0].shape[1]
        nmob = mo_coeff[1].shape[1]
        nao = self.mo_coeff[0].shape[0]
        naux = self.with_df.get_naoaux()
        kpts = self._scf.with_df.kpts
        max_memory = max(2000, self._scf.max_memory - current_memory()[0] - nao**2 * naux * 8 / 1e6)

        moa = np.asarray(mo_coeff[0], order='F')
        mob = np.asarray(mo_coeff[1], order='F')
        ijslicea = (0, nmoa, 0, nmoa)
        ijsliceb = (0, nmob, 0, nmob)

        from pyscf.pbc.df.fft_ao2mo import _format_kpts

        kptijkl = _format_kpts(kpts)
        eri_3d_a = []
        eri_3d_b = []
        for LpqR, _, _ in self._scf.with_df.sr_loop(kptijkl[:2], max_memory=0.2 * max_memory, compact=False):
            Lpqa = None
            Lpqb = None
            Lpqa = _ao2mo.nr_e2(LpqR.reshape(-1, nao, nao), moa, ijslicea, aosym='s1', mosym='s1', out=Lpqa)
            Lpqb = _ao2mo.nr_e2(LpqR.reshape(-1, nao, nao), mob, ijsliceb, aosym='s1', mosym='s1', out=Lpqb)
            eri_3d_a.append(Lpqa)
            eri_3d_b.append(Lpqb)
        eri_3d_a = np.vstack(eri_3d_a).reshape(-1, nmoa, nmoa)
        eri_3d_b = np.vstack(eri_3d_b).reshape(-1, nmob, nmob)
        return np.array([eri_3d_a, eri_3d_b])


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, df, tools
    from pyscf.pbc.lib import chkfile

    ucell_test = gto.Cell()
    ucell_test.build(
        unit='B',
        a=[[0.0, 6.74027466, 6.74027466], [6.74027466, 0.0, 6.74027466], [6.74027466, 6.74027466, 0.0]],
        atom="""H 0 0 0
                H 1.68506866 1.68506866 1.68506866
                H 3.37013733 3.37013733 3.37013733""",
        basis='gth-dzvp',
        pseudo='gth-pade',
        max_memory=20000,
        verbose=5,
        charge=0,
        spin=1,
    )

    kmesh_test = [3, 1, 1]
    cell_test = tools.super_cell(ucell_test, kmesh_test)
    cell_test.verbose = 5
    cell_test.spin = ucell_test.spin * 3

    gdf_test = df.RSDF(cell_test)
    gdf_fname_test = 'gdf_ints.h5'
    gdf_test._cderi_to_save = gdf_fname_test
    if not os.path.isfile(gdf_fname_test):
        gdf_test.build()

    chkfname_test = 'h3_hf.chk'
    if os.path.isfile(chkfname_test):
        kmf_test = scf.UHF(cell_test).rs_density_fit()
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        kmf_test.conv_tol = 1e-12
        data_test = chkfile.load(chkfname_test, 'scf')
        kmf_test.__dict__.update(data_test)
    else:
        kmf_test = scf.UHF(cell_test).rs_density_fit()
        kmf_test.with_df = gdf_test
        kmf_test.with_df._cderi = gdf_fname_test
        kmf_test.conv_tol = 1e-12
        kmf_test.chkfile = chkfname_test
        kmf_test.kernel()

    rpa_test = URPA(kmf_test)
    rpa_test.kernel()
    assert abs(rpa_test.e_corr - -0.12865055896219463) < 1e-6
    assert abs(rpa_test.e_tot - -4.7544193631366) < 1e-6

    print('passed tests!')
