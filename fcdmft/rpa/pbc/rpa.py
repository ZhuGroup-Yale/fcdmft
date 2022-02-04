#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Tianyu Zhu <zhutianyu1991@gmail.com>
#

"""
Periodic spin-restricted random phase approximation
(direct RPA/dRPA in chemistry) with N^4 scaling (Gamma only)

Method:
    Main routines are based on GW-AC method descirbed in:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    X. Ren et al., New J. Phys. 14, 053020 (2012)
"""

import time, h5py, os
from functools import reduce
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.pbc import df, dft, scf
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__
from fcdmft.rpa.mol.rpa import RPA, get_rpa_ecorr, _get_scaled_legendre_roots, \
                get_rho_response, _mo_energy_without_core, _mo_without_core

einsum = lib.einsum

def kernel(rpa, mo_energy, mo_coeff, Lpq=None, nw=None, verbose=logger.NOTE):
    """
    RPA correlation and total energy

    Args:
        Lpq : density fitting 3-center integral in MO basis.
        nw : number of frequency point on imaginary axis.
        vhf_df : using density fitting integral to compute HF exchange.

    Returns:
        e_tot : RPA total energy
        e_hf : EXX energy
        e_corr : RPA correlation energy
    """
    mf = rpa._scf
    assert(rpa.frozen is None)

    if Lpq is None:
        Lpq = rpa.ao2mo(mo_coeff)

    # Grids for integration on imaginary axis
    freqs, wts = _get_scaled_legendre_roots(nw)

    # Compute HF exchange energy (EXX)
    dm = mf.make_rdm1()
    rhf = scf.RHF(rpa.mol, exxdiv=mf.exxdiv)
    rhf.with_df = mf.with_df
    rhf.with_df._cderi = mf.with_df._cderi
    e_hf = rhf.energy_elec(dm=dm)[0]
    e_hf += mf.energy_nuc()

    # Compute RPA correlation energy
    e_corr = get_rpa_ecorr(rpa, Lpq, freqs, wts)

    # Compute totol energy
    e_tot = e_hf + e_corr

    logger.debug(rpa, '  RPA total energy = %s', e_tot)
    logger.debug(rpa, '  EXX energy = %s, RPA corr energy = %s', e_hf, e_corr)

    return e_tot, e_hf, e_corr


class RPA(RPA):

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

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, nw=40):
        """
        Args:
            mo_energy : 1D array (nmo), mean-field mo energy
            mo_coeff : 2D array (nmo, nmo), mean-field mo coefficient
            Lpq : 3D array (naux, nmo, nmo), 3-index ERI
            nw: interger, grid number

        Returns:
            self.e_tot : RPA total eenrgy
            self.e_hf : EXX energy
            self.e_corr : RPA correlation energy
        """
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self._scf.mo_energy)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.e_tot, self.e_hf, self.e_corr = \
                        kernel(self, mo_energy, mo_coeff, Lpq=Lpq, nw=nw, verbose=self.verbose)

        logger.timer(self, 'RPA', *cput0)
        return self.e_corr

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        naux = self.with_df.get_naoaux()
        kpts = self._scf.with_df.kpts
        mem_incore = (2 * nmo**2*naux) * 8 /1e6
        mem_now = lib.current_memory()[0]

        mo = np.asarray(mo_coeff, order='F')
        ijslice = (0, nmo, 0, nmo)
        Lpq = None

        eri_3d_kpts = []
        for i, kpti in enumerate(kpts):
            eri_3d_kpts.append([])
            for j, kptj in enumerate(kpts):
                eri_3d = []
                for LpqR, LpqI, sign in self._scf.with_df.sr_loop([kpti,kptj], max_memory=mem_now, compact=False):
                    eri_3d.append(LpqR+LpqI*1j)
                eri_3d = np.vstack(eri_3d).reshape(-1,nao,nao)
                eri_3d_kpts[i].append(eri_3d)

        if (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway:
            tao = []
            ao_loc = None
            Lpq = _ao2mo.r_e2(np.array(eri_3d_kpts[0][0]), mo, ijslice, tao, ao_loc, out=Lpq)
            return Lpq.real.reshape(naux,nmo,nmo)
        else:
            logger.warn(self, 'Memory not enough!')
            raise NotImplementedError

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, dft, df, tools
    from pyscf.pbc.lib import chkfile

    ucell = gto.Cell()
    ucell.build(unit = 'angstrom',
            a = '''
                0.000000     1.783500     1.783500
                1.783500     0.000000     1.783500
                1.783500     1.783500     0.000000
            ''',
            atom = 'C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
            dimension = 3,
            max_memory = 64000,
            verbose = 5,
            pseudo = 'gth-pbe',
            basis='gth-dzv',
            precision=1e-12)

    kmesh = [3,1,1]
    cell = tools.super_cell(ucell, kmesh)
    cell.verbose = 5

    gdf = df.GDF(cell)
    gdf_fname = 'gdf_ints.h5'
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'diamond_hf.chk'
    if os.path.isfile(chkfname):
        kmf = scf.RHF(cell).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.RHF(cell).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    rpa = RPA(kmf)
    rpa.kernel()
    assert(abs(rpa.e_corr- -0.5558316165999143) < 1e-6)
    assert(abs(rpa.e_tot- -32.08317615664809) < 1e-6)
