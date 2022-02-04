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
Periodic spin-unrestricted random phase approximation
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
from pyscf import df, dft, scf
from pyscf.mp.ump2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__
from fcdmft.rpa.mol.urpa import URPA, get_rpa_ecorr, \
                get_rho_response, _mo_energy_without_core, _mo_without_core
from fcdmft.rpa.mol.rpa import _get_scaled_legendre_roots

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
    uhf = scf.UHF(rpa.mol, exxdiv=mf.exxdiv)
    uhf.with_df = mf.with_df
    uhf.with_df._cderi = mf.with_df._cderi
    e_hf = uhf.energy_elec(dm=dm)[0]
    e_hf += mf.energy_nuc()

    # Compute RPA correlation energy
    e_corr = get_rpa_ecorr(rpa, Lpq, freqs, wts)

    # Compute totol energy
    e_tot = e_hf + e_corr

    logger.debug(rpa, '  RPA total energy = %s', e_tot)
    logger.debug(rpa, '  EXX energy = %s, RPA corr energy = %s', e_hf, e_corr)

    return e_tot, e_hf, e_corr

class URPA(URPA):

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, nw=40):
        """
        Args:
            mo_energy : 2D array (2, nmo), mean-field mo energy
            mo_coeff : 3D array (2, nmo, nmo), mean-field mo coefficient
            Lpq : 4D array (2, naux, nmo, nmo), 3-index ERI
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
        nmoa, nmob = self.nmo
        nao = self.mo_coeff[0].shape[0]
        naux = self.with_df.get_naoaux()
        kpts = self._scf.with_df.kpts
        mem_incore = (nmoa**2*naux + nmob**2*naux + nao**2*naux) * 8/1e6
        mem_now = lib.current_memory()[0]

        moa = np.asarray(mo_coeff[0], order='F')
        mob = np.asarray(mo_coeff[1], order='F')
        ijslicea = (0, nmoa, 0, nmoa)
        ijsliceb = (0, nmob, 0, nmob)
        Lpqa = None
        Lpqb = None

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
            Lpqa = _ao2mo.r_e2(np.array(eri_3d_kpts[0][0]), moa, ijslicea, tao, ao_loc, out=Lpqa)
            tao = []
            ao_loc = None
            Lpqb = _ao2mo.r_e2(np.array(eri_3d_kpts[0][0]), mob, ijsliceb, tao, ao_loc, out=Lpqb)
            return np.asarray((Lpqa.real.reshape(naux,nmoa,nmoa),Lpqb.real.reshape(naux,nmob,nmob)))
        else:
            logger.warn(self, 'Memory not enough!')
            raise NotImplementedError

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, dft, df, tools
    from pyscf.pbc.lib import chkfile

    ucell = gto.Cell()
    ucell.build(
        unit = 'B',
        a = [[ 0.,          6.74027466,  6.74027466],
             [ 6.74027466,  0.,          6.74027466],
             [ 6.74027466,  6.74027466,  0.        ]],
        atom = '''H 0 0 0
                  H 1.68506866 1.68506866 1.68506866
                  H 3.37013733 3.37013733 3.37013733''',
        basis = 'gth-dzvp',
        pseudo = 'gth-pade',
        verbose = 5,
        charge = 0,
        spin = 1)

    kmesh = [3,1,1]
    cell = tools.super_cell(ucell, kmesh)
    cell.verbose = 5
    cell.spin = ucell.spin * 3

    gdf = df.GDF(cell)
    gdf_fname = 'gdf_ints.h5'
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'h3_hf.chk'
    if os.path.isfile(chkfname):
        kmf = scf.UHF(cell).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.UHF(cell).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    rpa = URPA(kmf)
    rpa.kernel()
    assert(abs(rpa.e_corr- -0.12865054702442688) < 1e-6)
    assert(abs(rpa.e_tot- -4.767926645428772) < 1e-6)
