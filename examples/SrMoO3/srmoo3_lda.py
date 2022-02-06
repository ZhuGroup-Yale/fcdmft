#!/usr/bin/python
from pyscf.pbc import df, dft, scf, cc, gto
import numpy as np
import h5py
from libdmet_solid.system import lattice
from libdmet_solid.basis_transform import make_basis
from libdmet_solid.basis_transform import eri_transform
from pyscf.pbc.lib import chkfile
import os
from pyscf import lib
import libdmet_solid.utils.logger as log

log.verbose = 'DEBUG1'

einsum = lib.einsum

cell = gto.Cell()
cell.build(unit = 'angstrom',
           a = '''
           3.97600   0.00000   0.00000
           0.00000   3.97600   0.00000
           0.00000   0.00000   3.97600
           ''',
           atom = '''
           Mo   1.98800   1.98800   1.98800
           Sr   0.00000   0.00000   0.00000
            O   1.98800   1.98800   0.00000
            O   1.98800   0.00000   1.98800
            O   0.00000   1.98800   1.98800
           ''',
           dimension = 3,
           max_memory = 150000,
           verbose = 5,
           pseudo = 'gth-pade',
           basis='gth-dzvp-molopt-sr',
           precision=1e-12)

kmesh = [5,5,5]
Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = 'gdf_ints_555.h5'
gdf.auxbasis = df.aug_etb(cell, beta=2.3, use_lval=True, l_val_set={'Mo':2,'O':1,'Sr':1})
gdf.build()

chkfname = 'srmoo3_555.chk'
if os.path.isfile(chkfname):
    kmf = dft.KRKS(cell, kpts)
    kmf.xc = 'lda'
    kmf.exxdiv = None
    kmf = scf.addons.smearing_(kmf, sigma=5e-3, method="fermi")
    kmf.with_df = gdf
    kmf.with_df._cderi = 'gdf_ints_555.h5'
    data = chkfile.load(chkfname, 'scf')
    kmf.__dict__.update(data)
else:
    kmf = dft.KRKS(cell, kpts)
    kmf.xc = 'lda'
    kmf.exxdiv = None
    kmf = scf.addons.smearing_(kmf, sigma=5e-3, method="fermi")
    kmf.with_df = gdf
    kmf.with_df._cderi = 'gdf_ints_555.h5'
    kmf.conv_tol = 1e-12
    kmf.chkfile = chkfname
    kmf.diis_space = 15
    kmf.max_cycle = 50
    kmf.kernel()

nmo = len(kmf.mo_energy[0])

import copy
mo_occ_old = copy.deepcopy(kmf.mo_occ)
homo = -99; lumo = 99
for k in range(nkpts):
    for i in range(nmo):
        if kmf.mo_occ[k][i] > 1.0:
            kmf.mo_occ[k][i] = 2.
            if kmf.mo_energy[k][i] > homo:
                homo = kmf.mo_energy[k][i]
        else:
            kmf.mo_occ[k][i] = 0.
            if kmf.mo_energy[k][i] < lumo:
                lumo = kmf.mo_energy[k][i]

print (homo, lumo)

