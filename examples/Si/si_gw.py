#!/usr/bin/python
from pyscf.pbc import df, dft, gto
import numpy as np
import os, h5py
from pyscf.pbc.lib import chkfile
from pyscf import lib
from fcdmft.gw.pbc import krgw_gf
from fcdmft.utils import write

einsum = lib.einsum

cell = gto.Cell()
cell.build(unit = 'angstrom',
           a = '''
            0.000000     2.715000     2.715000
            2.715000     0.000000     2.715000
            2.715000     2.715000     0.000000
           ''',
           atom = 'Si 2.03625 2.03625 2.03625; Si 3.39375 3.39375 3.39375',
           dimension = 3,
           max_memory = 32000,
           verbose = 5,
           basis='gth-dzvp',
           pseudo='gth-pbe',
           precision=1e-12)

kmesh = [4,4,4]
kpts = cell.make_kpts(kmesh,scaled_center=[0,0,0])
gdf = df.GDF(cell, kpts)
gdf.auxbasis = df.aug_etb(cell, beta=2.0)
gdf_fname = 'gdf_ints_444.h5'
gdf._cderi_to_save = gdf_fname
if not os.path.isfile(gdf_fname):
    gdf.build()

chkfname = 'si_444.chk'
if os.path.isfile(chkfname):
    kmf = dft.KRKS(cell, kpts).density_fit()
    kmf.xc = 'pbe'
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.conv_tol = 1e-12
    data = chkfile.load(chkfname, 'scf')
    kmf.__dict__.update(data)
else:
    kmf = dft.KRKS(cell, kpts).density_fit()
    kmf.xc = 'pbe'
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.conv_tol = 1e-12
    kmf.chkfile = chkfname
    kmf.kernel()

gw = krgw_gf.KRGWGF(kmf)
gw.ac = 'pade'
Ha2ev = 27.211386
gw.eta = 0.1/Ha2ev
gw.fullsigma = True
gw.fc = True
omega = np.linspace(0.,18./Ha2ev,181)
# writefile must >= 1 for GW+DMFT calc
gf, gf0, sigma = gw.kernel(omega=omega, writefile=1)

gf = gf[np.newaxis, ...]
gf0 = gf0[np.newaxis, ...]
nkpts = gw.nkpts
gf = 1./nkpts * np.sum(gf, axis=1)
gf0 = 1./nkpts * np.sum(gf0, axis=1)

outdir = 'GW_DOS'
if not os.path.isdir(outdir):
    os.mkdir(outdir)
write.write_gf_to_dos(outdir+'/si_gw_dos', omega, gf)
write.write_gf_to_dos(outdir+'/si_pbe_dos', omega, gf0)

mo_energy = gw.mo_energy
nocc = gw.nocc
homo = -99.; lumo = 99.
for k in range(nkpts):
    if homo < mo_energy[k][nocc-1]:
        homo = mo_energy[k][nocc-1]
    if lumo > mo_energy[k][nocc]:
        lumo = mo_energy[k][nocc]
print ('VBM, CBM, Gap', homo*Ha2ev, lumo*Ha2ev, (lumo-homo)*Ha2ev)
