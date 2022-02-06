#!/usr/bin/python
from pyscf.pbc import df, dft, scf, cc, gto
import numpy as np
import h5py
from libdmet_solid.system import lattice
from pyscf.pbc.lib import chkfile
import os
from pyscf import lib
import libdmet_solid.utils.logger as log
from fcdmft.gw.pbc import krgw_gf
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

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
           max_memory = 40000,
           verbose = 5,
           pseudo = 'gth-pade',
           basis='gth-dzvp-molopt-sr',
           precision=1e-12)

kmesh = [5,5,5]
kpts = cell.make_kpts(kmesh,scaled_center=[0,0,0])
Lat = lattice.Lattice(cell, kmesh)
nao = Lat.nao
Lat.kpts = kpts
nkpts = Lat.nkpts

gdf_fname = 'gdf_ints_555.h5'
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.auxbasis = df.aug_etb(cell, beta=2.3, use_lval=True, l_val_set={'Mo':2,'O':1,'Sr':1})
if not os.path.isfile(gdf_fname):
    gdf.build()

chkfname = 'srmoo3_555.chk'
if os.path.isfile(chkfname):
    kmf = dft.KRKS(cell, kpts)
    kmf.xc = 'lda'
    kmf.exxdiv = None
    kmf = scf.addons.smearing_(kmf, sigma=5e-3, method="fermi")
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    data = chkfile.load(chkfname, 'scf')
    kmf.__dict__.update(data)
else:
    kmf = dft.KRKS(cell, kpts)
    kmf.xc = 'lda'
    kmf.exxdiv = None
    kmf = scf.addons.smearing_(kmf, sigma=5e-3, method="fermi")
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.conv_tol = 1e-12
    kmf.chkfile = chkfname
    kmf.diis_space = 15
    kmf.max_cycle = 50
    kmf.kernel()

gw = krgw_gf.KRGWGF(kmf)
nmo = len(kmf.mo_energy[0])
gw.ac = 'pade'
gw.eta = 0.1/27.211386
gw.fullsigma = True
gw.ef = 0.560479331479
omega = np.linspace(6./27.211386,24./27.211386,181)
gf, gf0, sigma = gw.kernel(omega=omega, orbs=range(0,nmo), writefile=1)

if rank == 0:
    for i in range(len(omega)):
        print (omega[i],(-np.trace(gf0[:,:,:,i].imag.sum(axis=0)/nkpts))/np.pi, \
                       (-np.trace(gf[:,:,:,i].imag.sum(axis=0)/nkpts))/np.pi)
    print ('------------------')
    for i in range(len(omega)):
        print (omega[i],(-np.trace(gf0[1,:,:,i].imag))/np.pi, \
                         (-np.trace(gf[1,:,:,i].imag))/np.pi)

