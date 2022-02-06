#!/usr/bin/python
from pyscf.pbc import df, dft, gto
import numpy as np
import os, h5py
from pyscf.pbc.lib import chkfile
from pyscf import lib

einsum = lib.einsum

def at(a0):
    a = np.zeros((3,3))       # generators of rhombohedral cell
    a[0,:] = ( 1.00, 0.50, 0.50) #
    a[1,:] = ( 0.50, 1.00, 0.50) #
    a[2,:] = ( 0.50,  0.50, 1.00)
    a *= a0

    g = np.zeros((4,3))        # ions in the cell, in crystal coordinates
    g[0,:]  = ( 0.00, 0.00, 0.00) # Ni
    g[1,:]  = ( 0.50, 0.50, 0.50) # Ni
    g[2,:]  = ( 0.25, 0.25, 0.25) # O
    g[3,:]  = ( 0.75, 0.75, 0.75) # O

    pos=[]
    for mu in range(4):
        v=[0.00,0.00,0.00]
        for nu in range(3):
            v=v[:]+g[mu,nu]*a[nu,:]    # actual coordinates of the ions
        z='Ni'
        if(mu>1): z='O'
        pos.append([z,v])

    return a,pos

a0 = 4.17
vec, posion  = at(a0)
cell = gto.Cell()
cell.build(unit = 'angstrom',
           a = vec,
           atom = posion,
           dimension = 3,
           max_memory = 64000,
           verbose = 5,
           basis='gth-dzvp-molopt-sr',
           pseudo='gth-pbe',
           precision=1e-12)

kmesh = [4,4,4]
kpts = cell.make_kpts(kmesh,scaled_center=[0,0,0],wrap_around=True)
gdf = df.GDF(cell, kpts)
gdf.auxbasis = df.aug_etb(cell, beta=2.3)
gdf_fname = 'gdf_ints_444.h5'
gdf._cderi_to_save = gdf_fname
gdf.mesh = np.asarray([25, 25, 25])
if not os.path.isfile(gdf_fname):
    gdf.build()

chkfname = 'nio_444.chk'
if os.path.isfile(chkfname):
    kmf = dft.KUKS(cell, kpts).density_fit()
    kmf.xc = 'pbe'
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.conv_tol = 1e-12
    data = chkfile.load(chkfname, 'scf')
    kmf.__dict__.update(data)
else:
    kmf = dft.KUKS(cell, kpts).density_fit()
    kmf.xc = 'pbe'
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.conv_tol = 1e-12
    kmf.chkfile = chkfname

    aoind = cell.aoslice_by_atom()
    dm = kmf.get_init_guess()
    dm[0,:,aoind[0][2]:aoind[0][3], aoind[0][2]:aoind[0][3]] = 2. * dm[0,:,aoind[0][2]:aoind[0][3], aoind[0][2]:aoind[0][3]]
    dm[0,:,aoind[1][2]:aoind[1][3], aoind[1][2]:aoind[1][3]] = 0. * dm[0,:,aoind[1][2]:aoind[1][3], aoind[1][2]:aoind[1][3]]
    dm[1,:,aoind[0][2]:aoind[0][3], aoind[0][2]:aoind[0][3]] = 0. * dm[1,:,aoind[0][2]:aoind[0][3], aoind[0][2]:aoind[0][3]]
    dm[1,:,aoind[1][2]:aoind[1][3], aoind[1][2]:aoind[1][3]] = 2. * dm[1,:,aoind[1][2]:aoind[1][3], aoind[1][2]:aoind[1][3]]
    kmf.kernel(dm)

