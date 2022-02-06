#!/usr/bin/python
import numpy as np
import h5py, os
from libdmet_solid.system import lattice
from libdmet_solid.basis_transform import make_basis
from libdmet_solid.basis_transform import eri_transform
import libdmet_solid.utils.logger as log

from pyscf.pbc import df, dft, scf, gto
from pyscf.pbc.lib import chkfile
from pyscf import lib
from pyscf import gto as gto_mol
from pyscf import scf as scf_mol

from fcdmft.gw.mol import gw_dc
from fcdmft.utils import cholesky

import copy
from libdmet_solid.utils.misc import kdot, mdot
from libdmet_solid.lo.iao import get_idx, get_idx_each_atom, \
        get_idx_each_orbital
from libdmet_solid.lo.iao import reference_mol, get_labels, get_idx_each

log.verbose = 'DEBUG1'

einsum = lib.einsum

# NOTE: lattice system setup by user
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
           max_memory = 100000,
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
gdf_fname = 'gdf_ints_555.h5'
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

# set spin
mo_energy = np.asarray(kmf.mo_energy)
mo_coeff = np.asarray(kmf.mo_coeff)
if len(mo_energy.shape) == 2:
    spin = 1
    mo_energy = mo_energy[np.newaxis, ...]
    mo_coeff = mo_coeff[np.newaxis, ...]
else:
    spin = 2

# NOTE: choose IAO basis by user
# set IAO (core+val)
minao = 'gth-szv-molopt-sr'
pmol = reference_mol(cell, minao=minao)
basis = pmol._basis

# set valence IAO
basis_val = {}
minao_val = 'gth-szv-molopt-sr-val'
pmol_val = pmol.copy()
pmol_val.basis = minao_val
pmol_val.build()
basis_val["Sr"] = copy.deepcopy(pmol_val._basis["Sr"])
basis_val["Mo"] = copy.deepcopy(pmol_val._basis["Mo"])
basis_val["O"] = copy.deepcopy(pmol_val._basis["O"])

pmol_val = pmol.copy()
pmol_val.basis = basis_val
pmol_val.build()

val_labels = pmol_val.ao_labels()
for i in range(len(val_labels)):
    val_labels[i] = val_labels[i].replace("Mo 4s", "Mo 5s")
    val_labels[i] = val_labels[i].replace("Sr 4s", "Sr 5s")
pmol_val.ao_labels = lambda *args: val_labels

# set core IAO
basis_core = {}
minao_core = 'gth-szv-molopt-sr-core'
pmol_core = pmol.copy()
pmol_core.basis = minao_core
pmol_core.build()
basis_core["Sr"] = copy.deepcopy(pmol_core._basis["Sr"])
basis_core["Mo"] = copy.deepcopy(pmol_core._basis["Mo"])
basis_core["O"] = copy.deepcopy(pmol_core._basis["O"])

pmol_core = pmol.copy()
pmol_core.basis = basis_core
pmol_core.build()
core_labels = pmol_core.ao_labels()

ncore = len(pmol_core.ao_labels())
nval = pmol_val.nao_nr()
nvirt = cell.nao_nr() - ncore - nval
Lat.set_val_virt_core(nval, nvirt, ncore)

# First construct IAO and PAO.
C_ao_iao, C_ao_iao_val, C_ao_iao_virt, C_ao_iao_core \
        = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao, full_return=True, \
        pmol_val=pmol_val, pmol_core=pmol_core, tol=1e-9)

# C_ao_lo: transformation matrix from AO to LO (IAO) basis
nlo = nao - ncore
C_ao_lo = np.zeros((spin,nkpts,nao,nlo),dtype=np.complex128)
for s in range(spin):
    C_ao_lo[s] = C_ao_iao[:,:,ncore:]

# C_mo_lo: transformation matrix from MO to LO (IAO) basis
S_ao_ao = kmf.get_ovlp()
C_mo_lo = np.zeros((spin,nkpts,nao,nlo),dtype=np.complex128)
for s in range(spin):
    for ki in range(nkpts):
        C_mo_lo[s][ki] = np.dot(np.dot(mo_coeff[s][ki].T.conj(), S_ao_ao[ki]), C_ao_lo[s][ki])
fn = 'C_mo_lo.h5'
feri = h5py.File(fn, 'w')
feri['C_ao_lo'] = np.asarray(C_ao_lo)
feri['C_mo_lo'] = np.asarray(C_mo_lo)
feri.close()

# get DFT density matrix in IAO basis
DM_ao = np.asarray(kmf.make_rdm1())
if len(DM_ao.shape) == 3:
    DM_ao = DM_ao[np.newaxis, ...]
DM_lo = np.zeros((spin,nkpts,nlo,nlo),dtype=DM_ao.dtype)
for s in range(spin):
    for ki in range(nkpts):
        Cinv = np.dot(C_ao_lo[s][ki].T.conj(),S_ao_ao[ki])
        DM_lo[s][ki] = np.dot(np.dot(Cinv, DM_ao[s][ki]), Cinv.T.conj())

for s in range(spin):
    nelec_lo = np.trace(DM_lo[s].sum(axis=0)/nkpts)
    print ('Nelec imp', nelec_lo.real)
fn = 'DM_iao_k.h5'
feri = h5py.File(fn, 'w')
feri['DM'] = np.asarray(DM_lo)
feri.close()

# get 4-index ERI
eri = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo, feri=gdf_fname)
fn = 'eri_imp111_iao.h5'
feri = h5py.File(fn, 'w')
feri['eri'] = np.asarray(eri.real)
feri.close()

# get one-electron integrals
hcore_ao = np.asarray(kmf.get_hcore())
JK_ao = np.asarray(kmf.get_veff())
if len(JK_ao.shape) == 3:
    JK_ao = JK_ao[np.newaxis, ...]
hcore_lo = np.zeros((spin,nkpts,nlo,nlo),dtype=hcore_ao.dtype)
JK_lo = np.zeros((spin,nkpts,nlo,nlo),dtype=JK_ao.dtype)
for s in range(spin):
    for ki in range(nkpts):
        hcore_lo[s,ki] = np.dot(np.dot(C_ao_lo[s,ki].T.conj(), hcore_ao[ki]), C_ao_lo[s,ki])
        JK_lo[s,ki] = np.dot(np.dot(C_ao_lo[s,ki].T.conj(), JK_ao[s,ki]), C_ao_lo[s,ki])

fn = 'hcore_JK_iao_k_dft.h5'
feri = h5py.File(fn, 'w')
feri['hcore'] = np.asarray(hcore_lo)
feri['JK'] = np.asarray(JK_lo)
feri.close()
assert(np.max(np.abs(hcore_lo.sum(axis=1).imag/nkpts))<1e-6)
assert(np.max(np.abs(JK_lo.sum(axis=1).imag/nkpts))<1e-6)

# get HF JK term using DFT density
kmf_hf = scf.KRHF(cell, kpts, exxdiv=None)
kmf_hf.with_df = gdf
kmf_hf.with_df._cderi = gdf_fname
kmf_hf.max_cycle = 0
JK_ao = np.asarray(kmf_hf.get_veff(dm_kpts=DM_ao[0]))
if len(JK_ao.shape) == 3:
    JK_ao = JK_ao[np.newaxis, ...]

# NOTE: choose finite size correction by user
# set gw_fc to True if finite size correction is used in kgw
gw_fc = False
if gw_fc:
    # finite size correction to exchange
    vk_corr = -2./np.pi * (6.*np.pi**2/cell.vol/nkpts)**(1./3.)
    nocc = cell.nelectron // 2
    JK_mo = np.zeros((spin,nkpts,nao,nao),dtype=JK_ao.dtype)
    for s in range(spin):
        for ki in range(nkpts):
            JK_mo[s,ki] = np.dot(np.dot(mo_coeff[s,ki].T.conj(), JK_ao[s,ki]), mo_coeff[s,ki])
            for i in range(nocc):
                JK_mo[s,ki][i,i] = JK_mo[s,ki][i,i] + vk_corr

    JK_lo = np.zeros((spin,nkpts,nlo,nlo),dtype=JK_ao.dtype)
    for s in range(spin):
        for ki in range(nkpts):
            JK_lo[s,ki] = np.dot(np.dot(C_mo_lo[s,ki].T.conj(), JK_mo[s,ki]), C_mo_lo[s,ki])
else:
    JK_lo = np.zeros((spin,nkpts,nlo,nlo),dtype=JK_ao.dtype)
    for s in range(spin):
        for ki in range(nkpts):
            JK_lo[s,ki] = np.dot(np.dot(C_ao_lo[s,ki].T.conj(), JK_ao[s,ki]), C_ao_lo[s,ki])

fn = 'hcore_JK_iao_k_hf.h5'
feri = h5py.File(fn, 'w')
feri['JK'] = np.asarray(JK_lo)
feri.close()

# Cholesky decomposition for generating 3-index density-fitted ERI (required by gw_dc)
if not os.path.isfile('cderi.h5'):
    try:
        cd = cholesky.cholesky(eri[0], tau=1e-7, dimQ=50)
    except:
        cd = cholesky.cholesky(eri[0], tau=1e-6, dimQ=50)
    cderi = cd.kernel()
    cderi = cderi.reshape(-1,nlo,nlo)
    print ('3-index ERI', cderi.shape)
    fn = 'cderi.h5'
    feri = h5py.File(fn, 'w')
    feri['cderi'] = np.asarray(cderi)
    feri.close()
else:
    fn = 'cderi.h5'
    feri = h5py.File(fn, 'r')
    cderi = np.asarray(feri['cderi'])
    feri.close()

# Compute GW double counting self-energy
naux, nimp, nimp = cderi.shape
nocc = cell.nelectron // 2
# NOTE: set ef by user for metals
ef = 0.560479331479

# NOTE: check analytic continuation stability (sigma) by user
mol = gto_mol.M()
mol.verbose = 5
mf = scf_mol.RHF(mol)
mf.max_memory = kmf.max_memory
gw = gw_dc.GWGF(mf)
gw.nmo = nimp
gw.nocc = nocc
gw.eta = 0.1/27.211386
gw.ac = 'pade'
gw.ef = ef
gw.fullsigma = True
omega = np.linspace(7./27.211386,23./27.211386,161)
sigma_lo = gw.kernel(Lpq=cderi, omega=omega, kmf=kmf, C_mo_lo=C_mo_lo, nw=100, nt=20000)
print('### local GW self-energy (trace) on real axis ###')
print('# freq  imag  real #')
for i in range(len(omega)):
    print (omega[i], np.trace(sigma_lo[:,:,i].imag), np.trace(sigma_lo[:,:,i].real))
