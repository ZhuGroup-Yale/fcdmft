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
from fcdmft.utils import interpolate

log.verbose = 'DEBUG1'

einsum = lib.einsum

def get_kgw_sigma_diff(freqs, eta):
    from fcdmft.gw.pbc import krgw_gf
    '''
    Get k-point GW-AC self-energy in MO basis
    sigma = v_hf + sigma_c - v_xc
    '''
    fn = 'ac_coeff.h5'
    feri = h5py.File(fn, 'r')
    coeff = np.asarray(feri['coeff'])
    ef = np.asarray(feri['fermi'])
    omega_fit = np.asarray(feri['omega_fit'])
    feri.close()

    fn = 'vxc.h5'
    feri = h5py.File(fn, 'r')
    vk = np.array(feri['vk'])
    v_mf = np.array(feri['v_mf'])
    feri.close()

    nkpts, nao, nao = vk.shape
    nw = len(freqs)
    sigma = np.zeros([nkpts,nao,nao,nw], dtype=np.complex)
    for k in range(nkpts):
        for p in range(nao):
            for q in range(nao):
                sigma[k,p,q] = krgw_gf.pade_thiele(freqs-ef+1j*eta, omega_fit, coeff[k,:,p,q])
                sigma[k,p,q] += vk[k,p,q] - v_mf[k,p,q]

    return sigma

def get_gf(hcore, sigma, freqs, delta):
    nw  = len(freqs)
    nao = hcore.shape[0]
    gf = np.zeros([nao, nao, nw], np.complex128)
    for iw, w in enumerate(freqs):
        gf[:,:,iw] = np.linalg.inv((w+1j*delta)*np.eye(nao)-hcore-sigma[:,:,iw])
    return gf

def get_gf0(hcore, freqs, delta):
    nw  = len(freqs)
    nao = hcore.shape[0]
    gf = np.zeros([nao, nao, nw], np.complex128)
    for iw, w in enumerate(freqs):
        gf[:,:,iw] = np.linalg.inv((w+1j*delta)*np.eye(nao)-hcore)
    return gf

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
           max_memory = 140000,
           verbose = 5,
           pseudo = 'gth-pade',
           basis='gth-dzvp-molopt-sr',
           precision=1e-12)

# save cell object (required for using band_interpolate in DMFT)
scf.chkfile.save_cell(cell, 'cell.chk')

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
# save kmf object (required for using band_interpolate in DMFT)
os.system('cp %s kmf.chk'%(chkfname))
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

from libdmet_solid.lo.iao import reference_mol, get_labels, get_idx_each
import copy
### Set up IAO Basis ###
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

# construct IAO and PAO.
C_ao_iao, C_ao_iao_val, C_ao_iao_virt, C_ao_iao_core \
        = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao, full_return=True, \
        pmol_val=pmol_val, pmol_core=pmol_core, tol=1e-9)
C_ao_lo = C_ao_iao[:,:,ncore:]
nlo = C_ao_lo.shape[-1]

### Band Interpolation ###
# get DFT bands at kpts_L, this kpts_L will be used for computing lattice GF/hyb in DMFT
# Note: Running DFT (LDA/GGA) only needs diagonal k-block of GDF (j_only=True)
kmesh_L = [12,12,12]
kpts_L = cell.make_kpts(kmesh_L,scaled_center=[0,0,0])
gdf2 = df.GDF(cell, kpts)
gdf2_fname = 'gdf_ints_555_band.h5'
gdf2._cderi_to_save = gdf2_fname
gdf2.auxbasis = gdf.auxbasis
gdf2.kpts_band = kpts_L
if not os.path.isfile(gdf2_fname):
    gdf2._j_only = True
    gdf2.build(j_only=True, kpts_band=kpts_L)

kmf2 = dft.KRKS(cell, kpts).density_fit()
kmf2.xc = 'lda'
kmf2.exxdiv = None
kmf2 = scf.addons.smearing_(kmf2, sigma=5e-3, method="fermi")
kmf2.with_df = gdf2
kmf2.with_df._cderi = gdf2_fname
kmf2.mo_energy = kmf.mo_energy
kmf2.mo_occ = kmf.mo_occ
kmf2.mo_coeff = kmf.mo_coeff

mo_energy_band, mo_coeff_band, hcore_band, veff_band = interpolate.get_bands(kmf2, kpts_L)

# set up a new mean-field object at kpts_L
kmf3 = dft.KRKS(cell, kpts_L).density_fit()
kmf3.xc = 'lda'
kmf3.exxdiv = None
kmf3 = scf.addons.smearing_(kmf3, sigma=5e-3, method="fermi")
kmf3.mo_energy = mo_energy_band
kmf3.mo_coeff = mo_coeff_band
kmf3.mo_occ = kmf3.get_occ(mo_energy_kpts=mo_energy_band, mo_coeff_kpts=mo_coeff_band)

# IAO+PAO orbitals at kpts_L
Lat2 = lattice.Lattice(cell, kmesh)
Lat2.kpts = kpts_L
Lat2.set_val_virt_core(nval, nvirt, ncore)
C_ao_iao, C_ao_iao_val, C_ao_iao_virt, C_ao_iao_core \
        = make_basis.get_C_ao_lo_iao(Lat2, kmf3, minao=minao, full_return=True, \
        pmol_val=pmol_val, pmol_core=pmol_core, tol=1e-9)
C_ao_lo_L = C_ao_iao[:,:,ncore:]

# Make sure two sets of wannier orbitals have same order and phase
ovlp = np.dot(C_ao_lo[0].T.conj(), kmf.get_ovlp()[0]).dot(C_ao_lo_L[0])
C_ao_lo_L_ordered = np.zeros_like(C_ao_lo_L)
for i in range(ovlp.shape[0]):
    idx = np.argmax(np.abs(ovlp[i]))
    C_ao_lo_L_ordered[:,:,i] = C_ao_lo_L[:,:,idx]
    if (ovlp[i,idx].real < 0):
        C_ao_lo_L_ordered[:,:,i] = -C_ao_lo_L_ordered[:,:,i]
C_ao_lo_L = C_ao_lo_L_ordered

# Get DFT Fock at kpts_L in LO basis
fock_band = hcore_band + veff_band
hcore_dft_band = np.zeros((len(kpts_L),nlo,nlo),dtype=np.complex)
veff_dft_band = np.zeros((len(kpts_L),nlo,nlo),dtype=np.complex)
for k in range(len(kpts_L)):
    hcore_dft_band[k] = np.dot(C_ao_lo_L[k].T.conj(), hcore_band[k]).dot(C_ao_lo_L[k])
    veff_dft_band[k] = np.dot(C_ao_lo_L[k].T.conj(), veff_band[k]).dot(C_ao_lo_L[k])
fock_dft_band = hcore_dft_band + veff_dft_band

# Save DFT get_bands results
fn = 'hcore_JK_iao_k_dft_band.h5'
feri = h5py.File(fn, 'w')
feri['hcore'] = np.asarray(hcore_dft_band)
feri['JK'] = np.asarray(veff_dft_band)
feri['kpts'] = np.asarray(kpts_L)
feri['C_ao_lo'] = np.asarray(C_ao_lo_L)
feri.close()

# Interpolate GW self-energy (this part will be rerun in DMFT, can be skipped here)
freqs = np.linspace(0.2, 0.9, 141)
delta = 0.005
sigma = get_kgw_sigma_diff(freqs, delta)
sigma_band_lo = interpolate.interpolate_selfenergy(kmf, kpts_L, sigma, C_ao_lo=C_ao_lo)

nkpts_band = len(kpts_L)
gf = np.zeros((nlo, nlo, len(freqs)), dtype=np.complex)
for k in range(nkpts_band):
    gf += 1./nkpts_band * get_gf(fock_dft_band[k], sigma_band_lo[k], freqs, delta)

print ('GW t2g, eg')
for i in range(len(freqs)):
    print (freqs[i], -gf[1,1,i].imag/np.pi, -gf[3,3,i].imag/np.pi)

print ('GW total')
for i in range(len(freqs)):
    print (freqs[i], -np.trace(gf[:,:,i]).imag/np.pi)

nkpts_band = len(kpts_L)
gf = np.zeros((nlo, nlo, len(freqs)), dtype=np.complex)
for k in range(nkpts_band):
    gf += 1./nkpts_band * get_gf0(fock_dft_band[k], freqs, delta)

print ('DFT t2g, eg')
for i in range(len(freqs)):
    print (freqs[i], -gf[1,1,i].imag/np.pi, -gf[3,3,i].imag/np.pi)

print ('DFT total')
for i in range(len(freqs)):
    print (freqs[i], -np.trace(gf[:,:,i]).imag/np.pi)

