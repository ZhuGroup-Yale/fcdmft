import numpy as np
from pyscf.pbc import df, gto, dft, scf
from pyscf.pbc.lib import chkfile
import os
import matplotlib.pyplot as plt
from ase.lattice import bulk
from ase.dft.kpoints import sc_special_points as special_points, get_bandpath
from fcdmft.utils import interpolate

'''
HF Wannier band interpolation:
  Step 1: Get LDA bands at larger k-mesh (6x6x6) from get_bands (non-SCF diagonalization)
  Step 2: Interpolate LDA bands at random kpts_band using 6x6x6 results -> fock_dft_band
  Step 3: Interpolate difference between HF and LDA (deltaFock) at kpts_band using 4x4x4 results ->
          dfock_band_lo, then get HF bands: fock_dft_band + dfock_band_lo

Note: Currently a k-mesh must be used for constructing Wannier orbitals
'''

cell = gto.Cell()
cell.build(unit = 'angstrom',
        a = np.array([[0.000000, 1.783500, 1.783500],
             [1.783500, 0.000000, 1.783500],
             [1.783500, 1.783500, 0.000000]]),
        atom = 'C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
        dimension = 3,
        max_memory = 32000,
        verbose = 4,
        pseudo = 'gth-pade',
        basis='gth-dzv',
        precision=1e-10)

kmesh = [4,4,4]
kpts = cell.make_kpts(kmesh,scaled_center=[0,0,0])
gdf = df.GDF(cell, kpts)
gdf_fname = 'gdf_ints_444.h5'
gdf._cderi_to_save = gdf_fname
if not os.path.isfile(gdf_fname):
    gdf.build()

chkfname = 'diamond_444.chk'
if os.path.isfile(chkfname):
    kmf = dft.KRKS(cell, kpts)
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    data = chkfile.load(chkfname, 'scf')
    kmf.__dict__.update(data)
else:
    kmf = dft.KRKS(cell, kpts)
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.conv_tol = 1e-12
    kmf.chkfile = chkfname
    kmf.kernel()

from libdmet_solid.basis_transform import make_basis
from libdmet_solid.lo import pywannier90
# valence Wannier orbitals
num_wann = 8
keywords = \
'''
num_iter = 1000
begin projections
C:sp3
end projections
exclude_bands : 9-%s
num_cg_steps = 100
precond = T
'''%(kmf.cell.nao_nr())

w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords)
#w90.use_atomic = True
#w90.use_bloch_phases = True
#w90.use_scdm = True
#w90.guiding_centres = False
w90.kernel()

C_ao_mo = np.asarray(w90.mo_coeff)[:, :, w90.band_included_list]
C_mo_lo = make_basis.tile_u_matrix(np.array(w90.U_matrix.transpose(2, 0, 1), \
    order='C'), u_virt=None, u_core=None)
C_ao_lo = make_basis.multiply_basis(C_ao_mo, C_mo_lo)
nbands = C_ao_lo.shape[-1]

# set up band_kpts
points = special_points['fcc']
G = points['G']
X = points['X']
W = points['W']
K = points['K']
L = points['L']
band_kpts, kpath, sp_points = get_bandpath([L, G, X, W, K, G], cell.a, npoints=50)
band_kpts = cell.get_abs_kpts(band_kpts)

# get DFT bands for a large k-mesh
kmesh_L = [6,6,6]
kpts_L = cell.make_kpts(kmesh_L,scaled_center=[0,0,0])
gdf2 = df.GDF(cell, kpts)
gdf2_fname = 'gdf_ints_large.h5'
gdf2._cderi_to_save = gdf2_fname
gdf2.auxbasis = gdf.auxbasis
gdf2.kpts_band = kpts_L
if not os.path.isfile(gdf2_fname):
    gdf2._j_only = True
    gdf2.build(j_only=True, kpts_band=kpts_L)

kmf2 = dft.KRKS(cell, kpts).density_fit()
kmf2.xc = kmf.xc
kmf2.exxdiv = kmf.exxdiv
if hasattr(kmf, 'sigma'):
    kmf2 = scf.addons.smearing_(kmf2, sigma=kmf.sigma, method="fermi")
kmf2.with_df = gdf2
kmf2.with_df._cderi = gdf2_fname
kmf2.mo_energy = kmf.mo_energy
kmf2.mo_occ = kmf.mo_occ
kmf2.mo_coeff = kmf.mo_coeff

mo_energy_L, mo_coeff_L, hcore_L, veff_L = interpolate.get_bands(kmf2, kpts_L)

# set up a new mean-field object at large k-mesh
kmf3 = dft.KRKS(cell, kpts_L).density_fit()
kmf3.xc = kmf.xc
kmf3.exxdiv = kmf.exxdiv
if hasattr(kmf, 'sigma'):
    kmf3 = scf.addons.smearing_(kmf3, sigma=kmf.sigma, method="fermi")
kmf3.mo_energy = mo_energy_L
kmf3.mo_coeff = mo_coeff_L
kmf3.mo_occ = kmf3.get_occ(mo_energy_kpts=mo_energy_L, mo_coeff_kpts=mo_coeff_L)

# Wannier basis for large k-mesh
w90_L = pywannier90.W90(kmf3, kmesh_L, num_wann, other_keywords = keywords)
w90_L.kernel()
C_ao_mo = np.asarray(w90_L.mo_coeff)[:, :, w90_L.band_included_list]
C_mo_lo = make_basis.tile_u_matrix(np.array(w90_L.U_matrix.transpose(2, 0, 1), \
    order='C'), u_virt=None, u_core=None)
C_ao_lo_L = make_basis.multiply_basis(C_ao_mo, C_mo_lo)

# Make sure two sets of wannier orbitals have same order and phase
# Search for Gamma point in kpts_L
idx_G = None
for i in range(len(kpts_L)):
    if np.linalg.norm(kpts_L[i]-kpts[0]) < 1e-8:
        idx_G = i
        break
assert (idx_G is not None)
ovlp = np.dot(C_ao_lo[0].T.conj(), kmf.get_ovlp()[0]).dot(C_ao_lo_L[idx_G])
C_ao_lo_L_ordered = np.zeros_like(C_ao_lo_L)
for i in range(ovlp.shape[0]):
    idx = np.argmax(np.abs(ovlp[i]))
    C_ao_lo_L_ordered[:,:,i] = C_ao_lo_L[:,:,idx]
    if (ovlp[i,idx].real < 0):
        C_ao_lo_L_ordered[:,:,i] = -C_ao_lo_L_ordered[:,:,i]
C_ao_lo_L = C_ao_lo_L_ordered

# Get DFT Fock at band_kpts through interpolation
mo_energy_band, mo_coeff_band, fock_dft_band = interpolate.interpolate_mf(kmf3, band_kpts, C_ao_lo=C_ao_lo_L,
                                                                          veff=veff_L, hcore=hcore_L, w90=w90_L)

# Get HF bands at band_kpts through interpolating veff diff
e_kn = interpolate.interpolate_hf_diff(kmf, band_kpts, fock_dft_band, C_ao_lo=C_ao_lo, w90=w90)[0]
vbmax = -99
for en in e_kn:
    vb_k = en[cell.nelectron//2-1]
    if vb_k > vbmax:
        vbmax = vb_k
e_kn = [en - vbmax for en in e_kn]

# Get HF bands at band_kpts through interpolating mo diff
e_kn_2 = interpolate.interpolate_hf_diff(kmf, band_kpts, fock_dft_band, mo_interpolate=True, C_ao_lo=C_ao_lo, w90=w90)[0]
vbmax = -99
for en in e_kn_2:
    vb_k = en[cell.nelectron//2-1]
    if vb_k > vbmax:
        vbmax = vb_k
e_kn_2 = [en - vbmax for en in e_kn_2]

au2ev = 27.21139
emin = -1.2*au2ev
emax = 1.2*au2ev

plt.figure(figsize=(5, 6))
for n in range(nbands):
    plt.plot(kpath, [e[n]*au2ev for e in e_kn_2], color='orange', alpha=0.7)
    plt.plot(kpath, [e[n]*au2ev for e in e_kn], color='#4169E1')
for p in sp_points:
    plt.plot([p, p], [emin, emax], 'k-')
plt.plot([0, sp_points[-1]], [0, 0], 'k-')
plt.xticks(sp_points, ['$%s$' % n for n in ['L', r'\Gamma', 'X', 'W', 'K', r'\Gamma']])
plt.axis(xmin=0, xmax=sp_points[-1], ymin=emin, ymax=emax)
plt.xlabel('k-vector')
plt.savefig('diamond_444_hf.png',dpi=600)
