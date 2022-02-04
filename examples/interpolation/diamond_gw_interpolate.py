import numpy as np
from pyscf.pbc import df, gto, dft, scf
from pyscf.pbc.lib import chkfile
import os
import matplotlib.pyplot as plt
from ase.lattice import bulk
from ase.dft.kpoints import sc_special_points as special_points, get_bandpath
from fcdmft.utils import interpolate

'''
GW Wannier band interpolation:
  Step 1: Get LDA bands at larger k-mesh (6x6x6) from get_bands (non-SCF diagonalization)
  Step 2: Interpolate LDA bands at random kpts_band using 6x6x6 results -> fock_dft_band
  Step 3: Interpolate GW self-energy at kpts_band using 4x4x4 results ->
          sigma_band_lo, then get GW bands

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
        verbose = 5,
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

# GW@LDA
from fcdmft.gw.pbc import krgw_gf
gw = krgw_gf.KRGWGF(kmf)
gw.eta = 1e-2
gw.fullsigma = True
gw.fc = True
omega = np.linspace(-12./27.211386, 34./27.211386, 231)
gf, gf0, sigma = gw.kernel(omega=omega, writefile=0)

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
band_kpts, kpath, sp_points = get_bandpath([L, G, X, W, K, G], cell.a, npoints=100)
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

# Interpolate GW self-energy
sigma_band_lo = interpolate.interpolate_selfenergy(kmf, band_kpts, sigma, C_ao_lo=C_ao_lo, w90=w90)

# Get GW bands at band_kpts through interpolating mo diff
e_kn_2 = interpolate.interpolate_hf_diff(kmf, band_kpts, fock_dft_band, mo_interpolate=True,
                                         mo_energy_hf=gw.mo_energy, C_ao_lo=C_ao_lo, w90=w90)[0]
vbmax = -99
for en in e_kn_2:
    vb_k = en[cell.nelectron//2-1]
    if vb_k > vbmax:
        vbmax = vb_k
e_kn_2 = [en - vbmax for en in e_kn_2]

# plot
nkpts_band, nlo, nlo, nw = sigma_band_lo.shape
delta = 0.2/27.211386
dens = np.zeros((nkpts_band, nw))
for iw in range(nw):
    for k in range(nkpts_band):
        gf_band = np.linalg.inv((omega[iw]+1j*delta)*np.eye(nbands)-fock_dft_band[k]-sigma_band_lo[k,:,:,iw])
        dens[k,iw] = -1./np.pi*np.trace(gf_band.imag)/27.211386
dens = dens.T

omega_new = omega
for iw in range(len(omega_new)):
    omega_new[iw] -= gw.mo_energy[0][3]

space_max = 0.
for i in range(1,nkpts_band):
    if kpath[i]-kpath[i-1] > space_max:
        space_max = kpath[i]-kpath[i-1]
k_max = (nkpts_band-1)*space_max
kpath_new = np.linspace(0,k_max,nkpts_band)

sp_points_new = np.zeros_like(sp_points)
for i in range(len(sp_points)):
    for j in range(nkpts_band):
        if sp_points[i] == kpath[j]:
            sp_points_new[i] = kpath_new[j]

xi = kpath_new
yi = omega_new*27.211386
fig = plt.subplots(figsize=(3.375,1.8))
ax = plt.subplot(111)
interpolation_method = 'gaussian'
extent = (xi[0],xi[-1],yi[0],yi[-1])
plt.imshow(dens, cmap='viridis', aspect=0.11, extent=extent, origin='lower', interpolation=interpolation_method)
cbar = plt.colorbar()
cbar.set_label('DOS [1/eV]', size=6)
cbar.ax.tick_params(labelsize=6)

emin = yi[0]
emax = yi[-1]
ax.axis(xmin=0, xmax=sp_points_new[-1], ymin=emin, ymax=emax)
plt.xticks(sp_points_new, ['%s' % n for n in ['L', r'$\Gamma$', 'X', 'W', 'K', r'$\Gamma$']])

for p in sp_points_new:
    ax.plot([p, p], [emin, emax], '-', color='k', linewidth=0.6)
ax.plot([0, sp_points_new[-1]], [0, 0], '--', color='k',linewidth=0.6)

au2ev = 27.211386
for n in range(nbands):
    ax.plot(kpath_new, [e[n]*au2ev for e in e_kn_2], '--',dashes=([2.2,1.4]),alpha=0.6,linewidth=0.6,color='white')

ax.set_ylabel('Energy(eV)',fontsize=9)
ax.yaxis.labelpad = -1

major_yticks=np.arange(-25,20.1,5.0)
#minor_yticks=np.arange(-25,20.1,1.0)
ax.set_yticks(major_yticks)
#ax.set_yticks(minor_yticks,minor=True)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)
ax.tick_params(axis='both',which='major',labelsize=8)
ax.tick_params(axis='both',which='major',length=2.4,width=0.4,pad=2)
ax.tick_params(axis='both',which='minor',length=1.5,width=0.4)

plt.subplots_adjust(left=0.08,right=0.99,bottom=0.10,top=0.95)
plt.savefig('diamond_444_GW.png', dpi=600)
