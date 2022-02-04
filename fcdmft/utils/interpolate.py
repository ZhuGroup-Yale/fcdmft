import numpy as np
import scipy
from pyscf.pbc.tools import k2gamma
from pyscf import lib

def interpolate_mf(mf, kpts_band, return_fock=True, C_ao_lo=None, w90=None, wigner_seitz=True,
                   veff=None, hcore=None, cell=None, dm=None, kpts=None):
    '''Get energy bands at the given (arbitrary) 'band' k-points through interpolating Fock matrix.

    Args:
        mf : mean-field object
        kpts_band : (nkpts_band,) ndarray, k-points for interpolation
        C_ao_lo : (nkpts, nao, nbands) ndarray, localized orbitals at kpts
        w90 : Wannier90 object
        wigner_seitz : use Wigner-Seitz supercell for interpolation

    Returns:
        mo_energy : (nbands,) ndarray or a list of (nbands,) ndarray
            Bands energies E_n(k)
        mo_coeff : (nao, nbands) ndarray or a list of (nao,nbands) ndarray
            Band orbitals psi_n(k)
        fock_band_lo : (nkpts_band, nbands, nbands)
    '''
    if cell is None: cell = mf.cell
    if dm is None: dm = mf.make_rdm1()
    if kpts is None: kpts = mf.kpts
    if veff is None: veff = np.array(mf.get_veff(cell, dm, kpts=kpts))
    if hcore is None: hcore = np.array(mf.get_hcore(cell, kpts))

    kpts_band = np.asarray(kpts_band)
    kpts_band = kpts_band.reshape(-1,3)

    nkpts = len(kpts)
    if C_ao_lo is not None:
        nkpts, nao, nlo = C_ao_lo.shape
    else:
        # Lowdin local orbitals
        s1e = mf.get_ovlp(cell, kpts)
        C_ao_lo = np.zeros_like(s1e)
        for k in range(nkpts):
            C_ao_lo[k] = scipy.linalg.inv(scipy.linalg.sqrtm(s1e[k]))
        nkpts, nao, nlo = C_ao_lo.shape

    # Computed Fock matrix in LO basis
    fock = veff + hcore
    fock_lo = np.zeros((nkpts,nlo,nlo),dtype=np.complex)
    for k in range(nkpts):
        fock_lo[k] = np.dot(C_ao_lo[k].T.conj(), fock[k]).dot(C_ao_lo[k])

    # Fock matrix interpolation
    if wigner_seitz:
        if w90 is None:
            from libdmet_solid.lo import pywannier90
            kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
            w90 = pywannier90.W90(mf, kmesh, nlo)
        ndegen, irvec, idx_center = get_wigner_seitz_supercell(w90)
        phase = get_phase_wigner_seitz(cell, kpts, irvec)
        phase /= np.sqrt(nkpts)
        phase_band = get_phase_wigner_seitz(cell, kpts_band, irvec)
        fock_sc = lib.einsum('Rk,kij,k->Rij', phase, fock_lo, phase[idx_center].conj())
        fock_band_lo = lib.einsum('R,Rm,Rij,m->mij', 1./ndegen, phase_band.conj(), fock_sc, phase_band[idx_center])
    else:
        scell, phase = k2gamma.get_phase(cell, kpts)
        kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        scell_band, phase_band = k2gamma.get_phase(cell, kpts_band, kmesh=kmesh)
        idx_center = kmesh[0]//2 * kmesh[1] * kmesh[2] + kmesh[1]//2 * kmesh[2] + kmesh[2]//2
        fock_sc = lib.einsum('Rk,kij,k->Rij', phase, fock_lo, phase[idx_center].conj())
        fock_band_lo = len(kpts) * lib.einsum('Rm,Rij,m->mij', phase_band.conj(), fock_sc, phase_band[idx_center])

    # Diagonalize interpolated Fock
    nkpts_band = len(kpts_band)
    mo_energy = []
    mo_coeff = []
    for k in range(nkpts_band):
        e, c = scipy.linalg.eigh(fock_band_lo[k])
        mo_energy.append(e)
        mo_coeff.append(c)

    if return_fock:
        return mo_energy, mo_coeff, fock_band_lo
    else:
        return mo_energy, mo_coeff

def interpolate_hf_diff(mf, kpts_band, fock_dft_band, mo_interpolate=False, mo_energy_hf=None, return_fock=True, C_ao_lo=None,
                        w90=None, wigner_seitz=True, veff=None, cell=None, dm=None, kpts=None):
    '''Get HF bands by interpolating the difference between HF and DFT Fock matrices.

    Args:
        mf : mean-field object
        kpts_band : (nkpts_band,) ndarray
        fock_dft_band : (nkpts_band, nbands, nbands), DFT Fock matrix in LO basis at kpts_band
        mo_interpolate : whether use HF mo_energy or veff for interpolation
        mo_energy_hf : HF/GW mo_energy

    Returns:
        mo_energy : (nbands,) ndarray or a list of (nbands,) ndarray
            Bands energies E_n(k)
        mo_coeff : (nao, nbands) ndarray or a list of (nao,nbands) ndarray
            Band orbitals psi_n(k)
        fock_band_lo : (nkpts_band, nbands, nbands)
    '''
    if cell is None: cell = mf.cell
    if dm is None: dm = mf.make_rdm1()
    if kpts is None: kpts = mf.kpts
    if veff is None: veff = np.array(mf.get_veff(cell, dm, kpts=kpts))

    kpts_band = np.asarray(kpts_band)
    kpts_band = kpts_band.reshape(-1,3)

    nkpts = len(kpts)
    if C_ao_lo is not None:
        nkpts, nao, nlo = C_ao_lo.shape
    else:
        # Lowdin local orbitals
        s1e = mf.get_ovlp(cell, kpts)
        C_ao_lo = np.zeros_like(s1e)
        for k in range(nkpts):
            C_ao_lo[k] = scipy.linalg.inv(scipy.linalg.sqrtm(s1e[k]))
        nkpts, nao, nlo = C_ao_lo.shape

    nkpts_band = len(kpts_band)
    nkpts_F, nlo_F, nlo_F = fock_dft_band.shape
    assert (nkpts_F == nkpts_band)
    assert (nlo_F == nlo)

    if not mo_interpolate:
        # Compute HF veff matrix at kpts
        from pyscf.pbc import scf
        rhf = scf.KRHF(cell, kpts, exxdiv=mf.exxdiv)
        if hasattr(mf, 'sigma'):
            rhf = scf.addons.smearing_(rhf, sigma=mf.sigma, method="fermi")
        rhf.with_df = mf.with_df
        rhf.with_df._cderi = mf.with_df._cderi
        veff_hf = np.array(rhf.get_veff(cell, dm, kpts=kpts))

        # Compute Fock difference (HF-DFT) in LO basis
        dfock = veff_hf - veff
        dfock_lo = np.zeros((nkpts,nlo,nlo),dtype=np.complex)
        for k in range(nkpts):
            dfock_lo[k] = np.dot(C_ao_lo[k].T.conj(), dfock[k]).dot(C_ao_lo[k])
    else:
        if mo_energy_hf is None:
            # Compute HF mo_energy
            from pyscf.pbc import scf
            rhf = scf.KRHF(cell, kpts, exxdiv=mf.exxdiv)
            if hasattr(mf, 'sigma'):
                rhf = scf.addons.smearing_(rhf, sigma=mf.sigma, method="fermi")
            rhf.with_df = mf.with_df
            rhf.with_df._cderi = mf.with_df._cderi
            veff_hf = np.array(rhf.get_veff(cell, dm, kpts=kpts))
            hcore = np.array(rhf.get_hcore(cell, kpts=kpts))
            fock_hf = hcore + veff_hf
            ovlp = rhf.get_ovlp()
            mo_energy_hf = []
            for k in range(nkpts):
                e, c = scipy.linalg.eigh(fock_hf[k], ovlp[k])
                mo_energy_hf.append(e)

        # Compute Fock difference (HF-DFT) in LO basis
        dfock_mo = np.zeros((nkpts,nao,nao),dtype=np.complex)
        for k in range(nkpts):
            dfock_mo[k] = np.diag(mo_energy_hf[k] - mf.mo_energy[k])
        ovlp = np.array(mf.get_ovlp())
        dfock_lo = np.zeros((nkpts,nlo,nlo),dtype=np.complex)
        for k in range(nkpts):
            CSC = np.dot(C_ao_lo[k].T.conj(), ovlp[k]).dot(mf.mo_coeff[k])
            dfock_lo[k] = np.dot(CSC, dfock_mo[k]).dot(CSC.T.conj())

    # Fock matrix difference interpolation
    if wigner_seitz:
        if w90 is None:
            from libdmet_solid.lo import pywannier90
            kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
            w90 = pywannier90.W90(mf, kmesh, nlo)
        ndegen, irvec, idx_center = get_wigner_seitz_supercell(w90)
        phase = get_phase_wigner_seitz(cell, kpts, irvec)
        phase /= np.sqrt(nkpts)
        phase_band = get_phase_wigner_seitz(cell, kpts_band, irvec)
        dfock_sc = lib.einsum('Rk,kij,k->Rij', phase, dfock_lo, phase[idx_center].conj())
        dfock_band_lo = lib.einsum('R,Rm,Rij,m->mij', 1./ndegen, phase_band.conj(), dfock_sc, phase_band[idx_center])
    else:
        scell, phase = k2gamma.get_phase(cell, kpts)
        kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        scell_band, phase_band = k2gamma.get_phase(cell, kpts_band, kmesh=kmesh)
        idx_center = kmesh[0]//2 * kmesh[1] * kmesh[2] + kmesh[1]//2 * kmesh[2] + kmesh[2]//2
        dfock_sc = lib.einsum('Rk,kij,k->Rij', phase, dfock_lo, phase[idx_center].conj())
        dfock_band_lo = len(kpts) * lib.einsum('Rm,Rij,m->mij', phase_band.conj(), dfock_sc, phase_band[idx_center])

    # Assemble and diagonalize interpolated Fock
    fock_band_lo = fock_dft_band + dfock_band_lo
    nkpts_band = len(kpts_band)
    mo_energy = []
    mo_coeff = []
    for k in range(nkpts_band):
        e, c = scipy.linalg.eigh(fock_band_lo[k])
        mo_energy.append(e)
        mo_coeff.append(c)

    if return_fock:
        return mo_energy, mo_coeff, fock_band_lo
    else:
        return mo_energy, mo_coeff

def interpolate_selfenergy(mf, kpts_band, sigma, C_ao_lo=None, w90=None, wigner_seitz=True, cell=None, kpts=None):
    '''Get self-energy at kpts_band by interpolation.

    Args:
        mf : mean-field object
        kpts_band : (nkpts_band,) ndarray
        sigma : (nkpts, nmo, nmo, nw), self-energy in MO basis at kpts

    Returns:
        sigma_band_lo : (nkpts_band, nlo, nlo, nw), self-energy in LO basis at kpts_band
    '''
    if cell is None: cell = mf.cell
    if kpts is None: kpts = mf.kpts

    kpts_band = np.asarray(kpts_band)
    kpts_band = kpts_band.reshape(-1,3)

    nkpts = len(kpts)
    if C_ao_lo is not None:
        nkpts, nao, nlo = C_ao_lo.shape
    else:
        # Lowdin local orbitals
        s1e = mf.get_ovlp(cell, kpts)
        C_ao_lo = np.zeros_like(s1e)
        for k in range(nkpts):
            C_ao_lo[k] = scipy.linalg.inv(scipy.linalg.sqrtm(s1e[k]))
        nkpts, nao, nlo = C_ao_lo.shape

    # Transform sigma from MO to LO basis
    nw = sigma.shape[-1]
    sigma_lo = np.zeros((nkpts, nlo, nlo, nw), dtype=np.complex)
    ovlp = mf.get_ovlp()
    for k in range(nkpts):
        CSC = np.dot(C_ao_lo[k].T.conj(), ovlp[k]).dot(mf.mo_coeff[k])
        sigma_lo[k] = lib.einsum('ikw,kl->ilw', lib.einsum('ij,jkw->ikw',CSC,sigma[k]), CSC.T.conj())

    # Self-energy interpolation
    if wigner_seitz:
        if w90 is None:
            from libdmet_solid.lo import pywannier90
            kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
            w90 = pywannier90.W90(mf, kmesh, nlo)
        ndegen, irvec, idx_center = get_wigner_seitz_supercell(w90)
        phase = get_phase_wigner_seitz(cell, kpts, irvec)
        phase /= np.sqrt(nkpts)
        phase_band = get_phase_wigner_seitz(cell, kpts_band, irvec)
        sigma_sc = lib.einsum('Rk,kijw,k->Rijw', phase, sigma_lo, phase[idx_center].conj())
        sigma_band_lo = lib.einsum('R,Rm,Rijw,m->mijw', 1./ndegen, phase_band.conj(), sigma_sc, phase_band[idx_center])
    else:
        scell, phase = k2gamma.get_phase(cell, kpts)
        kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        scell_band, phase_band = k2gamma.get_phase(cell, kpts_band, kmesh=kmesh)
        idx_center = kmesh[0]//2 * kmesh[1] * kmesh[2] + kmesh[1]//2 * kmesh[2] + kmesh[2]//2
        sigma_sc = lib.einsum('Rk,kijw,k->Rijw', phase, sigma_lo, phase[idx_center].conj())
        sigma_band_lo = len(kpts) * lib.einsum('Rm,Rijw,m->mijw', phase_band.conj(), sigma_sc, phase_band[idx_center])

    return sigma_band_lo


def get_bands(mf, kpts_band, cell=None, dm_kpts=None, kpts=None):
    '''Get energy bands at the given (arbitrary) 'band' k-points.

    Returns:
        mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
            Bands energies E_n(k)
        mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
            Band orbitals psi_n(k)
    '''
    if cell is None: cell = mf.cell
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if kpts is None: kpts = mf.kpts

    kpts_band = np.asarray(kpts_band)
    kpts_band = kpts_band.reshape(-1,3)

    hcore = mf.get_hcore(cell, kpts_band)
    veff = mf.get_veff(cell, dm_kpts, kpts=kpts, kpts_band=kpts_band)
    fock = hcore + veff
    s1e = mf.get_ovlp(cell, kpts_band)

    nkpts_band = len(kpts_band)
    mo_energy = []
    mo_coeff = []
    for k in range(nkpts_band):
        e, c = scipy.linalg.eigh(fock[k], s1e[k])
        mo_energy.append(e)
        mo_coeff.append(c)

    return mo_energy, mo_coeff, hcore, veff

def get_phase_wigner_seitz(cell, kpts, R_vec_rel):
    latt_vec = cell.lattice_vectors()
    R_vec_abs = np.einsum('nu, uv -> nv', R_vec_rel, latt_vec)
    phase = np.exp(1j*np.einsum('Ru, ku -> Rk', R_vec_abs, kpts))

    return phase

def get_wigner_seitz_supercell(w90, ws_search_size=[2,2,2], ws_distance_tol=1e-6):
    '''
    Adpated from pyWannier90 (https://github.com/hungpham2017/pyWannier90)

    Return a grid that contains all the lattice within the Wigner-Seitz supercell
    Ref: the hamiltonian_wigner_seitz(count_pts) in wannier90/src/hamittonian.F90
    '''

    real_metric = w90.real_lattice.T.dot(w90.real_lattice)
    dist_dim = np.prod(2 * (np.asarray(ws_search_size) + 1) + 1)
    ndegen = []
    irvec = []
    mp_grid = np.asarray(w90.mp_grid)
    n1_range =  np.arange(-ws_search_size[0] * mp_grid[0], ws_search_size[0]*mp_grid[0] + 1)
    n2_range =  np.arange(-ws_search_size[1] * mp_grid[1], ws_search_size[1]*mp_grid[1] + 1)
    n3_range =  np.arange(-ws_search_size[2] * mp_grid[2], ws_search_size[2]*mp_grid[2] + 1)
    x, y, z = np.meshgrid(n1_range, n2_range, n3_range)
    n_list = np.vstack([z.flatten('F'), x.flatten('F'), y.flatten('F')]).T
    i1 = np.arange(- ws_search_size[0] - 1, ws_search_size[0] + 2)
    i2 = np.arange(- ws_search_size[1] - 1, ws_search_size[1] + 2)
    i3 = np.arange(- ws_search_size[2] - 1, ws_search_size[2] + 2)
    x, y, z = np.meshgrid(i1, i2, i3)
    i_list = np.vstack([z.flatten('F'), x.flatten('F'), y.flatten('F')]).T

    nrpts = 0
    for n in n_list:
        # Calculate |r-R|^2
        ndiff = n - i_list * mp_grid
        dist = (ndiff.dot(real_metric).dot(ndiff.T)).diagonal()

        dist_min = dist.min()
        if abs(dist[(dist_dim + 1)//2 -1] - dist_min) < ws_distance_tol**2:
            temp = 0
            for i in range(0, dist_dim):
                if (abs(dist[i] - dist_min) < ws_distance_tol**2):
                    temp = temp + 1
            ndegen.append(temp)
            irvec.append(n.tolist())
            if (n**2).sum() < 1.e-10: rpt_origin = nrpts
            nrpts = nrpts + 1

    irvec = np.asarray(irvec)
    ndegen = np.asarray(ndegen)

    # Check the "sum rule"
    tot = np.sum(1/np.asarray(ndegen))
    assert tot - np.prod(mp_grid) < 1e-8, "Error in finding Wigner-Seitz points!!!"

    return ndegen, irvec, rpt_origin

if __name__ == '__main__':
    import numpy as np
    from pyscf.pbc import df, gto, dft, scf
    from pyscf.pbc.lib import chkfile
    import os
    import matplotlib.pyplot as plt
    from ase.lattice import bulk
    from ase.dft.kpoints import sc_special_points as special_points, get_bandpath

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

    ### Test Wannier interpolation ###
    from libdmet_solid.basis_transform import make_basis
    from libdmet_solid.lo import pywannier90
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

    points = special_points['fcc']
    G = points['G']
    X = points['X']
    W = points['W']
    K = points['K']
    L = points['L']
    band_kpts, kpath, sp_points = get_bandpath([L, G, X, W, K, G], cell.a, npoints=50)
    band_kpts = cell.get_abs_kpts(band_kpts)

    e_kn = interpolate_mf(kmf, band_kpts, C_ao_lo=C_ao_lo, w90=w90)[0]
    vbmax = -99
    for en in e_kn:
        vb_k = en[cell.nelectron//2-1]
        if vb_k > vbmax:
            vbmax = vb_k
    e_kn = [en - vbmax for en in e_kn]

    au2ev = 27.21139
    emin = -1*au2ev
    emax = 1*au2ev

    plt.figure(figsize=(5, 6))
    for n in range(nbands):
        plt.plot(kpath, [e[n]*au2ev for e in e_kn], color='#4169E1')
    for p in sp_points:
        plt.plot([p, p], [emin, emax], 'k-')
    plt.plot([0, sp_points[-1]], [0, 0], 'k-')
    plt.xticks(sp_points, ['$%s$' % n for n in ['L', r'\Gamma', 'X', 'W', 'K', r'\Gamma']])
    plt.axis(xmin=0, xmax=sp_points[-1], ymin=emin, ymax=emax)
    plt.xlabel('k-vector')
    plt.savefig('diamond_444_lda_wannier.png',dpi=600)

    ### Test IAO+PAO interpolation ###
    from libdmet_solid.system import lattice
    from libdmet_solid.basis_transform import make_basis

    Lat = lattice.Lattice(cell, kmesh)
    MINAO = {'C':'gth-szv'}
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=MINAO, full_return=True)
    C_ao_lo = C_ao_iao
    nbands = C_ao_lo.shape[-1]

    points = special_points['fcc']
    G = points['G']
    X = points['X']
    W = points['W']
    K = points['K']
    L = points['L']
    band_kpts, kpath, sp_points = get_bandpath([L, G, X, W, K, G], cell.a, npoints=50)
    band_kpts = cell.get_abs_kpts(band_kpts)

    e_kn = interpolate_mf(kmf, band_kpts, C_ao_lo=C_ao_lo)[0]
    vbmax = -99
    for en in e_kn:
        vb_k = en[cell.nelectron//2-1]
        if vb_k > vbmax:
            vbmax = vb_k
    e_kn = [en - vbmax for en in e_kn]

    au2ev = 27.21139
    emin = -1*au2ev
    emax = 1*au2ev

    plt.figure(figsize=(5, 6))
    for n in range(nbands):
        plt.plot(kpath, [e[n]*au2ev for e in e_kn], color='#4169E1')
    for p in sp_points:
        plt.plot([p, p], [emin, emax], 'k-')
    plt.plot([0, sp_points[-1]], [0, 0], 'k-')
    plt.xticks(sp_points, ['$%s$' % n for n in ['L', r'\Gamma', 'X', 'W', 'K', r'\Gamma']])
    plt.axis(xmin=0, xmax=sp_points[-1], ymin=emin, ymax=emax)
    plt.xlabel('k-vector')
    plt.savefig('diamond_444_lda_iao.png',dpi=600)
