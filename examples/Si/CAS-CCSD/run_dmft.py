'''
Main routine to set up DMFT parameters and run DMFT
'''

try:
    import block2
    from block2.su2 import MPICommunicator
    dmrg_ = True
except:
    dmrg_ = False
    pass
import numpy as np
import scipy, os, h5py
from fcdmft.utils import write
from fcdmft.dmft import gwdmft
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

def dmft_abinitio():
    '''
    List of DMFT parameters

    gw_dmft : choose to run GW+DMFT (True) or HF+DMFT (False)
    opt_mu : whether to optimize chemical potential during DMFT cycles
    solver_type : choose impurity solver ('cc', 'ucc', 'dmrg', 'dmrgsz', 'fci')
    disc_type : choose bath discretization method ('opt', 'direct', 'linear', 'gauss', 'log')
    max_memory : maximum memory for DMFT calculation (per MPI process for CC, per node for DMRG)
    dmft_max_cycle : maximum number of DMFT iterations (set to 0 for one-shot DMFT)
    chkfile : chkfile for saving DMFT self-consistent quantities (hyb and self-energy)
    diag_only : choose to only fit diagonal hybridization (optional)
    orb_fit : special orbitals (e.g. 3d/4f) with x5 weight in bath optimization (optional)
    delta : broadening for discretizing hybridization (often 0.02-0.1 Ha)
    nbath : number of bath energies (can be any integer)
    nb_per_e : number of bath orbitals per bath energy (should be no greater than nval-ncore)
                total bath number = nb_per_e * nbath
    mu : initial chemical potential
    nval : number of valence (plus core) impurity orbitals (only ncore:nval orbs coupled to bath)
    ncore : number of core impurity orbitals
    nelectron : electron number per cell
    gmres_tol : GMRES/GCROTMK convergence criteria for solvers in production run (often 1e-3)
    wl0, wh0: (optional) real-axis frequency range [wl0+mu, wh0+mu] for bath discretization
                in DMFT self-consistent iterations (defualt: -0.4, 0.4)
    wl, wh : real-axis frequnecy range [wl, wh] for production run
    eta : spectral broadening for production run (often 0.1-0.4 eV)
    twist_average : whether to use twist-averaged HF/GW for lattice GF and hybridization
    band_interpolation : whether to use interpolated HF/GW for lattice GF and hybridization
    '''
    # DMFT self-consistent loop parameters
    gw_dmft = True
    opt_mu = False
    solver_type = 'cc'
    disc_type = 'opt'
    max_memory = 32000
    dmft_max_cycle = 10
    chkfile = 'DMFT_chk.h5'
    diag_only = False
    orb_fit = None
    twist_average = False
    band_interpolate = False

    delta = 0.1
    mu = 0.267
    nbath = 12
    nb_per_e = 8
    wl0 = -0.4
    wh0 = 0.4

    nval = 8
    ncore = 0
    nelectron = 8

    # DMFT production run parameters
    Ha2eV = 27.211386
    wl = 2./Ha2eV
    wh = 13./Ha2eV
    eta = 0.1/Ha2eV
    gmres_tol = 1e-3
    # final self-energy on imag or real axis
    run_imagfreq = False

    '''
    load_mf : bool
        Load DMFT imp+bath mean-field object from saved file. if True, dmft_max_cycle = 0.
    save_mf : bool
        Save DMFT imp+bath mean-field object to saved file at the end of DMFT cycles.
        load_mf and save_mf cannot be True at the same time.
    '''
    load_mf = False
    save_mf = True

    '''
    specific parameters for CAS treatment of impurity problem:
        cas : use CASCI or not (default: False)
        casno : natural orbital method for CASCI
                (choices: 'gw': GW@HF, 'cc': CCSD, 'ci': CISD, 'hf' : HF)
        composite : whether to use GW or CCSD Green's function as the low-level GF
                    for impurity problem; if False, use HF Green's function as low-level GF
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-4.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh`.
        nocc_act : int
            Number of occupied NOs to keep. Default is None. If present, overrides `thresh` and `vno_only`.
        save_gf : bool
            Save CAS Green's function. Default is False.
        read_gf : bool
            Read saved CAS Green's function and skip solving CAS problem. Default is False.
        thresh2 : float
            Threshold on NO occupation numbers for a larger CAS problem solved
            by a cheaper impurity solver. Default is None.
        load_cas : bool
            Load DMFT CAS problem mean-field object from saved file (only support CCSD-NO based CAS).
    '''
    cas = True
    casno = 'gw'
    composite = False
    thresh = 1e-3
    nvir_act = None
    nocc_act = None
    save_gf = False
    read_gf = False
    thresh2 = None

    # spin-unrestricted
    nvir_act_a = None
    nocc_act_a = None
    nvir_act_b = None
    nocc_act_b = None

    load_cas = False

    # specific parameters for DMRG solvers (see fcdmft/solver/gfdmrg.py for detailed comments)
    gs_n_steps = 20
    gf_n_steps = 6
    gs_tol = 1E-10
    gf_tol = 1E-3
    gs_bond_dims = [400] * 5 + [800] * 5 + [1500] * 5 + [2000] * 5
    gs_noises = [1E-3] * 7 + [1E-4] * 5 + [1e-7] * 5 + [0]
    gf_bond_dims = [200] * 2 + [500] * 4
    gf_noises = [1E-4] * 1 + [1E-5] * 1 + [1E-7] * 1 + [0]
    dmrg_gmres_tol = 1E-7
    dmrg_verbose = 2
    reorder_method = 'gaopt'
    dmrg_local = True
    n_off_diag_cg = -2
    extra_nw = 5
    extra_dw = 0.1/Ha2eV
    # if extra_delta is not None, approx. DMRG treatment of freq will be used
    extra_delta = None
    # at lease one of 'load_dir' and 'save_dir' should be None
    load_dir = None
    save_dir = './gs_mps'

    # DMRG-MRCI parameters
    dyn_corr_method = None
    nocc_act_low = None # number of lowest occ orbs treated by MRCI
    nvir_act_high = None # number of highest vir orbs treated by MRCI

    ### Finishing parameter settings ###

    # read hcore
    fn = 'hcore_JK_iao_k_dft.h5'
    feri = h5py.File(fn, 'r')
    hcore_k = np.asarray(feri['hcore'])
    feri.close()

    # read HF-JK matrix
    fn = 'hcore_JK_iao_k_hf.h5'
    feri = h5py.File(fn, 'r')
    JK_k = np.asarray(feri['JK'])
    feri.close()

    # read density matrix
    fn = 'DM_iao_k.h5'
    feri = h5py.File(fn, 'r')
    DM_k = np.asarray(feri['DM'])
    feri.close()

    # read 4-index ERI
    fn = 'eri_imp111_iao.h5'
    feri = h5py.File(fn, 'r')
    eri = np.asarray(feri['eri'])
    feri.close()
    eri_new = eri
    if eri_new.shape[0] == 3:
        eri_new = np.zeros_like(eri)
        eri_new[0] = eri[0]
        eri_new[1] = eri[2]
        eri_new[2] = eri[1]
    del eri

    # Read interpolated Fock for more accurate hybridization
    if band_interpolate:
        fn = 'hcore_JK_iao_k_dft_band.h5'
        feri = h5py.File(fn, 'r')
        hcore_k_band = np.asarray(feri['hcore'])
        JK_k_dft_band = np.asarray(feri['JK'])
        feri.close()
        if hcore_k_band.ndim == 3:
            hcore_k_band = hcore_k_band[np.newaxis, ...]
            JK_k_dft_band = JK_k_dft_band[np.newaxis, ...]

    # Read twist average Fock for more accurate hybridization
    if twist_average:
        from pyscf import lib
        mesh = [0,1]
        center_list = lib.cartesian_prod((mesh, mesh, mesh))[1:]
        spin, nkpts, nao, nao = hcore_k.shape
        hcore_k_TA = np.zeros((len(center_list), spin, nkpts, nao, nao), dtype=np.complex)
        JK_k_TA = np.zeros((len(center_list), spin, nkpts, nao, nao), dtype=np.complex)
        for i in range(len(center_list)):
            center = center_list[i]

            # read hcore
            fn = 'hcore_JK_iao_k_dft_%d_%d_%d.h5'%(center[0],center[1],center[2])
            feri = h5py.File(fn, 'r')
            hcore_k_TA[i] = np.asarray(feri['hcore'])
            feri.close()

            # read HF-JK matrix
            fn = 'hcore_JK_iao_k_hf_%d_%d_%d.h5'%(center[0],center[1],center[2])
            feri = h5py.File(fn, 'r')
            JK_k_TA[i] = np.asarray(feri['JK'])
            feri.close()

        hcore_k_TA  = hcore_k_TA.transpose(1,0,2,3,4).reshape(spin, len(center_list)*nkpts, nao, nao)
        JK_k_TA  = JK_k_TA.transpose(1,0,2,3,4).reshape(spin, len(center_list)*nkpts, nao, nao)

    assert (not (band_interpolate and twist_average))

    # run self-consistent DMFT
    mydmft = gwdmft.DMFT(hcore_k, JK_k, DM_k, eri_new, nval, ncore, nbath,
                       nb_per_e, disc_type=disc_type, solver_type=solver_type)
    mydmft.gw_dmft = gw_dmft
    mydmft.verbose = 5
    mydmft.diis = True
    mydmft.gmres_tol = gmres_tol
    mydmft.max_memory = max_memory
    mydmft.chkfile = chkfile
    mydmft.diag_only = diag_only
    mydmft.orb_fit = orb_fit
    mydmft.twist_average = twist_average
    mydmft.band_interpolate = band_interpolate
    if twist_average:
        mydmft.center_list = center_list
        mydmft.hcore_k_band = hcore_k_TA
        mydmft.JK_k_band = JK_k_TA
    if band_interpolate:
        mydmft.hcore_k_band = hcore_k_band
        mydmft.JK_k_dft_band = JK_k_dft_band

    assert (not (load_mf and save_mf))
    if load_mf:
        dmft_max_cycle = 0
    mydmft.max_cycle = dmft_max_cycle
    mydmft.run_imagfreq = run_imagfreq
    if solver_type == 'dmrg' or solver_type == 'dmrgsz':
        if not dmrg_:
            raise ImportError
    mydmft.load_mf = load_mf
    mydmft.save_mf = save_mf

    if cas:
        mydmft.cas = cas
        mydmft.casno = casno
        mydmft.composite = composite
        mydmft.thresh = thresh
        mydmft.thresh2 = thresh2
        mydmft.nvir_act = nvir_act
        mydmft.nocc_act = nocc_act
        mydmft.save_gf = save_gf
        mydmft.read_gf = read_gf
        if casno == 'gw':
            assert(gw_dmft)
        if eri_new.shape[0] == 3:
            mydmft.nvir_act_a = nvir_act_a
            mydmft.nocc_act_a = nocc_act_a
            mydmft.nvir_act_b = nvir_act_b
            mydmft.nocc_act_b = nocc_act_b
        mydmft.load_cas = load_cas

    if solver_type == 'dmrg' or solver_type == 'dmrgsz':
        mydmft.gs_n_steps = gs_n_steps
        mydmft.gf_n_steps = gf_n_steps
        mydmft.gs_tol = gs_tol
        mydmft.gf_tol = gf_tol
        mydmft.gs_bond_dims = gs_bond_dims
        mydmft.gs_noises = gs_noises
        mydmft.gf_bond_dims = gf_bond_dims
        mydmft.gf_noises = gf_noises
        mydmft.dmrg_gmres_tol = dmrg_gmres_tol
        mydmft.dmrg_verbose = dmrg_verbose
        mydmft.reorder_method = reorder_method
        mydmft.n_off_diag_cg = n_off_diag_cg
        mydmft.load_dir = load_dir
        mydmft.save_dir = save_dir
        mydmft.dmrg_local = dmrg_local
        mydmft.dyn_corr_method = dyn_corr_method
        mydmft.nvir_act_high = nvir_act_high
        mydmft.nocc_act_low = nocc_act_low
        if nocc_act_low is not None:
            assert (nocc_act_low <= nocc_act)
        if nvir_act_high is not None:
            assert (nvir_act_high <= nvir_act)

    mydmft.kernel(mu0=mu, wl=wl0, wh=wh0, delta=delta, occupancy=nelectron, opt_mu=opt_mu)
    occupancy = np.trace(mydmft.get_rdm_imp())
    if rank == 0:
        print ('At mu =', mydmft.mu, ', occupancy =', occupancy)

    mydmft.verbose = 5
    mydmft._scf.mol.verbose = 5
    spin = mydmft.spin

    if not mydmft.run_imagfreq:
        # extra_freqs and extra_delta only for DMRG solver
        if extra_delta is not None and (mydmft.solver_type=='dmrg' or mydmft.solver_type=='dmrgsz'):
            nw = int(round((wh-wl)/(extra_dw * extra_nw)))+1
            freqs = np.linspace(wl, wh, nw)
            extra_freqs = []
            for i in range(len(freqs)):
                freqs_tmp = []
                if extra_nw % 2 == 0:
                    for w in range(-extra_nw // 2, extra_nw // 2):
                        freqs_tmp.append(freqs[i] + extra_dw * w)
                else:
                    for w in range(-(extra_nw-1) // 2, (extra_nw+1) // 2):
                        freqs_tmp.append(freqs[i] + extra_dw * w)
                extra_freqs.append(np.array(freqs_tmp))
            mydmft.extra_freqs = extra_freqs
            mydmft.extra_delta = extra_delta
            freqs_comp = np.array(extra_freqs).reshape(-1)
        else:
            nw = int(round((wh-wl)/eta))+1
            freqs = np.linspace(wl, wh, nw)
            freqs_comp = freqs

        # Get impurity DOS (production run)
        #ldos = mydmft.get_ldos_imp(freqs, eta)
 
        # Get lattice DOS (production run)
        ldos, ldos_gw = mydmft.get_ldos_latt(freqs, eta)
        spin = mydmft.spin
 
        filename = 'mu-%0.3f_n-%0.2f_%d-%d_eta-%.2f_d-%.2f_%s'%(
                    mu,occupancy,nval,nbath,eta*Ha2eV,delta,solver_type)
        if rank == 0:
            write.write_dos(filename, freqs_comp, ldos, occupancy=occupancy)
 
        if mydmft.gw_dmft:
            filename = 'mu-%0.3f_n-%0.2f_%d-%d_eta-%.2f_d-%.2f_gw'%(
                        mu,occupancy,nval,nbath,eta*Ha2eV,delta)
        else:
            filename = 'mu-%0.3f_n-%0.2f_%d-%d_eta-%.2f_d-%.2f_hf'%(
                        mu,occupancy,nval,nbath,eta*Ha2eV,delta)
        if rank == 0:
            write.write_dos(filename, freqs_comp, ldos_gw, occupancy=occupancy)

    else:
        nimp = eri_new.shape[-1]
        omega_ns = np.linspace(0./27.211386, 10.0/27.211386, 21)[1:]
        if solver_type == 'dmrg' or solver_type == 'dmrgsz':
            # TODO: check DMRG load_dir is correct
            if mydmft.load_dir is None:
                mydmft.load_dir = mydmft.save_dir
                mydmft.save_dir = None
            assert(mydmft.load_dir is not None)
            sigma = np.zeros((spin,nimp,nimp,len(omega_ns)),dtype=complex)
            for iw in range(len(omega_ns)):
                sigma[:,:,:,iw] = mydmft.get_sigma_imp(np.array([mu]), omega_ns[iw],
                                                       save_gf=save_gf, read_gf=read_gf)[:,:nimp,:nimp,0]
        else:
            sigma = mydmft.get_sigma_imp(mu+1j*omega_ns, 0.0,
                                         save_gf=save_gf, read_gf=read_gf)[:,:nimp,:nimp]

        tmpdir = 'dmft_dos'
        if rank == 0:
            if not os.path.isdir(tmpdir):
                os.mkdir(tmpdir)
            for i in range(nimp):
                write.write_sigma_elem(tmpdir+'/dmft_sigma_imag_orb-%d'%(i), omega_ns, sigma[:,i,i,:])

if __name__ == '__main__':
    dmft_abinitio()
