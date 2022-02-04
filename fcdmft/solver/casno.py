from pyscf import lib, gto, scf, mp, cc, ao2mo
import pyscf
import numpy, scipy, copy, h5py
import numpy as np
from pyscf.lib import logger
from fcdmft.gw.mol import gw_gf
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

einsum = lib.einsum

def make_casno_mp(mp, thresh=1e-4, nvir_act=None, nocc_act=None, vno_only=False,
                  ea_no=False, ip_no=False, return_dm=False, get_cas_mo=True,
                  local=False):
    '''
    MP2 frozen natural orbitals for CASCI calculation

    Attributes:
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-4.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh`.
        nocc_act : int
            Number of occupied NOs to keep. Default is None. If present, overrides `thresh` and `vno_only`.
        ea_no : bool
            Include negatively charged density matrix for making NOs.
        ip_no : bool
            Include positively charged density matrix for making NOs.
        vno_only : bool
            Only construct virtual natural orbitals. Default is False.
        return_rdm : bool
            Return correlated density matrix. Default is False.
        get_cas_mo : bool
            Diagonalize CAS Hamiltonian to get mo_coeff and mo_energy. Default is True.

    Returns:
        mf_cas : mean-field object with all integrals in NO basis.
        no_coeff : ndarray
            Semicanonical NO coefficients in the AO basis
        dm : ndarray, correlated density matrix in MO basis (optional).
    '''
    mf = mp._scf
    dm = None
    if rank == 0:
        dm = mp.make_rdm1()
    comm.Barrier()
    dm = comm.bcast(dm, root=0)
    nmo = mp.nmo
    nocc = mp.nocc

    if ea_no:
        mol_ea = copy.copy(mf.mol)
        mol_ea.charge = -1
        mol_ea.spin = 1
        mol_ea.verbose = 0
        mol_ea.build()

        mf_ea = scf.UHF(mol_ea)
        mf_ea.kernel()
        mp_ea = pyscf.mp.UMP2(mf_ea)
        mp_ea.kernel()
        dm_ea = mp_ea.make_rdm1()[0]
        CSC = np.dot(mf.mo_coeff.T, np.dot(mf.get_ovlp(), mf_ea.mo_coeff[0]))
        dm_ea = np.dot(CSC, np.dot(dm_ea, CSC.T))
        ne_ea = np.trace(dm_ea[nocc:,nocc:]-0.5*dm[nocc:,nocc:])
        dm[nocc:,nocc:] = 0.5 * dm[nocc:,nocc:] + dm_ea[nocc:,nocc:]

    if ip_no:
        mol_ip = copy.copy(mf.mol)
        mol_ip.charge = 1
        mol_ip.spin = 1
        mol_ip.verbose = 0
        mol_ip.build()

        mf_ip = scf.UHF(mol_ip)
        mf_ip.kernel()
        mp_ip = pyscf.mp.UMP2(mf_ip)
        mp_ip.kernel()
        dm_ip = mp_ip.make_rdm1()[1]
        CSC = np.dot(mf.mo_coeff.T, np.dot(mf.get_ovlp(), mf_ip.mo_coeff[1]))
        dm_ip = np.dot(CSC, np.dot(dm_ip, CSC.T))
        ne_ip = np.trace(dm_ip[:nocc,:nocc]-0.5*dm[:nocc,:nocc])
        dm[:nocc,:nocc] = 0.5 * dm[:nocc,:nocc] + dm_ip[:nocc,:nocc]

    no_occ_v, no_coeff_v = np.linalg.eigh(dm[nocc:,nocc:])
    no_occ_v = np.flip(no_occ_v)
    no_coeff_v = np.flip(no_coeff_v, axis=1)
    if rank == 0:
        logger.info(mf, 'Full no_occ_v = \n %s', no_occ_v)
    if nocc_act is not None:
        vno_only = False
    if not vno_only:
        no_occ_o, no_coeff_o = np.linalg.eigh(dm[:nocc,:nocc])
        no_occ_o = np.flip(no_occ_o)
        no_coeff_o = np.flip(no_coeff_o, axis=1)
        if rank == 0:
            logger.info(mf, 'Full no_occ_o = \n %s', no_occ_o)

    if nvir_act is None and nocc_act is None:
        no_idx_v = np.where(no_occ_v > thresh)[0]
        if not vno_only:
            no_idx_o = np.where(2-no_occ_o > thresh)[0]
        else:
            no_idx_o = range(0, nocc)
    elif nvir_act is None and nocc_act is not None:
        no_idx_v = range(0, nmo-nocc)
        no_idx_o = range(nocc-nocc_act, nocc)
    elif nvir_act is not None and nocc_act is None:
        no_idx_v = range(0, nvir_act)
        no_idx_o = range(0, nocc)
    else:
        no_idx_v = range(0, nvir_act)
        no_idx_o = range(nocc-nocc_act, nocc)

    # semi-canonicalization
    fvv = numpy.diag(mf.mo_energy[nocc:])
    fvv_no = numpy.dot(no_coeff_v.T, numpy.dot(fvv, no_coeff_v))
    no_vir = len(no_idx_v)
    _, v_canon_v = numpy.linalg.eigh(fvv_no[:no_vir,:no_vir])
    if not vno_only:
        foo = numpy.diag(mf.mo_energy[:nocc])
        foo_no = numpy.dot(no_coeff_o.T, numpy.dot(foo, no_coeff_o))
        no_occ = nocc - len(no_idx_o)
        _, v_canon_o = numpy.linalg.eigh(foo_no[no_occ:,no_occ:])

    no_coeff_v = numpy.dot(mf.mo_coeff[:,nocc:], numpy.dot(no_coeff_v[:,:no_vir], v_canon_v))
    if not vno_only:
        no_coeff_o = numpy.dot(mf.mo_coeff[:,:nocc], numpy.dot(no_coeff_o[:,no_occ:], v_canon_o))

    if not vno_only:
        ne_sum = np.sum(no_occ_o[no_idx_o]) + np.sum(no_occ_v[no_idx_v])
        n_no = len(no_idx_o) + len(no_idx_v)
        if rank == 0:
            logger.info(mf, 'CAS no_occ_o = \n %s, \n no_occ_v = \n %s', no_occ_o[no_idx_o], no_occ_v[no_idx_v])
    else:
        ne_sum = np.trace(dm[:nocc,:nocc]) + np.sum(no_occ_v[no_idx_v])
        n_no = nocc + len(no_idx_v)
        if rank == 0:
            logger.info(mf, 'CAS mo_occ_o = \n %s, \n no_occ_v = \n %s', dm[:nocc,:nocc].diagonal(), no_occ_v[no_idx_v])
    if ea_no:
        ne_sum -= ne_ea
    if ip_no:
        ne_sum -= ne_ip
    nelectron = int(round(ne_sum))
    if rank == 0:
        logger.info(mf, 'CAS norb = %s, nelec = %s, ne_no = %s', n_no, nelectron, ne_sum)

    if not vno_only:
        if local:
            no_coeff_o = scdm(no_coeff_o, np.eye(no_coeff_o.shape[0]))
            no_coeff_v = scdm(no_coeff_v, np.eye(no_coeff_v.shape[0]))
        no_coeff = np.concatenate((no_coeff_o, no_coeff_v), axis=1)
    else:
        if local:
            no_coeff_o = scdm(mf.mo_coeff[:,:nocc], np.eye(mf.mo_coeff[:,:nocc].shape[0]))
            no_coeff_v = scdm(no_coeff_v, np.eye(no_coeff_v.shape[0]))
            no_coeff = np.concatenate((no_coeff_o, no_coeff_v), axis=1)
        else:
            no_coeff = np.concatenate((mf.mo_coeff[:,:nocc], no_coeff_v), axis=1)

    # new mf object for CAS
    mol_cas = gto.M()
    mol_cas.nelectron = nelectron
    mol_cas.verbose = mf.mol.verbose
    mol_cas.symmetry = 'c1'
    mol_cas.max_memory = mf.max_memory
    mol_cas.incore_anyway = True
    mf_cas = scf.RHF(mol_cas)

    # compute CAS integrals
    h1e = np.dot(no_coeff.T, np.dot(mf.get_hcore(), no_coeff))
    g2e = ao2mo.restore(8, ao2mo.kernel(mf._eri, no_coeff), n_no)

    dm_hf = mf.make_rdm1()
    ovlp = mf.get_ovlp()
    CS = np.dot(no_coeff.T, ovlp)
    dm_cas_no = np.dot(CS, np.dot(dm_hf, CS.T))
    JK_cas_no = _get_veff(dm_cas_no, g2e)[0]
    JK_full_no = np.dot(no_coeff.T, np.dot(mf.get_veff(), no_coeff))
    h1e = h1e + JK_full_no - JK_cas_no
    h1e = 0.5 * (h1e + h1e.T)
    comm.Barrier()

    h1e = comm.bcast(h1e, root=0)
    if rank == 0:
        fn = 'cas_g2e.h5'
        feri = h5py.File(fn, 'w')
        feri['g2e'] = np.asarray(g2e)
        feri.close()
    comm.Barrier()
    if rank > 0:
        fn = 'cas_g2e.h5'
        feri = h5py.File(fn, 'r')
        g2e = np.asarray(feri['g2e'])
        feri.close()
    comm.Barrier()

    # set up integrals for mf_cas
    mf_cas.get_hcore = lambda *args: h1e
    mf_cas.get_ovlp = lambda *args: np.eye(n_no)
    mf_cas._eri = g2e
    if get_cas_mo:
        if rank == 0:
            mf_cas.max_cycle = 1
            mf_cas.kernel(dm_cas_no)
        comm.Barrier()
        mf_cas.mo_occ = comm.bcast(mf_cas.mo_occ, root=0)
        mf_cas.mo_energy = comm.bcast(mf_cas.mo_energy, root=0)
        mf_cas.mo_coeff = comm.bcast(mf_cas.mo_coeff, root=0)
    else:
        # fake mo_coeff and mo_energy
        mf_cas.mo_coeff = np.eye(n_no)
        mf_cas.mo_energy = np.zeros(n_no)
        mf_cas.mo_occ = np.zeros(n_no)
    no_coeff = comm.bcast(no_coeff, root=0)

    if get_cas_mo:
        if return_dm:
            if ip_no:
                dm[:nocc,:nocc] = 2. * (dm[:nocc,:nocc] - dm_ip[:nocc,:nocc])
            if ea_no:
                dm[nocc:,nocc:] = 2. * (dm[nocc:,nocc:] - dm_ea[nocc:,nocc:])
            return mf_cas, no_coeff, dm
        else:
            return mf_cas, no_coeff
    else:
        if return_dm:
            if ip_no:
                dm[:nocc,:nocc] = 2. * (dm[:nocc,:nocc] - dm_ip[:nocc,:nocc])
            if ea_no:
                dm[nocc:,nocc:] = 2. * (dm[nocc:,nocc:] - dm_ea[nocc:,nocc:])
            return mf_cas, no_coeff, h1e + JK_cas_no, dm_cas_no, dm
        else:
            return mf_cas, no_coeff, h1e + JK_cas_no, dm_cas_no

def make_casno_gw(gw, thresh=1e-4, nvir_act=None, nocc_act=None, vno_only=False,
                  ea_cut=None, ip_cut=None, ea_no=None, ip_no=None, return_dm=False,
                  get_cas_mo=True, local=False):
    '''
    GW frozen natural orbitals for CASCI calculation

    Attributes:
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-4.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh`.
        nocc_act : int
            Number of occupied NOs to keep. Default is None. If present, overrides `thresh` and `vno_only`.
        ea_cut : float
            Energy cutoff for determining number of EA charged density matrices.
        ip_cut : float
            Energy cutoff for determining number of IP charged density matrices.
        ea_no : int
            Number of negatively charged density matrices included for making NOs.
        ip_no : int
            Number of positively charged density matrices included for making NOs.
        vno_only : bool
            Only construct virtual natural orbitals. Default is False.
        return_rdm : bool
            Return correlated density matrix. Default is False.
        get_cas_mo : bool
            Diagonalize CAS Hamiltonian to get mo_coeff and mo_energy. Default is True.

    Returns:
        mf_cas : mean-field object with all integrals in NO basis.
        no_coeff : ndarray
            Semicanonical NO coefficients in the AO basis
        dm : ndarray, correlated density matrix in MO basis (optional).
    '''
    mf = gw._scf
    dm = None
    if rank == 0:
        dm = gw.make_rdm1()
    comm.Barrier()
    dm = comm.bcast(dm, root=0)
    dm_gs = dm.copy()
    nmo = gw.nmo
    nocc = gw.nocc
    assert(abs(np.trace(dm_gs)-2.*gw.nocc) < 1e-3 * 2.*gw.nocc)

    if ea_cut is not None and ea_no is None:
        ea_no = np.count_nonzero(gw.mo_energy[nocc:] < gw.mo_energy[nocc] + ea_cut)
    if ip_cut is not None and ip_no is None:
        ip_no = np.count_nonzero(gw.mo_energy[:nocc] > gw.mo_energy[nocc-1] - ip_cut)

    if ea_no is not None:
        assert(gw.omega_emo)
        if not isinstance(ea_no, int):
            ea_no = 1
        gf = gw.gf[:,:,:nmo]
        eta = gw.eta
        ne_ea = 0.
        for i in range(nocc,nocc+ea_no):
            dm[nocc:,nocc:] += -gf[nocc:,nocc:,i].imag * eta / ea_no
            ne_ea += -np.trace(gf[nocc:,nocc:,i].imag) * eta / ea_no

    if ip_no is not None:
        assert(gw.omega_emo)
        if not isinstance(ip_no, int):
            ip_no = 1
        gf = gw.gf[:,:,:nmo]
        eta = gw.eta
        ne_ip = 0.
        for i in range(nocc-ip_no,nocc):
            dm[:nocc,:nocc] += gf[:nocc,:nocc,i].imag * eta / ip_no
            ne_ip += np.trace(gf[:nocc,:nocc,i].imag) * eta / ip_no

    no_occ_v, no_coeff_v = np.linalg.eigh(dm[nocc:,nocc:])
    no_occ_v = np.flip(no_occ_v)
    no_coeff_v = np.flip(no_coeff_v, axis=1)
    if rank == 0:
        logger.info(mf, 'Full no_occ_v = \n %s', no_occ_v)
    if nocc_act is not None:
        vno_only = False
    if not vno_only:
        no_occ_o, no_coeff_o = np.linalg.eigh(dm[:nocc,:nocc])
        no_occ_o = np.flip(no_occ_o)
        no_coeff_o = np.flip(no_coeff_o, axis=1)
        if rank == 0:
            logger.info(mf, 'Full no_occ_o = \n %s', no_occ_o)

    if nvir_act is None and nocc_act is None:
        no_idx_v = np.where(no_occ_v > thresh)[0]
        if not vno_only:
            no_idx_o = np.where(2-no_occ_o > thresh)[0]
        else:
            no_idx_o = range(0, nocc)
    elif nvir_act is None and nocc_act is not None:
        no_idx_v = range(0, nmo-nocc)
        no_idx_o = range(nocc-nocc_act, nocc)
    elif nvir_act is not None and nocc_act is None:
        no_idx_v = range(0, nvir_act)
        no_idx_o = range(0, nocc)
    else:
        no_idx_v = range(0, nvir_act)
        no_idx_o = range(nocc-nocc_act, nocc)

    # semi-canonicalization
    fvv = numpy.diag(mf.mo_energy[nocc:])
    fvv_no = numpy.dot(no_coeff_v.T, numpy.dot(fvv, no_coeff_v))
    no_vir = len(no_idx_v)
    _, v_canon_v = numpy.linalg.eigh(fvv_no[:no_vir,:no_vir])
    if not vno_only:
        foo = numpy.diag(mf.mo_energy[:nocc])
        foo_no = numpy.dot(no_coeff_o.T, numpy.dot(foo, no_coeff_o))
        no_occ = nocc - len(no_idx_o)
        _, v_canon_o = numpy.linalg.eigh(foo_no[no_occ:,no_occ:])

    no_coeff_v = numpy.dot(mf.mo_coeff[:,nocc:], numpy.dot(no_coeff_v[:,:no_vir], v_canon_v))
    if not vno_only:
        no_coeff_o = numpy.dot(mf.mo_coeff[:,:nocc], numpy.dot(no_coeff_o[:,no_occ:], v_canon_o))

    if not vno_only:
        ne_sum = np.sum(no_occ_o[no_idx_o]) + np.sum(no_occ_v[no_idx_v])
        n_no = len(no_idx_o) + len(no_idx_v)
        if rank == 0:
            logger.info(mf, 'CAS no_occ_o = \n %s, \n no_occ_v = \n %s', no_occ_o[no_idx_o], no_occ_v[no_idx_v])
    else:
        ne_sum = np.trace(dm[:nocc,:nocc]) + np.sum(no_occ_v[no_idx_v])
        n_no = nocc + len(no_idx_v)
        if rank == 0:
            logger.info(mf, 'CAS mo_occ_o = \n %s, \n no_occ_v = \n %s', dm[:nocc,:nocc].diagonal(), no_occ_v[no_idx_v])
    if ea_no:
        ne_sum -= ne_ea
    if ip_no:
        ne_sum -= ne_ip
    nelectron = int(round(ne_sum))
    if rank == 0:
        logger.info(mf, 'CAS norb = %s, nelec = %s, ne_no = %s', n_no, nelectron, ne_sum)

    if not vno_only:
        if local:
            no_coeff_o = scdm(no_coeff_o, np.eye(no_coeff_o.shape[0]))
            no_coeff_v = scdm(no_coeff_v, np.eye(no_coeff_v.shape[0]))
        no_coeff = np.concatenate((no_coeff_o, no_coeff_v), axis=1)
    else:
        if local:
            no_coeff_o = scdm(mf.mo_coeff[:,:nocc], np.eye(mf.mo_coeff[:,:nocc].shape[0]))
            no_coeff_v = scdm(no_coeff_v, np.eye(no_coeff_v.shape[0]))
            no_coeff = np.concatenate((no_coeff_o, no_coeff_v), axis=1)
        else:
            no_coeff = np.concatenate((mf.mo_coeff[:,:nocc], no_coeff_v), axis=1)

    # new mf object for CAS
    mol_cas = gto.M()
    mol_cas.nelectron = nelectron
    mol_cas.verbose = mf.mol.verbose
    mol_cas.symmetry = 'c1'
    mol_cas.max_memory = mf.max_memory
    mol_cas.incore_anyway = True
    mf_cas = scf.RHF(mol_cas)

    # compute CAS integrals
    h1e = np.dot(no_coeff.T, np.dot(mf.get_hcore(), no_coeff))
    g2e = ao2mo.restore(8, ao2mo.kernel(mf._eri, no_coeff), n_no)

    dm_hf = mf.make_rdm1()
    ovlp = mf.get_ovlp()
    CS = np.dot(no_coeff.T, ovlp)
    dm_cas_no = np.dot(CS, np.dot(dm_hf, CS.T))
    JK_cas_no = _get_veff(dm_cas_no, g2e)[0]
    JK_full_no = np.dot(no_coeff.T, np.dot(mf.get_veff(), no_coeff))
    h1e = h1e + JK_full_no - JK_cas_no
    h1e = 0.5 * (h1e + h1e.T)
    comm.Barrier()

    h1e = comm.bcast(h1e, root=0)
    if rank == 0:
        fn = 'cas_g2e.h5'
        feri = h5py.File(fn, 'w')
        feri['g2e'] = np.asarray(g2e)
        feri.close()
    comm.Barrier()
    if rank > 0:
        fn = 'cas_g2e.h5'
        feri = h5py.File(fn, 'r')
        g2e = np.asarray(feri['g2e'])
        feri.close()
    comm.Barrier()

    # set up integrals for mf_cas
    mf_cas.get_hcore = lambda *args: h1e
    mf_cas.get_ovlp = lambda *args: np.eye(n_no)
    mf_cas._eri = g2e
    if get_cas_mo:
        if rank == 0:
            mf_cas.max_cycle = 1
            mf_cas.kernel(dm_cas_no)
        comm.Barrier()
        mf_cas.mo_occ = comm.bcast(mf_cas.mo_occ, root=0)
        mf_cas.mo_energy = comm.bcast(mf_cas.mo_energy, root=0)
        mf_cas.mo_coeff = comm.bcast(mf_cas.mo_coeff, root=0)
    else:
        # fake mo_coeff and mo_energy
        mf_cas.mo_coeff = np.eye(n_no)
        mf_cas.mo_energy = np.zeros(n_no)
        mf_cas.mo_occ = np.zeros(n_no)
    no_coeff = comm.bcast(no_coeff, root=0)

    if get_cas_mo:
        if return_dm:
            return mf_cas, no_coeff, dm_gs
        else:
            return mf_cas, no_coeff
    else:
        if return_dm:
            return mf_cas, no_coeff, h1e + JK_cas_no, dm_cas_no, dm_gs
        else:
            return mf_cas, no_coeff, h1e + JK_cas_no, dm_cas_no

def make_casno_cc(mycc, thresh=1e-4, nvir_act=None, nocc_act=None, vno_only=False,
                  ea_cut=None, ip_cut=None, ea_no=None, ip_no=None, return_dm=False,
                  qp_cutoff=0.1, get_cas_mo=True, local=False, load_cc=False,
                  save_fcidump=False, nocc_act_low=None, nvir_act_high=None):
    '''
    CCSD frozen natural orbitals for CASCI calculation

    Attributes:
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-4.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh`.
        nocc_act : int
            Number of occupied NOs to keep. Default is None. If present, overrides `thresh` and `vno_only`.
        ea_cut : float
            Energy cutoff for determining number of EA charged density matrices.
        ip_cut : float
            Energy cutoff for determining number of IP charged density matrices.
        ea_no : int
            Number of negatively charged density matrices included for making NOs.
        ip_no : int
            Number of positively charged density matrices included for making NOs.
        vno_only : bool
            Only construct virtual natural orbitals. Default is False.
        return_rdm : bool
            Return correlated density matrix. Default is False.
        qp_cutoff : float
            Quasiparticle weight cutoff for using EOM-CCSD density matrices. Default is 0.1.
        get_cas_mo : bool
            Diagonalize CAS Hamiltonian to get mo_coeff and mo_energy. Default is True.
        local : bool
            Use localized natural orbitals. Default is False.
        load_cc : bool
            Load saved CCSD amplitudes. Default is False.
        nvir_act_high : int
            Number of highest virtual NOs. Default is None. If present, separately localize from other nvir_act.
        nocc_act_low : int
            Number of lowest occupied NOs. Default is None. If present, separately localize from other nocc_act.

    Returns:
        mf_cas : mean-field object with all integrals in NO basis.
        no_coeff : ndarray
            Semicanonical NO coefficients in the AO basis
        dm : ndarray, correlated density matrix in MO basis (optional).
    '''
    from pyscf.cc.eom_rccsd import vector_to_amplitudes_ip, vector_to_amplitudes_ea

    mf = mycc._scf
    if not load_cc:
        if rank == 0:
            mycc.solve_lambda()
            fn = 'amplitudes_casno.h5'
            feri = h5py.File(fn, 'w')
            feri['t1'] = np.asarray(mycc.t1)
            feri['t2'] = np.asarray(mycc.t2)
            feri['l1'] = np.asarray(mycc.l1)
            feri['l2'] = np.asarray(mycc.l2)
            feri.close()
        comm.Barrier()
        if rank > 0:
            fn = 'amplitudes_casno.h5'
            feri = h5py.File(fn, 'r')
            mycc.t1 = np.asarray(feri['t1'])
            mycc.t2 = np.asarray(feri['t2'])
            mycc.l1 = np.asarray(feri['l1'])
            mycc.l2 = np.asarray(feri['l2'])
            feri.close()
        comm.Barrier()
    else:
        fn = 'amplitudes_casno.h5'
        feri = h5py.File(fn, 'r')
        mycc.t1 = np.asarray(feri['t1'])
        mycc.t2 = np.asarray(feri['t2'])
        mycc.l1 = np.asarray(feri['l1'])
        mycc.l2 = np.asarray(feri['l2'])
        feri.close()

    dm = mycc.make_rdm1()
    dm_gs = dm.copy()
    nmo = mycc.nmo
    nocc = mycc.nocc
    if rank == 0:
        mycc.verbose = 4
    else:
        mycc.verbose = 0

    if ea_cut is not None and ea_no is None:
        ea_no = np.count_nonzero(mf.mo_energy[nocc:] < mf.mo_energy[nocc] + ea_cut)
    if ip_cut is not None and ip_no is None:
        ip_no = np.count_nonzero(mf.mo_energy[:nocc] > mf.mo_energy[nocc-1] - ip_cut)

    if ea_no is not None:
        if not isinstance(ea_no, int):
            ea_no = 1
        eea, cea = mycc.eaccsd(nroots=ea_no, koopmans=True)
        ea_count = 0
        for i in range(ea_no):
            r1, r2 = vector_to_amplitudes_ea(cea[i], nmo, nocc)
            qp_weight = np.linalg.norm(r1)**2
            if qp_weight > qp_cutoff:
                ea_count += 1
        ne_ea = 0.
        for i in range(ea_no):
            r1, r2 = vector_to_amplitudes_ea(cea[i], nmo, nocc)
            qp_weight = np.linalg.norm(r1)**2
            if qp_weight > qp_cutoff:
                dm_ea = np.outer(r1, r1)
                # TODO: need to check this
                dm_ea += einsum('ica, icb -> ab', r2, r2)
                dm[nocc:,nocc:] += dm_ea / ea_count
                ne_ea += np.trace(dm_ea) / ea_count
        if rank == 0:
            logger.info(mf, 'State average EA density matrices = %s', ea_count)

    if ip_no is not None:
        if not isinstance(ip_no, int):
            ip_no = 1
        eip, cip = mycc.ipccsd(nroots=ip_no, koopmans=True)
        ip_count = 0
        for i in range(ip_no):
            r1, r2 = vector_to_amplitudes_ip(cip[i], nmo, nocc)
            qp_weight = np.linalg.norm(r1)**2
            if qp_weight > qp_cutoff:
                ip_count += 1
        ne_ip = 0.
        for i in range(ip_no):
            r1, r2 = vector_to_amplitudes_ip(cip[i], nmo, nocc)
            qp_weight = np.linalg.norm(r1)**2
            if qp_weight > qp_cutoff:
                dm_ip = np.outer(r1, r1)
                # TODO: need to check this
                dm_ip += einsum('ija, ika -> jk', r2, r2)
                dm[:nocc,:nocc] -= dm_ip / ip_count
                ne_ip -= np.trace(dm_ip) / ip_count
        if rank == 0:
            logger.info(mf, 'State average IP density matrices = %s', ip_count)

    no_occ_v, no_coeff_v = np.linalg.eigh(dm[nocc:,nocc:])
    no_occ_v = np.flip(no_occ_v)
    no_coeff_v = np.flip(no_coeff_v, axis=1)
    if rank == 0:
        logger.info(mf, 'Full no_occ_v = \n %s', no_occ_v)
    if nocc_act is not None:
        vno_only = False
    if not vno_only:
        no_occ_o, no_coeff_o = np.linalg.eigh(dm[:nocc,:nocc])
        no_occ_o = np.flip(no_occ_o)
        no_coeff_o = np.flip(no_coeff_o, axis=1)
        if rank == 0:
            logger.info(mf, 'Full no_occ_o = \n %s', no_occ_o)

    if nvir_act is None and nocc_act is None:
        no_idx_v = np.where(no_occ_v > thresh)[0]
        if not vno_only:
            no_idx_o = np.where(2-no_occ_o > thresh)[0]
        else:
            no_idx_o = range(0, nocc)
    elif nvir_act is None and nocc_act is not None:
        no_idx_v = range(0, nmo-nocc)
        no_idx_o = range(nocc-nocc_act, nocc)
    elif nvir_act is not None and nocc_act is None:
        no_idx_v = range(0, nvir_act)
        no_idx_o = range(0, nocc)
    else:
        no_idx_v = range(0, nvir_act)
        no_idx_o = range(nocc-nocc_act, nocc)

    # semi-canonicalization
    fvv = numpy.diag(mf.mo_energy[nocc:])
    fvv_no = numpy.dot(no_coeff_v.T, numpy.dot(fvv, no_coeff_v))
    no_vir = len(no_idx_v)
    _, v_canon_v = numpy.linalg.eigh(fvv_no[:no_vir,:no_vir])
    if not vno_only:
        foo = numpy.diag(mf.mo_energy[:nocc])
        foo_no = numpy.dot(no_coeff_o.T, numpy.dot(foo, no_coeff_o))
        no_occ = nocc - len(no_idx_o)
        _, v_canon_o = numpy.linalg.eigh(foo_no[no_occ:,no_occ:])

    no_coeff_v = numpy.dot(mf.mo_coeff[:,nocc:], numpy.dot(no_coeff_v[:,:no_vir], v_canon_v))
    if not vno_only:
        no_coeff_o = numpy.dot(mf.mo_coeff[:,:nocc], numpy.dot(no_coeff_o[:,no_occ:], v_canon_o))

    if not vno_only:
        ne_sum = np.sum(no_occ_o[no_idx_o]) + np.sum(no_occ_v[no_idx_v])
        n_no = len(no_idx_o) + len(no_idx_v)
        if rank == 0:
            logger.info(mf, 'CAS no_occ_o = \n %s, \n no_occ_v = \n %s', no_occ_o[no_idx_o], no_occ_v[no_idx_v])
    else:
        ne_sum = np.trace(dm[:nocc,:nocc]) + np.sum(no_occ_v[no_idx_v])
        n_no = nocc + len(no_idx_v)
        if rank == 0:
            logger.info(mf, 'CAS mo_occ_o = \n %s, \n no_occ_v = \n %s', dm[:nocc,:nocc].diagonal(), no_occ_v[no_idx_v])
    if ea_no:
        ne_sum -= ne_ea
    if ip_no:
        ne_sum -= ne_ip
    nelectron = int(round(ne_sum))
    if rank == 0:
        logger.info(mf, 'CAS norb = %s, nelec = %s, ne_no = %s', n_no, nelectron, ne_sum)

    if not vno_only:
        if local:
            if nocc_act_low is not None and nocc_act_low > 0 and nocc_act_low < nocc_act:
                no_coeff_o_low = scdm(no_coeff_o[:,:nocc_act_low], np.eye(no_coeff_o.shape[0]))
                no_coeff_o_high = scdm(no_coeff_o[:,nocc_act_low:], np.eye(no_coeff_o.shape[0]))
                no_coeff_o = np.concatenate((no_coeff_o_low, no_coeff_o_high), axis=1)
            else:
                no_coeff_o = scdm(no_coeff_o, np.eye(no_coeff_o.shape[0]))
            if nvir_act_high is not None and nvir_act_high > 0 and nvir_act_high < nvir_act:
                no_coeff_v_low = scdm(no_coeff_v[:,:(nvir_act-nvir_act_high)], np.eye(no_coeff_v.shape[0]))
                no_coeff_v_high = scdm(no_coeff_v[:,(nvir_act-nvir_act_high):], np.eye(no_coeff_v.shape[0]))
                no_coeff_v = np.concatenate((no_coeff_v_low, no_coeff_v_high), axis=1)
            else:
                no_coeff_v = scdm(no_coeff_v, np.eye(no_coeff_v.shape[0]))
        no_coeff = np.concatenate((no_coeff_o, no_coeff_v), axis=1)
    else:
        if local:
            if nocc_act_low is not None and nocc_act_low > 0 and nocc_act_low < nocc_act:
                no_coeff_o_low = scdm(mf.mo_coeff[:,:nocc_act_low], np.eye(mf.mo_coeff[:,:nocc].shape[0]))
                no_coeff_o_high = scdm(mf.mo_coeff[:,nocc_act_low:], np.eye(mf.mo_coeff[:,:nocc].shape[0]))
                no_coeff_o = np.concatenate((no_coeff_o_low, no_coeff_o_high), axis=1)
            else:
                no_coeff_o = scdm(mf.mo_coeff[:,:nocc], np.eye(mf.mo_coeff[:,:nocc].shape[0]))
            if nvir_act_high is not None and nvir_act_high > 0 and nvir_act_high < nvir_act:
                no_coeff_v_low = scdm(no_coeff_v[:,:(nvir_act-nvir_act_high)], np.eye(no_coeff_v.shape[0]))
                no_coeff_v_high = scdm(no_coeff_v[:,(nvir_act-nvir_act_high):], np.eye(no_coeff_v.shape[0]))
                no_coeff_v = np.concatenate((no_coeff_v_low, no_coeff_v_high), axis=1)
            else:
                no_coeff_v = scdm(no_coeff_v, np.eye(no_coeff_v.shape[0]))
            no_coeff = np.concatenate((no_coeff_o, no_coeff_v), axis=1)
        else:
            no_coeff = np.concatenate((mf.mo_coeff[:,:nocc], no_coeff_v), axis=1)

    # new mf object for CAS
    mol_cas = gto.M()
    mol_cas.nelectron = nelectron
    mol_cas.verbose = mf.mol.verbose
    mol_cas.symmetry = 'c1'
    mol_cas.max_memory = mf.max_memory
    mol_cas.incore_anyway = True
    mf_cas = scf.RHF(mol_cas)

    # compute CAS integrals
    h1e = np.dot(no_coeff.T, np.dot(mf.get_hcore(), no_coeff))
    g2e = ao2mo.restore(8, ao2mo.kernel(mf._eri, no_coeff), n_no)

    dm_hf = mf.make_rdm1()
    ovlp = mf.get_ovlp()
    CS = np.dot(no_coeff.T, ovlp)
    dm_cas_no = np.dot(CS, np.dot(dm_hf, CS.T))
    JK_cas_no = _get_veff(dm_cas_no, g2e)[0]
    JK_full_no = np.dot(no_coeff.T, np.dot(mf.get_veff(), no_coeff))
    h1e = h1e + JK_full_no - JK_cas_no
    h1e = 0.5 * (h1e + h1e.T)
    comm.Barrier()

    h1e = comm.bcast(h1e, root=0)
    if rank == 0:
        fn = 'cas_g2e.h5'
        feri = h5py.File(fn, 'w')
        feri['g2e'] = np.asarray(g2e)
        feri.close()
    comm.Barrier()
    if rank > 0:
        fn = 'cas_g2e.h5'
        feri = h5py.File(fn, 'r')
        g2e = np.asarray(feri['g2e'])
        feri.close()
    comm.Barrier()

    # set up integrals for mf_cas
    mf_cas.get_hcore = lambda *args: h1e
    mf_cas.get_ovlp = lambda *args: np.eye(n_no)
    mf_cas._eri = g2e
    if get_cas_mo:
        if rank == 0:
            mf_cas.max_cycle = 1
            mf_cas.kernel(dm_cas_no)
        comm.Barrier()
        mf_cas.mo_occ = comm.bcast(mf_cas.mo_occ, root=0)
        mf_cas.mo_energy = comm.bcast(mf_cas.mo_energy, root=0)
        mf_cas.mo_coeff = comm.bcast(mf_cas.mo_coeff, root=0)
    else:
        # fake mo_coeff and mo_energy
        mf_cas.mo_coeff = np.eye(n_no)
        mf_cas.mo_energy = np.zeros(n_no)
        mf_cas.mo_occ = np.zeros(n_no)
    no_coeff = comm.bcast(no_coeff, root=0)

    if rank == 0:
        if save_fcidump:
            from pyscf import tools
            tools.fcidump.from_integrals('FCIDUMP', h1e, ao2mo.restore(1,g2e,n_no),
                                        n_no, nelectron, ms=0)
    comm.Barrier()

    mycc.verbose = mf.verbose
    if get_cas_mo:
        if return_dm:
            return mf_cas, no_coeff, dm_gs
        else:
            return mf_cas, no_coeff
    else:
        if return_dm:
            return mf_cas, no_coeff, h1e + JK_cas_no, dm_cas_no, dm_gs
        else:
            return mf_cas, no_coeff, h1e + JK_cas_no, dm_cas_no

def make_casno_ucc(mycc, thresh=1e-4, nvir_act_a=None, nocc_act_a=None, nvir_act_b=None,
                   nocc_act_b=None, vno_only=False, return_dm=False, get_cas_mo=True,
                   local=False, load_cc=False):
    '''
    UCCSD frozen natural orbitals for CASCI calculation

    Attributes:
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-4.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh`.
        nocc_act : int
            Number of occupied NOs to keep. Default is None. If present, overrides `thresh` and `vno_only`.
        vno_only : bool
            Only construct virtual natural orbitals. Default is False.
        return_rdm : bool
            Return correlated density matrix. Default is False.
        get_cas_mo : bool
            Diagonalize CAS Hamiltonian to get mo_coeff and mo_energy. Default is True.
        local : bool
            Use localized natural orbitals. Default is False.
        load_cc : bool
            Load saved CCSD amplitudes. Default is False.

    Returns:
        mf_cas : mean-field object with all integrals in NO basis.
        no_coeff : ndarray
            Semicanonical NO coefficients in the AO basis
        dm : ndarray, correlated density matrix in MO basis (optional).
    '''
    mf = mycc._scf
    if not load_cc:
        if rank == 0:
            mycc.solve_lambda()
            fn = 'amplitudes.h5'
            feri = h5py.File(fn, 'w')
            feri['t1a'] = np.asarray(mycc.t1[0])
            feri['t1b'] = np.asarray(mycc.t1[1])
            feri['t2aa'] = np.asarray(mycc.t2[0])
            feri['t2ab'] = np.asarray(mycc.t2[1])
            feri['t2bb'] = np.asarray(mycc.t2[2])
            feri['l1a'] = np.asarray(mycc.l1[0])
            feri['l1b'] = np.asarray(mycc.l1[1])
            feri['l2aa'] = np.asarray(mycc.l2[0])
            feri['l2ab'] = np.asarray(mycc.l2[1])
            feri['l2bb'] = np.asarray(mycc.l2[2])
            feri.close()
        comm.Barrier()
        if rank > 0:
            fn = 'amplitudes.h5'
            feri = h5py.File(fn, 'r')
            mycc.t1 = [np.asarray(feri['t1a']), np.asarray(feri['t1b'])]
            mycc.t2 = [np.asarray(feri['t2aa']), np.asarray(feri['t2ab']), np.asarray(feri['t2bb'])]
            mycc.l1 = [np.asarray(feri['l1a']), np.asarray(feri['l1b'])]
            mycc.l2 = [np.asarray(feri['l2aa']), np.asarray(feri['l2ab']), np.asarray(feri['l2bb'])]
            feri.close()
        comm.Barrier()
    else:
        fn = 'amplitudes.h5'
        feri = h5py.File(fn, 'r')
        mycc.t1 = [np.asarray(feri['t1a']), np.asarray(feri['t1b'])]
        mycc.t2 = [np.asarray(feri['t2aa']), np.asarray(feri['t2ab']), np.asarray(feri['t2bb'])]
        mycc.l1 = [np.asarray(feri['l1a']), np.asarray(feri['l1b'])]
        mycc.l2 = [np.asarray(feri['l2aa']), np.asarray(feri['l2ab']), np.asarray(feri['l2bb'])]
        feri.close()

    dm = mycc.make_rdm1()
    dm_gs = np.array(dm)
    nmoa, nmob = mycc.nmo
    nocca, noccb = mycc.nocc
    if rank == 0:
        mycc.verbose = 4
    else:
        mycc.verbose = 0

    no_occ_v_a, no_coeff_v_a = np.linalg.eigh(dm[0][nocca:,nocca:])
    no_occ_v_a = np.flip(no_occ_v_a)
    no_coeff_v_a = np.flip(no_coeff_v_a, axis=1)
    no_occ_v_b, no_coeff_v_b = np.linalg.eigh(dm[1][noccb:,noccb:])
    no_occ_v_b = np.flip(no_occ_v_b)
    no_coeff_v_b = np.flip(no_coeff_v_b, axis=1)
    if rank == 0:
        logger.info(mf, 'Full no_occ_v_a = \n %s', no_occ_v_a)
        logger.info(mf, 'Full no_occ_v_b = \n %s', no_occ_v_b)

    if nocc_act_a is not None and nocc_act_b is not None:
        vno_only = False
    if not vno_only:
        no_occ_o_a, no_coeff_o_a = np.linalg.eigh(dm[0][:nocca,:nocca])
        no_occ_o_a = np.flip(no_occ_o_a)
        no_coeff_o_a = np.flip(no_coeff_o_a, axis=1)
        no_occ_o_b, no_coeff_o_b = np.linalg.eigh(dm[1][:noccb,:noccb])
        no_occ_o_b = np.flip(no_occ_o_b)
        no_coeff_o_b = np.flip(no_coeff_o_b, axis=1)
        if rank == 0:
            logger.info(mf, 'Full no_occ_o_a = \n %s', no_occ_o_a)
            logger.info(mf, 'Full no_occ_o_b = \n %s', no_occ_o_b)

    if nvir_act_a is None and nocc_act_a is None:
        no_idx_v_a = np.where(no_occ_v_a > thresh)[0]
        if not vno_only:
            no_idx_o_a = np.where(1-no_occ_o_a > thresh)[0]
        else:
            no_idx_o_a = range(0, nocca)
    elif nvir_act_a is None and nocc_act_a is not None:
        no_idx_v_a = range(0, nmoa-nocca)
        no_idx_o_a = range(nocca-nocc_act_a, nocca)
    elif nvir_act_a is not None and nocc_act_a is None:
        no_idx_v_a = range(0, nvir_act_a)
        no_idx_o_a = range(0, nocca)
    else:
        no_idx_v_a = range(0, nvir_act_a)
        no_idx_o_a = range(nocca-nocc_act_a, nocca)

    if nvir_act_b is None and nocc_act_b is None:
        no_idx_v_b = np.where(no_occ_v_b > thresh)[0]
        if not vno_only:
            no_idx_o_b = np.where(1-no_occ_o_b > thresh)[0]
        else:
            no_idx_o_b = range(0, noccb)
    elif nvir_act_b is None and nocc_act_b is not None:
        no_idx_v_b = range(0, nmob-noccb)
        no_idx_o_b = range(noccb-nocc_act_b, noccb)
    elif nvir_act_b is not None and nocc_act_b is None:
        no_idx_v_b = range(0, nvir_act_b)
        no_idx_o_b = range(0, noccb)
    else:
        no_idx_v_b = range(0, nvir_act_b)
        no_idx_o_b = range(noccb-nocc_act_b, noccb)

    # semi-canonicalization
    fvv_a = numpy.diag(mf.mo_energy[0][nocca:])
    fvv_no_a = numpy.dot(no_coeff_v_a.T, numpy.dot(fvv_a, no_coeff_v_a))
    no_vir_a = len(no_idx_v_a)
    _, v_canon_v_a = numpy.linalg.eigh(fvv_no_a[:no_vir_a,:no_vir_a])
    if not vno_only:
        foo_a = numpy.diag(mf.mo_energy[0][:nocca])
        foo_no_a = numpy.dot(no_coeff_o_a.T, numpy.dot(foo_a, no_coeff_o_a))
        no_occ_a = nocca - len(no_idx_o_a)
        _, v_canon_o_a = numpy.linalg.eigh(foo_no_a[no_occ_a:,no_occ_a:])

    fvv_b = numpy.diag(mf.mo_energy[1][noccb:])
    fvv_no_b = numpy.dot(no_coeff_v_b.T, numpy.dot(fvv_b, no_coeff_v_b))
    no_vir_b = len(no_idx_v_b)
    _, v_canon_v_b = numpy.linalg.eigh(fvv_no_b[:no_vir_b,:no_vir_b])
    if not vno_only:
        foo_b = numpy.diag(mf.mo_energy[1][:noccb])
        foo_no_b = numpy.dot(no_coeff_o_b.T, numpy.dot(foo_b, no_coeff_o_b))
        no_occ_b = noccb - len(no_idx_o_b)
        _, v_canon_o_b = numpy.linalg.eigh(foo_no_b[no_occ_b:,no_occ_b:])

    no_coeff_v_a = numpy.dot(mf.mo_coeff[0][:,nocca:], numpy.dot(no_coeff_v_a[:,:no_vir_a], v_canon_v_a))
    no_coeff_v_b = numpy.dot(mf.mo_coeff[1][:,noccb:], numpy.dot(no_coeff_v_b[:,:no_vir_b], v_canon_v_b))
    if not vno_only:
        no_coeff_o_a = numpy.dot(mf.mo_coeff[0][:,:nocca], numpy.dot(no_coeff_o_a[:,no_occ_a:], v_canon_o_a))
        no_coeff_o_b = numpy.dot(mf.mo_coeff[1][:,:noccb], numpy.dot(no_coeff_o_b[:,no_occ_b:], v_canon_o_b))

    if not vno_only:
        ne_sum_a = np.sum(no_occ_o_a[no_idx_o_a]) + np.sum(no_occ_v_a[no_idx_v_a])
        ne_sum_b = np.sum(no_occ_o_b[no_idx_o_b]) + np.sum(no_occ_v_b[no_idx_v_b])
        n_no_a = len(no_idx_o_a) + len(no_idx_v_a)
        n_no_b = len(no_idx_o_b) + len(no_idx_v_b)
        if rank == 0:
            logger.info(mf, 'CAS no_occ_o_a = \n %s, \n no_occ_v_a = \n %s', no_occ_o_a[no_idx_o_a], no_occ_v_a[no_idx_v_a])
            logger.info(mf, 'CAS no_occ_o_b = \n %s, \n no_occ_v_b = \n %s', no_occ_o_b[no_idx_o_b], no_occ_v_b[no_idx_v_b])
    else:
        ne_sum_a = np.trace(dm[0][:nocca,:nocca]) + np.sum(no_occ_v_a[no_idx_v_a])
        ne_sum_b = np.trace(dm[1][:noccb,:noccb]) + np.sum(no_occ_v_b[no_idx_v_b])
        n_no_a = nocca + len(no_idx_v_a)
        n_no_b = noccb + len(no_idx_v_b)
        if rank == 0:
            logger.info(mf, 'CAS mo_occ_o_a = \n %s, \n no_occ_v_a = \n %s', dm[0][:nocca,:nocca].diagonal(), no_occ_v_a[no_idx_v_a])
            logger.info(mf, 'CAS mo_occ_o_b = \n %s, \n no_occ_v_b = \n %s', dm[1][:noccb,:noccb].diagonal(), no_occ_v_b[no_idx_v_b])
    assert(n_no_a == n_no_b)
    nelectron_a = int(round(ne_sum_a))
    nelectron_b = int(round(ne_sum_b))
    if rank == 0:
        logger.info(mf, 'CAS norb_a = %s, nelec_a = %s, ne_no_a = %s', n_no_a, nelectron_a, ne_sum_a)
        logger.info(mf, 'CAS norb_b = %s, nelec_b = %s, ne_no_b = %s', n_no_b, nelectron_b, ne_sum_b)

    if not vno_only:
        if local:
            no_coeff_o_a = scdm(no_coeff_o_a, np.eye(no_coeff_o_a.shape[0]))
            no_coeff_o_b = scdm(no_coeff_o_b, np.eye(no_coeff_o_b.shape[0]))
            no_coeff_v_a = scdm(no_coeff_v_a, np.eye(no_coeff_v_a.shape[0]))
            no_coeff_v_b = scdm(no_coeff_v_b, np.eye(no_coeff_v_b.shape[0]))
        no_coeff_a = np.concatenate((no_coeff_o_a, no_coeff_v_a), axis=1)
        no_coeff_b = np.concatenate((no_coeff_o_b, no_coeff_v_b), axis=1)
    else:
        if local:
            no_coeff_o_a = scdm(mf.mo_coeff[0][:,:nocca], np.eye(mf.mo_coeff[0][:,:nocca].shape[0]))
            no_coeff_o_b = scdm(mf.mo_coeff[1][:,:noccb], np.eye(mf.mo_coeff[1][:,:noccb].shape[0]))
            no_coeff_v_a = scdm(no_coeff_v_a, np.eye(no_coeff_v_a.shape[0]))
            no_coeff_v_b = scdm(no_coeff_v_b, np.eye(no_coeff_v_b.shape[0]))
            no_coeff_a = np.concatenate((no_coeff_o_a, no_coeff_v_a), axis=1)
            no_coeff_b = np.concatenate((no_coeff_o_b, no_coeff_v_b), axis=1)
        else:
            no_coeff_a = np.concatenate((mf.mo_coeff[0][:,:nocca], no_coeff_v_a), axis=1)
            no_coeff_b = np.concatenate((mf.mo_coeff[1][:,:noccb], no_coeff_v_b), axis=1)

    # new mf object for CAS
    from fcdmft.solver import scf_mu
    mol_cas = gto.M()
    mol_cas.nelectron = nelectron_a + nelectron_b
    mol_cas.spin = nelectron_a - nelectron_b
    mol_cas.verbose = mf.mol.verbose
    mol_cas.symmetry = 'c1'
    mol_cas.max_memory = mf.max_memory
    mol_cas.incore_anyway = True
    mf_cas = scf_mu.UHFNOMU(mol_cas)

    # compute CAS integrals
    h1e = mf.get_hcore()
    h1e_a = np.dot(no_coeff_a.T, np.dot(h1e[0], no_coeff_a))
    h1e_b = np.dot(no_coeff_b.T, np.dot(h1e[1], no_coeff_b))
    h1e = (h1e_a, h1e_b)
    g2e_aa = ao2mo.restore(1, ao2mo.kernel(mf._eri[0], no_coeff_a), n_no_a)
    g2e_bb = ao2mo.restore(1, ao2mo.kernel(mf._eri[1], no_coeff_b), n_no_b)
    g2e_ab = ao2mo.kernel(mf._eri[2], [no_coeff_a, no_coeff_a, no_coeff_b, no_coeff_b])
    g2e_ab = ao2mo.restore(1, g2e_ab, n_no_a)
    g2e = (g2e_aa, g2e_bb, g2e_ab)

    dm_hf = mf.make_rdm1()
    ovlp = mf.get_ovlp()
    CS_a = np.dot(no_coeff_a.T, ovlp)
    CS_b = np.dot(no_coeff_b.T, ovlp)
    dm_cas_no_a = np.dot(CS_a, np.dot(dm_hf[0], CS_a.T))
    dm_cas_no_b = np.dot(CS_b, np.dot(dm_hf[1], CS_b.T))
    dm_cas_no = (dm_cas_no_a, dm_cas_no_b)
    JK_cas_no = _get_veff(dm_cas_no, g2e)
    JK_full_no_a = np.dot(no_coeff_a.T, np.dot(mf.get_veff()[0], no_coeff_a))
    JK_full_no_b = np.dot(no_coeff_b.T, np.dot(mf.get_veff()[1], no_coeff_b))
    JK_full_no = np.array((JK_full_no_a, JK_full_no_b))
    h1e = h1e + JK_full_no - JK_cas_no
    h1e = 0.5 * (h1e + h1e.transpose(0,2,1))
    comm.Barrier()

    h1e = comm.bcast(h1e, root=0)
    if rank == 0:
        fn = 'cas_g2e.h5'
        feri = h5py.File(fn, 'w')
        feri['g2e'] = np.asarray(g2e)
        feri.close()
    comm.Barrier()
    if rank > 0:
        fn = 'cas_g2e.h5'
        feri = h5py.File(fn, 'r')
        g2e = np.asarray(feri['g2e'])
        feri.close()
    comm.Barrier()

    # set up integrals for mf_cas
    mf_cas.get_hcore = lambda *args: h1e
    mf_cas.get_ovlp = lambda *args: np.eye(n_no_a)
    mf_cas._eri = g2e
    if get_cas_mo:
        if rank == 0:
            mf_cas.max_cycle = 1
            mf_cas.kernel(dm_cas_no)
        comm.Barrier()
        mf_cas.mo_occ = comm.bcast(mf_cas.mo_occ, root=0)
        mf_cas.mo_energy = comm.bcast(mf_cas.mo_energy, root=0)
        mf_cas.mo_coeff = comm.bcast(mf_cas.mo_coeff, root=0)
    else:
        # fake mo_coeff and mo_energy
        mf_cas.mo_coeff = (np.eye(n_no_a), np.eye(n_no_b))
        mf_cas.mo_energy = (np.zeros(n_no_a), np.zeros(n_no_b))
        mf_cas.mo_occ = (np.zeros(n_no_a), np.zeros(n_no_b))
    no_coeff_a = comm.bcast(no_coeff_a, root=0)
    no_coeff_b = comm.bcast(no_coeff_b, root=0)
    no_coeff = (no_coeff_a, no_coeff_b)

    mycc.verbose = mf.verbose
    if get_cas_mo:
        if return_dm:
            return mf_cas, no_coeff, dm
        else:
            return mf_cas, no_coeff
    else:
        if return_dm:
            return mf_cas, no_coeff, h1e + JK_cas_no, dm_cas_no, dm
        else:
            return mf_cas, no_coeff, h1e + JK_cas_no, dm_cas_no

def make_casno_cisd(myci, thresh=1e-4, nvir_act=None, nocc_act=None, vno_only=False,
                    return_dm=False, get_cas_mo=True, local=False, save_fcidump=False,
                    nocc_act_low=None, nvir_act_high=None):
    '''
    CISD frozen natural orbitals for CASCI calculation

    Attributes:
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-4.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh`.
        nocc_act : int
            Number of occupied NOs to keep. Default is None. If present, overrides `thresh` and `vno_only`.
        vno_only : bool
            Only construct virtual natural orbitals. Default is True.
        return_rdm : bool
            Return correlated density matrix. Default is False.
        get_cas_mo : bool
            Diagonalize CAS Hamiltonian to get mo_coeff and mo_energy. Default is True.

    Returns:
        mf_cas : mean-field object with all integrals in NO basis.
        no_coeff : ndarray
            Semicanonical NO coefficients in the AO basis
        dm : ndarray, correlated density matrix in MO basis (optional).
    '''
    mf = myci._scf
    dm = None
    if rank == 0:
        dm = myci.make_rdm1()
    comm.Barrier()
    dm = comm.bcast(dm, root=0)
    dm_gs = dm.copy()
    nmo = myci.nmo
    nocc = myci.nocc

    no_occ_v, no_coeff_v = np.linalg.eigh(dm[nocc:,nocc:])
    no_occ_v = np.flip(no_occ_v)
    no_coeff_v = np.flip(no_coeff_v, axis=1)
    if rank == 0:
        logger.info(mf, 'Full no_occ_v = \n %s', no_occ_v)
    if nocc_act is not None:
        vno_only = False
    if not vno_only:
        no_occ_o, no_coeff_o = np.linalg.eigh(dm[:nocc,:nocc])
        no_occ_o = np.flip(no_occ_o)
        no_coeff_o = np.flip(no_coeff_o, axis=1)
        if rank == 0:
            logger.info(mf, 'Full no_occ_o = \n %s', no_occ_o)

    if nvir_act is None and nocc_act is None:
        no_idx_v = np.where(no_occ_v > thresh)[0]
        if not vno_only:
            no_idx_o = np.where(2-no_occ_o > thresh)[0]
        else:
            no_idx_o = range(0, nocc)
    elif nvir_act is None and nocc_act is not None:
        no_idx_v = range(0, nmo-nocc)
        no_idx_o = range(nocc-nocc_act, nocc)
    elif nvir_act is not None and nocc_act is None:
        no_idx_v = range(0, nvir_act)
        no_idx_o = range(0, nocc)
    else:
        no_idx_v = range(0, nvir_act)
        no_idx_o = range(nocc-nocc_act, nocc)

    # semi-canonicalization
    fvv = numpy.diag(mf.mo_energy[nocc:])
    fvv_no = numpy.dot(no_coeff_v.T, numpy.dot(fvv, no_coeff_v))
    no_vir = len(no_idx_v)
    _, v_canon_v = numpy.linalg.eigh(fvv_no[:no_vir,:no_vir])
    if not vno_only:
        foo = numpy.diag(mf.mo_energy[:nocc])
        foo_no = numpy.dot(no_coeff_o.T, numpy.dot(foo, no_coeff_o))
        no_occ = nocc - len(no_idx_o)
        _, v_canon_o = numpy.linalg.eigh(foo_no[no_occ:,no_occ:])

    no_coeff_v = numpy.dot(mf.mo_coeff[:,nocc:], numpy.dot(no_coeff_v[:,:no_vir], v_canon_v))
    if not vno_only:
        no_coeff_o = numpy.dot(mf.mo_coeff[:,:nocc], numpy.dot(no_coeff_o[:,no_occ:], v_canon_o))

    if not vno_only:
        ne_sum = np.sum(no_occ_o[no_idx_o]) + np.sum(no_occ_v[no_idx_v])
        n_no = len(no_idx_o) + len(no_idx_v)
        if rank == 0:
            logger.info(mf, 'CAS no_occ_o = \n %s, \n no_occ_v = \n %s', no_occ_o[no_idx_o], no_occ_v[no_idx_v])
    else:
        ne_sum = np.trace(dm[:nocc,:nocc]) + np.sum(no_occ_v[no_idx_v])
        n_no = nocc + len(no_idx_v)
        if rank == 0:
            logger.info(mf, 'CAS mo_occ_o = \n %s, \n no_occ_v = \n %s', dm[:nocc,:nocc].diagonal(), no_occ_v[no_idx_v])
    nelectron = int(round(ne_sum))
    if rank == 0:
        logger.info(mf, 'CAS norb = %s, nelec = %s, ne_no = %s', n_no, nelectron, ne_sum)

    if not vno_only:
        if local:
            if nocc_act_low is not None and nocc_act_low > 0 and nocc_act_low < nocc_act:
                no_coeff_o_low = scdm(no_coeff_o[:,:nocc_act_low], np.eye(no_coeff_o.shape[0]))
                no_coeff_o_high = scdm(no_coeff_o[:,nocc_act_low:], np.eye(no_coeff_o.shape[0]))
                no_coeff_o = np.concatenate((no_coeff_o_low, no_coeff_o_high), axis=1)
            else:
                no_coeff_o = scdm(no_coeff_o, np.eye(no_coeff_o.shape[0]))
            if nvir_act_high is not None and nvir_act_high > 0 and nvir_act_high < nvir_act:
                no_coeff_v_low = scdm(no_coeff_v[:,:(nvir_act-nvir_act_high)], np.eye(no_coeff_v.shape[0]))
                no_coeff_v_high = scdm(no_coeff_v[:,(nvir_act-nvir_act_high):], np.eye(no_coeff_v.shape[0]))
                no_coeff_v = np.concatenate((no_coeff_v_low, no_coeff_v_high), axis=1)
            else:
                no_coeff_v = scdm(no_coeff_v, np.eye(no_coeff_v.shape[0]))
        no_coeff = np.concatenate((no_coeff_o, no_coeff_v), axis=1)
    else:
        if local:
            if nocc_act_low is not None and nocc_act_low > 0 and nocc_act_low < nocc_act:
                no_coeff_o_low = scdm(mf.mo_coeff[:,:nocc_act_low], np.eye(mf.mo_coeff[:,:nocc].shape[0]))
                no_coeff_o_high = scdm(mf.mo_coeff[:,nocc_act_low:], np.eye(mf.mo_coeff[:,:nocc].shape[0]))
                no_coeff_o = np.concatenate((no_coeff_o_low, no_coeff_o_high), axis=1)
            else:
                no_coeff_o = scdm(mf.mo_coeff[:,:nocc], np.eye(mf.mo_coeff[:,:nocc].shape[0]))
            if nvir_act_high is not None and nvir_act_high > 0 and nvir_act_high < nvir_act:
                no_coeff_v_low = scdm(no_coeff_v[:,:(nvir_act-nvir_act_high)], np.eye(no_coeff_v.shape[0]))
                no_coeff_v_high = scdm(no_coeff_v[:,(nvir_act-nvir_act_high):], np.eye(no_coeff_v.shape[0]))
                no_coeff_v = np.concatenate((no_coeff_v_low, no_coeff_v_high), axis=1)
            else:
                no_coeff_v = scdm(no_coeff_v, np.eye(no_coeff_v.shape[0]))
            no_coeff = np.concatenate((no_coeff_o, no_coeff_v), axis=1)
        else:
            no_coeff = np.concatenate((mf.mo_coeff[:,:nocc], no_coeff_v), axis=1)

    # new mf object for CAS
    mol_cas = gto.M()
    mol_cas.nelectron = nelectron
    mol_cas.verbose = mf.mol.verbose
    mol_cas.symmetry = 'c1'
    mol_cas.max_memory = mf.max_memory
    mol_cas.incore_anyway = True
    mf_cas = scf.RHF(mol_cas)

    # compute CAS integrals
    h1e = np.dot(no_coeff.T, np.dot(mf.get_hcore(), no_coeff))
    g2e = ao2mo.restore(8, ao2mo.kernel(mf._eri, no_coeff), n_no)

    dm_hf = mf.make_rdm1()
    ovlp = mf.get_ovlp()
    CS = np.dot(no_coeff.T, ovlp)
    dm_cas_no = np.dot(CS, np.dot(dm_hf, CS.T))
    JK_cas_no = _get_veff(dm_cas_no, g2e)[0]
    JK_full_no = np.dot(no_coeff.T, np.dot(mf.get_veff(), no_coeff))
    h1e = h1e + JK_full_no - JK_cas_no
    h1e = 0.5 * (h1e + h1e.T)
    comm.Barrier()

    h1e = comm.bcast(h1e, root=0)
    if rank == 0:
        fn = 'cas_g2e.h5'
        feri = h5py.File(fn, 'w')
        feri['g2e'] = np.asarray(g2e)
        feri.close()
    comm.Barrier()
    if rank > 0:
        fn = 'cas_g2e.h5'
        feri = h5py.File(fn, 'r')
        g2e = np.asarray(feri['g2e'])
        feri.close()
    comm.Barrier()

    # set up integrals for mf_cas
    mf_cas.get_hcore = lambda *args: h1e
    mf_cas.get_ovlp = lambda *args: np.eye(n_no)
    mf_cas._eri = g2e
    if get_cas_mo:
        if rank == 0:
            mf_cas.max_cycle = 1
            mf_cas.kernel(dm_cas_no)
        comm.Barrier()
        mf_cas.mo_occ = comm.bcast(mf_cas.mo_occ, root=0)
        mf_cas.mo_energy = comm.bcast(mf_cas.mo_energy, root=0)
        mf_cas.mo_coeff = comm.bcast(mf_cas.mo_coeff, root=0)
    else:
        # fake mo_coeff and mo_energy
        mf_cas.mo_coeff = np.eye(n_no)
        mf_cas.mo_energy = np.zeros(n_no)
        mf_cas.mo_occ = np.zeros(n_no)
    no_coeff = comm.bcast(no_coeff, root=0)

    if rank == 0:
        if save_fcidump:
            from pyscf import tools
            tools.fcidump.from_integrals('FCIDUMP', h1e, ao2mo.restore(1,g2e,n_no),
                                        n_no, nelectron, ms=0)
    comm.Barrier()

    if get_cas_mo:
        if return_dm:
            return mf_cas, no_coeff, dm_gs
        else:
            return mf_cas, no_coeff
    else:
        if return_dm:
            return mf_cas, no_coeff, h1e + JK_cas_no, dm_cas_no, dm_gs
        else:
            return mf_cas, no_coeff, h1e + JK_cas_no, dm_cas_no

def make_cas_hf(myhf, nvir_act=None, nocc_act=None, ecut_occ=None, ecut_vir=None,
                save_fcidump=False):
    '''
    Mean-field molecular orbitals for CASCI calculation

    Attributes:
        nvir_act : int
            Number of virtual MOs to keep. Default is None.
        nocc_act : int
            Number of occupied MOs to keep. Default is None.
        ecut_vir : float
            Energy range to keep virtual MOs.
        ecut_occ : float
            Energy range to keep occupied MOs.
        save_fcidump : bool
            Whether to dump CAS integrals into FCIDUMP file.

    Returns:
        mf_cas : mean-field object with all integrals in MO basis.
        no_coeff : ndarray
            CAS MO coefficients in the AO basis
    '''
    mf = myhf
    nocc = int(np.sum(mf.mo_occ)) // 2

    if ecut_occ is not None and ecut_vir is not None:
        no_idx = np.where((mf.mo_energy >= mf.mo_energy[nocc-1] - ecut_occ) \
                          & (mf.mo_energy < mf.mo_energy[nocc] + ecut_vir))[0]
    else:
        no_idx = range(nocc-nocc_act, nocc+nvir_act)

    n_no = len(no_idx)
    nelectron = int(np.sum(mf.mo_occ[no_idx]))
    if rank == 0:
        logger.info(mf, 'CAS norb = %s, nelec = %s', n_no, nelectron)
    no_coeff = mf.mo_coeff[:,no_idx]

    # new mf object for CAS
    mol_cas = gto.M()
    mol_cas.nelectron = nelectron
    mol_cas.verbose = mf.mol.verbose
    mol_cas.symmetry = 'c1'
    mol_cas.max_memory = mf.max_memory
    mol_cas.incore_anyway = True
    mf_cas = scf.RHF(mol_cas)

    # compute CAS integrals
    h1e = np.dot(no_coeff.T, np.dot(mf.get_hcore(), no_coeff))
    g2e = ao2mo.restore(8, ao2mo.kernel(mf._eri, no_coeff), n_no)

    dm_hf = mf.make_rdm1()
    ovlp = mf.get_ovlp()
    CS = np.dot(no_coeff.T, ovlp)
    dm_cas_no = np.dot(CS, np.dot(dm_hf, CS.T))
    JK_cas_no = _get_veff(dm_cas_no, g2e)[0]
    JK_full_no = np.dot(no_coeff.T, np.dot(mf.get_veff(), no_coeff))
    h1e = h1e + JK_full_no - JK_cas_no
    h1e = 0.5 * (h1e + h1e.T)
    comm.Barrier()

    h1e = comm.bcast(h1e, root=0)
    if rank == 0:
        fn = 'cas_g2e.h5'
        feri = h5py.File(fn, 'w')
        feri['g2e'] = np.asarray(g2e)
        feri.close()
    comm.Barrier()
    if rank > 0:
        fn = 'cas_g2e.h5'
        feri = h5py.File(fn, 'r')
        g2e = np.asarray(feri['g2e'])
        feri.close()
    comm.Barrier()

    # set up integrals for mf_cas
    mf_cas.get_hcore = lambda *args: h1e
    mf_cas.get_ovlp = lambda *args: np.eye(n_no)
    mf_cas._eri = g2e
    if rank == 0:
        mf_cas.max_cycle = 0
        mf_cas.kernel(dm_cas_no)
    comm.Barrier()
    no_coeff = comm.bcast(no_coeff, root=0)
    mf_cas.mo_occ = mf.mo_occ[no_idx]
    mf_cas.mo_energy = mf.mo_energy[no_idx]
    mf_cas.mo_coeff = np.eye(n_no)

    if rank == 0:
        if save_fcidump:
            from pyscf import tools
            tools.fcidump.from_integrals('FCIDUMP', h1e, ao2mo.restore(1,g2e,n_no),
                                        n_no, nelectron, ms=0)
    comm.Barrier()

    return mf_cas, no_coeff

def scdm(coeff, overlap):
    from pyscf import lo
    aux = lo.orth.lowdin(overlap)
    no = coeff.shape[1]
    ova = coeff.T @ overlap @ aux
    piv = scipy.linalg.qr(ova, pivoting=True)[2]
    bc = ova[:, piv[:no]]
    ova = np.dot(bc.T, bc)
    s12inv = lo.orth.lowdin(ova)
    return coeff @ bc @ s12inv

def _get_jk(dm, eri):
    """
    Get J and K potential from rdm and ERI.
    vj00 = np.tensordot(dm[0], eri[0], ((0,1), (0,1))) # J a from a
    vj11 = np.tensordot(dm[1], eri[1], ((0,1), (0,1))) # J b from b
    vj10 = np.tensordot(dm[0], eri[2], ((0,1), (0,1))) # J b from a
    vj01 = np.tensordot(dm[1], eri[2], ((1,0), (3,2))) # J a from b
    vk00 = np.tensordot(dm[0], eri[0], ((0,1), (0,3))) # K a from a
    vk11 = np.tensordot(dm[1], eri[1], ((0,1), (0,3))) # K b from b
    JK = np.asarray([vj00 + vj01 - vk00, vj11 + vj10 - vk11])
    """
    dm = np.asarray(dm, dtype=np.double)
    eri = np.asarray(eri, dtype=np.double)
    if len(dm.shape) == 2:
        dm = dm[np.newaxis, ...]
    if len(eri.shape) == 4:
        eri = eri[np.newaxis, ...]
    spin = dm.shape[0]
    norb = dm.shape[-1]
    if spin == 1:
        eri = ao2mo.restore(8, eri, norb)
        vj, vk = scf.hf.dot_eri_dm(eri, dm, hermi=1)
    else:
        eri_aa = ao2mo.restore(8, eri[0], norb)
        eri_bb = ao2mo.restore(8, eri[1], norb)
        eri_ab = ao2mo.restore(4, eri[2], norb)
        vj00, vk00 = scf.hf.dot_eri_dm(eri_aa, dm[0], hermi=1)
        vj11, vk11 = scf.hf.dot_eri_dm(eri_bb, dm[1], hermi=1)
        vj01, _ = scf.hf.dot_eri_dm(eri_ab, dm[1], hermi=1, with_j=True, with_k=False)
        # ZHC NOTE the transpose, since the dot_eri_dm uses the convention ijkl, kl -> ij
        vj10, _ = scf.hf.dot_eri_dm(eri_ab.T, dm[0], hermi=1, with_j=True, with_k=False)
        # ZHC NOTE explicit write down vj, without broadcast
        vj = np.asarray([[vj00, vj11], [vj01, vj10]])
        vk = np.asarray([vk00, vk11])
    return vj, vk

def _get_veff(dm, eri):
    """
    Get HF effective potential from rdm and ERI.
    """
    dm = np.asarray(dm, dtype=np.double)
    if len(dm.shape) == 2:
        dm = dm[np.newaxis, ...]
    spin = dm.shape[0]
    vj, vk = _get_jk(dm, eri)
    if spin == 1:
        JK = vj - vk*0.5 
    else:
        JK = vj[0] + vj[1] - vk
    return JK

if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        ['H' , (0. , -0.757 , 0.587)],
        ['H' , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 4
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    if rank == 0:
        mf.kernel()
    comm.Barrier()
    mf.mo_occ = comm.bcast(mf.mo_occ, root=0)
    mf.mo_energy = comm.bcast(mf.mo_energy, root=0)
    mf.mo_coeff = comm.bcast(mf.mo_coeff, root=0)
    mf._eri = comm.bcast(mf._eri, root=0)
    comm.Barrier()

    # MP2 frozen natural orbitals for CASCI
    mymp = mp.MP2(mf).set(verbose=0)
    mymp.kernel()
    mf_cas, no_coeff = make_casno_mp(mymp, thresh=1e-3, nvir_act=None, nocc_act=None,
                                     ea_no=True, ip_no=False, vno_only=True)
    mycc = cc.CCSD(mf_cas)
    if rank == 0:
        mycc.kernel()
        eip,cip = mycc.ipccsd(nroots=3)
        eea,cea = mycc.eaccsd(nroots=10)

    # GW frozen natural orbitals for CASCI
    gw = gw_gf.GWGF(mf)
    gw.rdm = True
    gw.ac = 'pade'
    gw.fullsigma = True
    gw.eta = 0.2/27.211386
    omega = np.linspace(-10./27.211386, 10./27.211386, 2)
    gw.omega_emo = True
    gf, gf0, sigma = gw.kernel(omega=omega)
    mf_cas, no_coeff = make_casno_gw(gw, thresh=1e-3, nvir_act=None, nocc_act=None,
                                     ea_no=4, ip_no=None, vno_only=True)
    mycc = cc.CCSD(mf_cas)
    if rank == 0:
        mycc.kernel()
        eip,cip = mycc.ipccsd(nroots=3)
        eea,cea = mycc.eaccsd(nroots=10)

    # CCSD frozen natural orbitals for CASCI
    mycc = cc.CCSD(mf)
    mycc.kernel()
    mf_cas, no_coeff = make_casno_cc(mycc, thresh=1e-3, nvir_act=None, nocc_act=None,
                                     ea_no=4, ip_no=None, vno_only=True)
    mycc = cc.CCSD(mf_cas)
    if rank == 0:
        mycc.kernel()
        eip,cip = mycc.ipccsd(nroots=3)
        eea,cea = mycc.eaccsd(nroots=10)
