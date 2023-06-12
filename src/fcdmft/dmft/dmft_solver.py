import time, sys, os, h5py
import numpy as np
from scipy import linalg

from pyscf.lib import logger
from pyscf import lib, gto, ao2mo, cc
from fcdmft import solver
from fcdmft.solver import scf_mu as scf
from mpi4py import MPI

einsum = lib.einsum

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

fci_ = False
try:
    import PyCheMPS2
    import ctypes
    fci_ = True
except:
    pass

'''
Impurity solver interfaces for DMFT calculation
'''

def mf_kernel(himp, eri_imp, mu, nao, dm0, max_mem, verbose=logger.NOTE):
    '''
    HF calculation with fixed chemical potential and fluctuating occupancy

    Args:
        himp : (spin, nimp, nimp) ndarray
        eri_imp : (spin, nimp, nimp, nimp, nimp) ndarray
        mu : float
        nao: interger
        dm0 : (spin, nimp, nimp) ndarray

    Returns:
        mf : mean-field class
    '''
    spin, n = himp.shape[0:2]
    mol = gto.M()
    mol.verbose = verbose
    mol.incore_anyway = True
    mol.build()
    if spin == 1:
        mf = scf.RHF(mol, mu)
        mf.max_memory = max_mem
        mf.mo_energy = np.zeros([n])
        mf.max_cycle = 100
        mf.conv_tol = 1e-12
        mf.diis_space = 10

        mf.get_hcore = lambda *args: himp[0]
        mf.get_ovlp = lambda *args: np.eye(n)
        mf._eri = ao2mo.restore(8, eri_imp[0], n)

        mf.smearing = None
        if rank == 0:
            mf.kernel(dm0[0])
        else:
            mf.verbose = 1
            mf.kernel(dm0[0])
        if mf.converged is False:
            # TODO: disable smearing for now
            raise RuntimeError('SCF with smearing not converged.')
            exit()
            # If SCF does not converge, try smearing for convergence,
            # then do one-shot HF without smearing
            mf.smearing = 0.01
            mf.conv_tol = 1e-10
            mf.kernel(dm0[0])
            if mf.converged is False:
                raise RuntimeError('SCF with smearing not converged.')
                exit()

        mf.verbose = verbose
        dm = mf.make_rdm1()
        if rank == 0:
            logger.info(mf, 'HF Nelec = %s', np.trace(dm[:nao,:nao]))
        if mf.smearing is not None:
            mf.smearing = None
            mf.max_cycle = 1
            mf.kernel(dm)
    else:
        mf = scf.UHF(mol, mu)
        mf.max_memory = max_mem
        mf.mo_energy = np.zeros([2,n])
        mf.max_cycle = 100
        mf.conv_tol = 1e-12
        mf.diis_space = 10

        mf.get_hcore = lambda *args: himp
        mf.get_ovlp = lambda *args: np.eye(n)
        mf._eri = (ao2mo.restore(4, eri_imp[0], n), ao2mo.restore(4, eri_imp[1], n),
                   ao2mo.restore(4, eri_imp[2], n))

        mf.smearing = None
        if rank == 0:
            mf.kernel(dm0)
        else:
            mf.verbose = 1
            mf.kernel(dm0)
            mf.verbose = verbose
        dm = mf.make_rdm1()
        if rank == 0:
            logger.info(mf, 'HF Nelec_up = %s, Nelec_dn = %s, Nelec = %s',
                        np.trace(dm[0,:nao,:nao]),np.trace(dm[1,:nao,:nao]),
                        np.trace(dm[0,:nao,:nao])+np.trace(dm[1,:nao,:nao]))
        if mf.converged is False:
            raise RuntimeError('SCF not converged.')
            exit()

    comm.Barrier()
    mo_coeff  = comm.bcast(mf.mo_coeff,root=0)
    mo_energy = comm.bcast(mf.mo_energy,root=0)
    mo_occ    = comm.bcast(mf.mo_occ,root=0)
    mf.mo_coeff = mo_coeff
    mf.mo_energy = mo_energy
    mf.mo_occ = mo_occ
    comm.Barrier()

    return mf

def mf_gf(mf, freqs, delta, ao_orbs=None):
    '''Calculate the mean-field GF matrix in AO basis'''
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    if len(mo_coeff.shape) == 2:
        mo_coeff = mo_coeff[np.newaxis, ...]
        mo_energy = mo_energy[np.newaxis, ...]
    spin, nmo = mo_coeff.shape[0:2]
    if ao_orbs is None:
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    nw = len(freqs)
    gf = np.zeros([spin, nmo, nmo, nw], np.complex128)
    for s in range(spin):
        for iw, w in enumerate(freqs):
            g = np.diag(1./((w+1j*delta) * \
                        np.ones([nmo], np.complex128) - mo_energy[s]))
            gf[s,:,:,iw] = np.dot(mo_coeff[s], np.dot(g, mo_coeff[s].T))

    return gf[:,:nao,:nao]

def mf_gf_withfrz(mf, freqs, delta, ao_orbs=None, nfrz=0):
    '''Calculate mean-field GF matrix in AO basis with freezing core'''
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    if len(mo_coeff.shape) == 2:
        mo_coeff = mo_coeff[np.newaxis, ...]
        mo_energy = mo_energy[np.newaxis, ...]
    spin, nmo = mo_coeff.shape[0:2]
    if ao_orbs is None:
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    nw = len(freqs)

    gf = np.zeros([spin, nmo, nmo, nw], np.complex128)
    for s in range(spin):
        for iw, w in enumerate(freqs):
            g = np.diag(1./((w+1j*delta) * \
                        np.ones([nmo], np.complex128) - mo_energy[s]))
            gf[s,:,:,iw] = np.dot(mo_coeff[s], np.dot(g, mo_coeff[s].T))

    gf_frz = np.zeros([spin, nmo, nmo, nw], np.complex128)
    for s in range(spin):
        for iw, w in enumerate(freqs):
            g = np.diag(1./((w+1j*delta) * \
                      np.ones([nmo-nfrz], np.complex128) - mo_energy[s][nfrz:]))
            gf_frz[s,:,:,iw] = np.dot(mo_coeff[s][:,nfrz:], np.dot(g, mo_coeff[s][:,nfrz:].T))

    return gf[:,:nao,:nao], gf_frz[:,:nao,:nao]

def cas_gw(mf, freqs, delta, composite=True, thresh=5e-3, nvir_act=None, nocc_act=None,
           local=False):
    from fcdmft.gw.mol import gw_gf
    from fcdmft.solver.casno import make_casno_gw
    from pyscf.ao2mo import _ao2mo

    # Full GW@HF
    fn = 'cderi.h5'
    feri = h5py.File(fn, 'r')
    cderi = np.asarray(feri['cderi'])
    feri.close()
    naux, nimp, nimp = cderi.shape
    nmo = len(mf.mo_energy)
    Lpq = np.zeros((naux, nmo, nmo))
    Lpq[:,:nimp,:nimp] = cderi
    mo = np.asarray(mf.mo_coeff, order='F')
    ijslice = (0, nmo, 0, nmo)
    Lpq_mo = None
    Lpq_mo = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s1', mosym='s1', out=Lpq_mo)
    Lpq_mo = Lpq_mo.reshape(naux, nmo, nmo)

    gf_gw = None
    gw = gw_gf.GWGF(mf)
    gw.rdm = True
    gw.fullsigma = True
    gw.eta = delta
    gw.omega_emo = True
    nmo = gw.nmo
    if rank == 0:
        gf_gw, gf0, sigma = gw.kernel(Lpq=Lpq_mo, omega=freqs)
        gf_gw = gf_gw[:,:,nmo:]
        logger.info(mf, 'Full GW@HF: energy DOS')
        for i in range(len(freqs)):
            logger.info(mf, '%s %s',freqs[i], -np.trace(gf_gw[:,:,i].imag)/np.pi)
    comm.Barrier()
    gf_gw = comm.bcast(gf_gw, root=0)
    gw.gf = comm.bcast(gw.gf, root=0)
    gw.mo_energy = comm.bcast(gw.mo_energy, root=0)

    # Construct CAS problem from GW density matrices
    mf_cas, no_coeff, dm = make_casno_gw(gw, thresh=thresh, nvir_act=nvir_act, nocc_act=nocc_act,
                                         return_dm=True, local=local)
    dm_ao = np.dot(mf.mo_coeff, np.dot(dm, mf.mo_coeff.T))
    if rank == 0:
        logger.info(mf, 'Full GW Nelec = %s', np.trace(dm_ao[:nimp,:nimp]))
        logger.info(mf, 'Full GW 1-RDM diag = \n %s', dm_ao[:nimp,:nimp].diagonal())

    # CAS GW@HF for self-energy composite method
    if composite:
        gf_gw_cas = None
        dm_cas_ao = None
        if rank == 0:
            nmo_cas = len(mf_cas.mo_energy)
            mo = np.asarray(np.dot(no_coeff, mf_cas.mo_coeff), order='F')
            ijslice = (0, nmo_cas, 0, nmo_cas)
            Lpq_cas = None
            Lpq_cas = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s1', mosym='s1', out=Lpq_cas)
            Lpq_cas = Lpq_cas.reshape(naux, nmo_cas, nmo_cas)
            gw_cas = gw_gf.GWGF(mf_cas)
            gw_cas.rdm = True
            gw_cas.fullsigma = True
            gw_cas.eta = gw.eta
            gw_cas.omega_emo = False
            gw_cas.with_df = gw.with_df
            gf_gw_cas, gf0, sigma = gw_cas.kernel(Lpq=Lpq_cas, omega=freqs)
            logger.info(mf, 'CAS GW@HF: energy DOS')
            for i in range(len(freqs)):
                logger.info(mf, '%s %s',freqs[i], -np.trace(gf_gw_cas[:,:,i].imag)/np.pi)
            dm_cas = gw_cas.make_rdm1()
            dm_cas_ao = np.dot(mf_cas.mo_coeff, np.dot(dm_cas, mf_cas.mo_coeff.T))
        comm.Barrier()
        gf_gw_cas = comm.bcast(gf_gw_cas, root=0)
        dm_cas_ao = comm.bcast(dm_cas_ao, root=0)

    no_coeff = no_coeff[np.newaxis, ...]
    if composite:
        for iw in range(len(freqs)):
            gf_gw[:,:,iw] = np.dot(mf.mo_coeff, np.dot(gf_gw[:,:,iw], mf.mo_coeff.T))
            gf_gw_cas[:,:,iw] = np.dot(mf_cas.mo_coeff, np.dot(gf_gw_cas[:,:,iw], mf_cas.mo_coeff.T))
        gf_gw = gf_gw[np.newaxis, ...]
        gf_gw_cas = gf_gw_cas[np.newaxis, ...]
        return mf_cas, no_coeff, gf_gw, gf_gw_cas, dm_ao, dm_cas_ao
    else:
        gf_hf = mf_gf(mf, freqs, delta)
        gf_hf_cas = mf_gf(mf_cas, freqs, delta)
        dm_ao = mf.make_rdm1()
        dm_cas_ao = mf_cas.make_rdm1()
        return mf_cas, no_coeff, gf_hf, gf_hf_cas, dm_ao, dm_cas_ao

def cas_ccsd(mf, freqs, delta, nimp=None, composite=True, thresh=5e-3, nvir_act=None, nocc_act=None,
             cc_gmres_tol=1e-3, local=False, load_cc=False, save_cas=False, thresh2=None,
             nocc_act_low=None, nvir_act_high=None):
    from fcdmft.solver.casno import make_casno_cc
    from fcdmft.solver import mpiccgf as ccgf

    # Full CCSD
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-8
    mycc.conv_tol_normt = 1e-5
    mycc.diis_space = 15
    mycc.max_cycle = 200
    if not load_cc:
        if rank == 0:
            mycc.kernel()
            if mycc.converged is False:
                log = logger.Logger(sys.stdout, 4)
                log.warn('!!! Full CCSD not converged !!!')
        comm.Barrier()
    nmo = mycc.nmo
    mf_cas, no_coeff, dm = make_casno_cc(mycc, thresh=thresh, nvir_act=nvir_act, nocc_act=nocc_act,
                                         return_dm=True, local=local, load_cc=load_cc,
                                         nocc_act_low=nocc_act_low, nvir_act_high=nvir_act_high)
    if thresh2 is not None:
        mf_cas_2, no_coeff_2 = make_casno_cc(mycc, thresh=thresh2, local=local, load_cc=True)

    if load_cc and os.path.isfile('mf_cas.h5'):
        fn = 'mf_cas.h5'
        feri = h5py.File(fn, 'r')
        mf_cas.mo_coeff = np.asarray(feri['mo_coeff'])
        mf_cas.mo_occ = np.asarray(feri['mo_occ'])
        mf_cas.mo_energy = np.asarray(feri['mo_energy'])
        h1e = np.asarray(feri['h1e'])
        g2e = np.asarray(feri['g2e'])
        no_coeff = np.asarray(feri['no_coeff'])
        feri.close()
        mf_cas.get_hcore = lambda *args: h1e
        mf_cas._eri = g2e

    if save_cas:
        if rank == 0:
            fn = 'mf_cas.h5'
            feri = h5py.File(fn, 'w')
            feri['mo_coeff'] = np.asarray(mf_cas.mo_coeff)
            feri['mo_occ'] = np.asarray(mf_cas.mo_occ)
            feri['mo_energy'] = np.asarray(mf_cas.mo_energy)
            feri['h1e'] = np.asarray(mf_cas.get_hcore())
            feri['g2e'] = np.asarray(mf_cas._eri)
            feri['no_coeff'] = np.asarray(no_coeff)
            feri.close()
        comm.Barrier()

    dm_ao = np.dot(mf.mo_coeff, np.dot(dm, mf.mo_coeff.T))
    if rank == 0:
        if nimp is None:
            nimp = nmo
        logger.info(mf, 'Full CCSD Nelec = %s', np.trace(dm_ao[:nimp,:nimp]))
        logger.info(mf, 'Full CCSD 1-RDM diag = \n %s', dm_ao[:nimp,:nimp].diagonal())

    # Full and CAS CCGF for self-energy composite method
    if composite:
        if thresh2 is None:
            gf = ccgf.CCGF(mycc, tol=cc_gmres_tol)
            orbs = range(len(mf.mo_energy))
            g_ip = gf.ipccsd_mo(orbs, orbs, freqs.conj(), delta).conj()
            g_ea = gf.eaccsd_mo(orbs, orbs, freqs, delta)
            gf_cc_full = g_ip + g_ea
        else:
            cc_cas_2 = cc.CCSD(mf_cas_2)
            cc_cas_2.conv_tol = 1e-8
            cc_cas_2.conv_tol_normt = 1e-5
            cc_cas_2.diis_space = 15
            cc_cas_2.max_cycle = 200
            if rank == 0:
                cc_cas_2.kernel()
                if cc_cas_2.converged is False:
                    log = logger.Logger(sys.stdout, 4)
                    log.warn('!!! CAS2 CCSD not converged !!!')
                cc_cas_2.solve_lambda()
                fn = 'amplitudes_cas_2.h5'
                feri = h5py.File(fn, 'w')
                feri['t1'] = np.asarray(cc_cas_2.t1)
                feri['t2'] = np.asarray(cc_cas_2.t2)
                feri['l1'] = np.asarray(cc_cas_2.l1)
                feri['l2'] = np.asarray(cc_cas_2.l2)
                feri.close()
            comm.Barrier()
            if rank > 0:
                fn = 'amplitudes_cas_2.h5'
                feri = h5py.File(fn, 'r')
                cc_cas_2.t1 = np.asarray(feri['t1'])
                cc_cas_2.t2 = np.asarray(feri['t2'])
                cc_cas_2.l1 = np.asarray(feri['l1'])
                cc_cas_2.l2 = np.asarray(feri['l2'])
                feri.close()
            comm.Barrier()

            gf = ccgf.CCGF(cc_cas_2, tol=cc_gmres_tol)
            orbs = range(len(mf_cas_2.mo_energy))
            g_ip = gf.ipccsd_mo(orbs, orbs, freqs.conj(), delta).conj()
            g_ea = gf.eaccsd_mo(orbs, orbs, freqs, delta)
            gf_cc_cas_2 = g_ip + g_ea

        cc_cas = cc.CCSD(mf_cas)
        cc_cas.conv_tol = 1e-8
        cc_cas.conv_tol_normt = 1e-5
        cc_cas.diis_space = 15
        cc_cas.max_cycle = 200
        if rank == 0:
            cc_cas.kernel()
            if cc_cas.converged is False:
                log = logger.Logger(sys.stdout, 4)
                log.warn('!!! CAS CCSD not converged !!!')
            cc_cas.solve_lambda()
            fn = 'amplitudes_cas.h5'
            feri = h5py.File(fn, 'w')
            feri['t1'] = np.asarray(cc_cas.t1)
            feri['t2'] = np.asarray(cc_cas.t2)
            feri['l1'] = np.asarray(cc_cas.l1)
            feri['l2'] = np.asarray(cc_cas.l2)
            feri.close()
        comm.Barrier()
        if rank > 0:
            fn = 'amplitudes_cas.h5'
            feri = h5py.File(fn, 'r')
            cc_cas.t1 = np.asarray(feri['t1'])
            cc_cas.t2 = np.asarray(feri['t2'])
            cc_cas.l1 = np.asarray(feri['l1'])
            cc_cas.l2 = np.asarray(feri['l2'])
            feri.close()
        comm.Barrier()
        dm_cas = cc_cas.make_rdm1()
        dm_cas_ao = np.dot(mf_cas.mo_coeff, np.dot(dm_cas, mf_cas.mo_coeff.T))

        gf = ccgf.CCGF(cc_cas, tol=cc_gmres_tol)
        orbs = range(len(mf_cas.mo_energy))
        g_ip = gf.ipccsd_mo(orbs, orbs, freqs.conj(), delta).conj()
        g_ea = gf.eaccsd_mo(orbs, orbs, freqs, delta)
        gf_cc_cas = g_ip + g_ea

    no_coeff = no_coeff[np.newaxis, ...]
    if composite:
        if thresh2 is None:
            for iw in range(len(freqs)):
                gf_cc_full[:,:,iw] = np.dot(mf.mo_coeff, np.dot(gf_cc_full[:,:,iw], mf.mo_coeff.T))
            gf_cc_full = gf_cc_full[np.newaxis, ...]
        else:
            gf_hf = mf_gf(mf, freqs, delta)
            gf_hf_cas_2 = mf_gf(mf_cas_2, freqs, delta)
            for iw in range(len(freqs)):
                gf_cc_cas_2[:,:,iw] = np.dot(mf_cas_2.mo_coeff, np.dot(gf_cc_cas_2[:,:,iw], mf_cas_2.mo_coeff.T))
            gf_cc_cas_2 = gf_cc_cas_2[np.newaxis, ...]

            sigma_cas = get_sigma(gf_hf_cas_2, gf_cc_cas_2)
            sigma_full = np.zeros_like(gf_hf)
            spin = gf_hf.shape[0]
            for s in range(spin):
                for iw in range(len(freqs)):
                    sigma_full[s,:,:,iw] = np.dot(no_coeff_2, np.dot(sigma_cas[s,:,:,iw], no_coeff_2.T))
            gf_cc_full = np.zeros_like(gf_hf)
            for s in range(spin):
                for iw in range(len(freqs)):
                    gf_cc_full[s,:,:,iw] = np.linalg.inv(np.linalg.inv(gf_hf[s,:,:,iw]) - sigma_full[s,:,:,iw])
        for iw in range(len(freqs)):
            gf_cc_cas[:,:,iw] = np.dot(mf_cas.mo_coeff, np.dot(gf_cc_cas[:,:,iw], mf_cas.mo_coeff.T))
        gf_cc_cas = gf_cc_cas[np.newaxis, ...]
        return mf_cas, no_coeff, gf_cc_full, gf_cc_cas, dm_ao, dm_cas_ao
    else:
        gf_hf = mf_gf(mf, freqs, delta)
        gf_hf_cas = mf_gf(mf_cas, freqs, delta)
        dm_ao = mf.make_rdm1()
        dm_cas_ao = mf_cas.make_rdm1()
        return mf_cas, no_coeff, gf_hf, gf_hf_cas, dm_ao, dm_cas_ao

def cas_uccsd(mf, freqs, delta, nimp=None, composite=True, thresh=5e-3, nvir_act_a=None, nocc_act_a=None,
              nvir_act_b=None, nocc_act_b=None, cc_gmres_tol=1e-3, local=False, load_cc=False,
              save_cas=False, thresh2=None):
    from fcdmft.solver import ucc_eri
    from fcdmft.solver.casno import make_casno_ucc
    from fcdmft.solver import mpiuccgf as uccgf

    # Full UCCSD
    mycc = cc.UCCSD(mf)
    mycc.conv_tol = 1e-8
    mycc.conv_tol_normt = 1e-5
    mycc.diis_space = 15
    mycc.max_cycle = 200
    if not load_cc:
        if rank == 0:
            mycc.kernel()
            if mycc.converged is False:
                log = logger.Logger(sys.stdout, 4)
                log.warn('!!! Full CCSD not converged !!!')
        comm.Barrier()
    nmoa, nmob = mycc.nmo
    mf_cas, no_coeff, dm = make_casno_ucc(mycc, thresh=thresh, nvir_act_a=nvir_act_a,
                                      nocc_act_a=nocc_act_a, nvir_act_b=nvir_act_b,
                                      nocc_act_b=nocc_act_b, return_dm=True,
                                      local=local, load_cc=load_cc)

    if thresh2 is not None:
        mf_cas_2, no_coeff_2 = make_casno_ucc(mycc, thresh=thresh2, local=local, load_cc=True)

    if load_cc and os.path.isfile('mf_cas.h5'):
        fn = 'mf_cas.h5'
        feri = h5py.File(fn, 'r')
        mf_cas.mo_coeff = np.asarray(feri['mo_coeff'])
        mf_cas.mo_occ = np.asarray(feri['mo_occ'])
        mf_cas.mo_energy = np.asarray(feri['mo_energy'])
        h1e = np.asarray(feri['h1e'])
        g2e = np.asarray(feri['g2e'])
        no_coeff = np.asarray(feri['no_coeff'])
        feri.close()
        mf_cas.get_hcore = lambda *args: h1e
        mf_cas._eri = g2e

    if save_cas:
        if rank == 0:
            fn = 'mf_cas.h5'
            feri = h5py.File(fn, 'w')
            feri['mo_coeff'] = np.asarray(mf_cas.mo_coeff)
            feri['mo_occ'] = np.asarray(mf_cas.mo_occ)
            feri['mo_energy'] = np.asarray(mf_cas.mo_energy)
            feri['h1e'] = np.asarray(mf_cas.get_hcore())
            feri['g2e'] = np.asarray(mf_cas._eri)
            feri['no_coeff'] = np.asarray(no_coeff)
            feri.close()
        comm.Barrier()

    dm_ao = np.array(dm)
    for s in range(2):
        dm_ao[s] = np.dot(mf.mo_coeff[s], np.dot(dm[s], mf.mo_coeff[s].T))
    if rank == 0:
        if nimp is None:
            nimp = nmoa
        logger.info(mf, 'Full CCSD Nelec_up = %s, Nelec_dn = %s, Nelec = %s',
                    np.trace(dm_ao[0][:nimp,:nimp]), np.trace(dm_ao[1][:nimp,:nimp]),
                    np.trace(dm_ao[0][:nimp,:nimp])+np.trace(dm_ao[1][:nimp,:nimp]))
        logger.info(mf, 'Full CCSD 1-RDM up diag = \n %s', dm_ao[0][:nimp,:nimp].diagonal())
        logger.info(mf, 'Full CCSD 1-RDM dn diag = \n %s', dm_ao[1][:nimp,:nimp].diagonal())

    # Full and CAS CCGF for self-energy composite method
    if composite:
        if thresh2 is None:
            gf = uccgf.UCCGF(mycc, tol=cc_gmres_tol)
            orbs = range(len(mf.mo_energy[0]))
            g_ip = gf.ipccsd_mo(orbs, orbs, freqs.conj(), delta).conj()
            g_ea = gf.eaccsd_mo(orbs, orbs, freqs, delta)
            gf_cc_full = g_ip + g_ea
        else:
            cc_cas_2 = cc.UCCSD(mf_cas_2)
            cc_cas_2.conv_tol = 1e-8
            cc_cas_2.conv_tol_normt = 1e-5
            cc_cas_2.diis_space = 15
            cc_cas_2.max_cycle = 200
            if rank == 0:
                cc_cas_2.kernel()
                if cc_cas_2.converged is False:
                    log = logger.Logger(sys.stdout, 4)
                    log.warn('!!! CAS2 CCSD not converged !!!')
                cc_cas_2.solve_lambda()
                fn = 'amplitudes_cas_2.h5'
                feri = h5py.File(fn, 'w')
                feri['t1a'] = np.asarray(cc_cas_2.t1[0])
                feri['t1b'] = np.asarray(cc_cas_2.t1[1])
                feri['t2aa'] = np.asarray(cc_cas_2.t2[0])
                feri['t2ab'] = np.asarray(cc_cas_2.t2[1])
                feri['t2bb'] = np.asarray(cc_cas_2.t2[2])
                feri['l1a'] = np.asarray(cc_cas_2.l1[0])
                feri['l1b'] = np.asarray(cc_cas_2.l1[1])
                feri['l2aa'] = np.asarray(cc_cas_2.l2[0])
                feri['l2ab'] = np.asarray(cc_cas_2.l2[1])
                feri['l2bb'] = np.asarray(cc_cas_2.l2[2])
                feri.close()
            comm.Barrier()
            if rank > 0:
                fn = 'amplitudes_cas_2.h5'
                feri = h5py.File(fn, 'r')
                cc_cas_2.t1 = [np.asarray(feri['t1a']), np.asarray(feri['t1b'])]
                cc_cas_2.t2 = [np.asarray(feri['t2aa']), np.asarray(feri['t2ab']), np.asarray(feri['t2bb'])]
                cc_cas_2.l1 = [np.asarray(feri['l1a']), np.asarray(feri['l1b'])]
                cc_cas_2.l2 = [np.asarray(feri['l2aa']), np.asarray(feri['l2ab']), np.asarray(feri['l2bb'])]
                feri.close()
            comm.Barrier()

            gf = uccgf.UCCGF(cc_cas_2, tol=cc_gmres_tol)
            orbs = range(len(mf_cas_2.mo_energy[0]))
            g_ip = gf.ipccsd_mo(orbs, orbs, freqs.conj(), delta).conj()
            g_ea = gf.eaccsd_mo(orbs, orbs, freqs, delta)
            gf_cc_cas_2 = g_ip + g_ea

        cc_cas = cc.UCCSD(mf_cas)
        cc_cas.conv_tol = 1e-8
        cc_cas.conv_tol_normt = 1e-5
        cc_cas.diis_space = 15
        cc_cas.max_cycle = 200
        if rank == 0:
            cc_cas.kernel()
            if cc_cas.converged is False:
                log = logger.Logger(sys.stdout, 4)
                log.warn('!!! CAS CCSD not converged !!!')
            cc_cas.solve_lambda()
            fn = 'amplitudes_cas.h5'
            feri = h5py.File(fn, 'w')
            feri['t1a'] = np.asarray(cc_cas.t1[0])
            feri['t1b'] = np.asarray(cc_cas.t1[1])
            feri['t2aa'] = np.asarray(cc_cas.t2[0])
            feri['t2ab'] = np.asarray(cc_cas.t2[1])
            feri['t2bb'] = np.asarray(cc_cas.t2[2])
            feri['l1a'] = np.asarray(cc_cas.l1[0])
            feri['l1b'] = np.asarray(cc_cas.l1[1])
            feri['l2aa'] = np.asarray(cc_cas.l2[0])
            feri['l2ab'] = np.asarray(cc_cas.l2[1])
            feri['l2bb'] = np.asarray(cc_cas.l2[2])
            feri.close()
        comm.Barrier()
        if rank > 0:
            fn = 'amplitudes_cas.h5'
            feri = h5py.File(fn, 'r')
            cc_cas.t1 = [np.asarray(feri['t1a']), np.asarray(feri['t1b'])]
            cc_cas.t2 = [np.asarray(feri['t2aa']), np.asarray(feri['t2ab']), np.asarray(feri['t2bb'])]
            cc_cas.l1 = [np.asarray(feri['l1a']), np.asarray(feri['l1b'])]
            cc_cas.l2 = [np.asarray(feri['l2aa']), np.asarray(feri['l2ab']), np.asarray(feri['l2bb'])]
            feri.close()
        comm.Barrier()
        dm_cas = cc_cas.make_rdm1()
        dm_cas_ao = np.array(dm_cas)
        for s in range(2):
            dm_cas_ao[s] = np.dot(mf_cas.mo_coeff[s], np.dot(dm_cas[s], mf_cas.mo_coeff[s].T))

        gf = uccgf.UCCGF(cc_cas, tol=cc_gmres_tol)
        orbs = range(len(mf_cas.mo_energy[0]))
        g_ip = gf.ipccsd_mo(orbs, orbs, freqs.conj(), delta).conj()
        g_ea = gf.eaccsd_mo(orbs, orbs, freqs, delta)
        gf_cc_cas = g_ip + g_ea

    if composite:
        if thresh2 is None:
            for s in range(2):
                for iw in range(len(freqs)):
                    gf_cc_full[s,:,:,iw] = np.dot(mf.mo_coeff[s], np.dot(gf_cc_full[s,:,:,iw], mf.mo_coeff[s].T))
        else:
            gf_hf = mf_gf(mf, freqs, delta)
            gf_hf_cas_2 = mf_gf(mf_cas_2, freqs, delta)
            for s in range(2):
                for iw in range(len(freqs)):
                    gf_cc_cas_2[s,:,:,iw] = np.dot(mf_cas_2.mo_coeff[s], np.dot(gf_cc_cas_2[s,:,:,iw], mf_cas_2.mo_coeff[s].T))

            sigma_cas = get_sigma(gf_hf_cas_2, gf_cc_cas_2)
            sigma_full = np.zeros_like(gf_hf)
            spin = gf_hf.shape[0]
            for s in range(spin):
                for iw in range(len(freqs)):
                    sigma_full[s,:,:,iw] = np.dot(no_coeff_2[s], np.dot(sigma_cas[s,:,:,iw], no_coeff_2[s].T))
            gf_cc_full = np.zeros_like(gf_hf)
            for s in range(spin):
                for iw in range(len(freqs)):
                    gf_cc_full[s,:,:,iw] = np.linalg.inv(np.linalg.inv(gf_hf[s,:,:,iw]) - sigma_full[s,:,:,iw])

        for s in range(2):
            for iw in range(len(freqs)):
                gf_cc_cas[s,:,:,iw] = np.dot(mf_cas.mo_coeff[s], np.dot(gf_cc_cas[s,:,:,iw], mf_cas.mo_coeff[s].T))
        return mf_cas, no_coeff, gf_cc_full, gf_cc_cas, dm_ao, dm_cas_ao
    else:
        gf_hf = mf_gf(mf, freqs, delta)
        gf_hf_cas = mf_gf(mf_cas, freqs, delta)
        dm_ao = mf.make_rdm1()
        dm_cas_ao = mf_cas.make_rdm1()
        return mf_cas, no_coeff, gf_hf, gf_hf_cas, dm_ao, dm_cas_ao

def cas_cisd(mf, freqs, delta, nimp=None, thresh=5e-3, nvir_act=None, nocc_act=None,
             local=False, nocc_act_low=None, nvir_act_high=None):
    from fcdmft.solver.casno import make_casno_cisd
    from pyscf import ci
    from pyscf.lib import chkfile

    # Full CISD
    myci = ci.CISD(mf)
    chkfname = 'myci.chk'
    if os.path.isfile(chkfname):
        data = chkfile.load(chkfname, 'cisd')
        myci.__dict__.update(data)
    else:
        myci.chkfile = chkfname
        if rank == 0:
            myci.kernel()
            myci.dump_chk()
        comm.Barrier()
    nmo = myci.nmo
    mf_cas, no_coeff, dm = make_casno_cisd(myci, thresh=thresh, nvir_act=nvir_act, nocc_act=nocc_act,
                                           return_dm=True, local=local, nocc_act_low=nocc_act_low,
                                           nvir_act_high=nvir_act_high)
    dm_ao = np.dot(mf.mo_coeff, np.dot(dm, mf.mo_coeff.T))
    if rank == 0:
        if nimp is None:
            nimp = nmo
        logger.info(mf, 'Full CISD Nelec = %s', np.trace(dm_ao[:nimp,:nimp]))
        logger.info(mf, 'Full CISD 1-RDM diag = \n %s', dm_ao[:nimp,:nimp].diagonal())

    no_coeff = no_coeff[np.newaxis, ...]
    gf_hf = mf_gf(mf, freqs, delta)
    gf_hf_cas = mf_gf(mf_cas, freqs, delta)
    dm_ao = mf.make_rdm1()
    dm_cas_ao = mf_cas.make_rdm1()
    return mf_cas, no_coeff, gf_hf, gf_hf_cas, dm_ao, dm_cas_ao

def cas_hf(mf, freqs, delta, nimp=None, nvir_act=None, nocc_act=None):
    from fcdmft.solver.casno import make_cas_hf

    mf_cas, no_coeff = make_cas_hf(mf, nvir_act=nvir_act, nocc_act=nocc_act)
    dm_ao = mf.make_rdm1()
    nmo = len(mf.mo_energy)
    if rank == 0:
        if nimp is None:
            nimp = nmo
        logger.info(mf, 'Full HF Nelec = %s', np.trace(dm_ao[:nimp,:nimp]))
        logger.info(mf, 'Full HF 1-RDM diag = \n %s', dm_ao[:nimp,:nimp].diagonal())

    no_coeff = no_coeff[np.newaxis, ...]
    gf_hf = mf_gf(mf, freqs, delta)
    gf_hf_cas = mf_gf(mf_cas, freqs, delta)
    dm_cas_ao = mf_cas.make_rdm1()
    return mf_cas, no_coeff, gf_hf, gf_hf_cas, dm_ao, dm_cas_ao

def cc_gf(mf, freqs, delta, ao_orbs=None, gmres_tol=1e-4, nimp=None,
          cas=False, casno='gw', composite=True, thresh=5e-3, nvir_act=None,
          nocc_act=None, save_gf=False, read_gf=False, load_cas=False):
    '''Calculate CCSD GF matrix in the AO basis'''
    from fcdmft.solver import mpiccgf as ccgf

    if ao_orbs is None:
        nmo = mf.mo_coeff.shape[0]
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    if nimp is None:
        nimp = nao

    if cas:
        if casno == 'gw':
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_gw(mf, freqs, delta, composite=False, thresh=thresh,
                               nvir_act=nvir_act, nocc_act=nocc_act)
        elif casno == 'cc':
            assert(not composite)
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_ccsd(mf, freqs, delta, nimp, composite=False, thresh=thresh,
                               nvir_act=nvir_act, nocc_act=nocc_act, load_cc=load_cas)
        elif casno == 'ci':
            assert(not composite)
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_cisd(mf, freqs, delta, nimp, thresh=thresh,
                               nvir_act=nvir_act, nocc_act=nocc_act)
        elif casno == 'hf':
            assert(not composite)
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_hf(mf, freqs, delta, nimp, nvir_act=nvir_act, nocc_act=nocc_act)
        else:
            raise NotImplementedError
        mycc = cc.RCCSD(mf_cas)
        # save nocc_act and nvir_act for later DMFT cycles
        mf.nocc_act = mf_cas.mol.nelectron // 2
        mf.nvir_act = len(mf_cas.mo_energy) - mf.nocc_act
    else:
        mycc = cc.RCCSD(mf)
    mycc.conv_tol = 1e-8
    mycc.conv_tol_normt = 1e-5
    mycc.diis_space = 15
    mycc.max_cycle = 200
    if rank == 0:
        mycc.kernel()
        if mycc.converged is False:
            log = logger.Logger(sys.stdout, 4)
            log.warn('!!! Ground-state CCSD not converged !!!')
        mycc.solve_lambda()
        dm = mycc.make_rdm1()
        if not cas:
            dm_ao = np.dot(mf.mo_coeff, np.dot(dm, mf.mo_coeff.T))
        else:
            dm_cc_cas = np.dot(mf_cas.mo_coeff, np.dot(dm, mf_cas.mo_coeff.T))
            ddm = dm_cc_cas - dm_low_cas
            ddm = np.dot(no_coeff[0], np.dot(ddm, no_coeff[0].T))
            dm_ao = dm_low + ddm
        logger.info(mf, 'CC Nelec = %s', np.trace(dm_ao[:nimp,:nimp]))
        logger.info(mf, 'CC 1-RDM diag = \n %s', dm_ao[:nimp,:nimp].diagonal())

        fn = 'amplitudes.h5'
        feri = h5py.File(fn, 'w')
        feri['t1'] = np.asarray(mycc.t1)
        feri['t2'] = np.asarray(mycc.t2)
        feri['l1'] = np.asarray(mycc.l1)
        feri['l2'] = np.asarray(mycc.l2)
        feri.close()

    comm.Barrier()
    if rank > 0:
        fn = 'amplitudes.h5'
        feri = h5py.File(fn, 'r')
        mycc.t1 = np.asarray(feri['t1'])
        mycc.t2 = np.asarray(feri['t2'])
        mycc.l1 = np.asarray(feri['l1'])
        mycc.l2 = np.asarray(feri['l2'])
        feri.close()
    comm.Barrier()

    gf = ccgf.CCGF(mycc, tol=gmres_tol)
    if cas:
        orbs = range(len(mf_cas.mo_energy))
        if read_gf:
            fn = 'cc_gf.h5'
            feri = h5py.File(fn, 'r')
            gf_cc = np.asarray(feri['gf'])
            freqs0 = np.asarray(feri['freqs'])
            delta0 = np.asarray(feri['delta'])
            feri.close()
            assert(abs(delta0-delta)<1e-5)
            assert(np.max(np.abs(freqs0-freqs))<1e-5)
        else:
            g_ip = gf.ipccsd_ao(orbs, freqs.conj(), mf_cas.mo_coeff, delta).conj()
            g_ea = gf.eaccsd_ao(orbs, freqs, mf_cas.mo_coeff, delta)
            gf_cc = g_ip + g_ea
            gf_cc = gf_cc[np.newaxis, ...]

        if save_gf:
            if rank == 0:
                fn = 'cc_gf.h5'
                feri = h5py.File(fn, 'w')
                feri['gf'] = np.asarray(gf_cc)
                feri['freqs'] = np.asarray(freqs)
                feri['delta'] = delta
                feri.close()
            comm.Barrier()

        sigma_cas = get_sigma(gf_low_cas, gf_cc)
        sigma_full = np.zeros_like(gf_low)
        spin = gf_low.shape[0]
        for s in range(spin):
            for iw in range(len(freqs)):
                sigma_full[s,:,:,iw] = np.dot(no_coeff[s], np.dot(sigma_cas[s,:,:,iw], no_coeff[s].T))
        gf = np.zeros_like(gf_low)
        for s in range(spin):
            for iw in range(len(freqs)):
                gf[s,:,:,iw] = np.linalg.inv(np.linalg.inv(gf_low[s,:,:,iw]) - sigma_full[s,:,:,iw])
    else:
        # Note .conj()'s to make this the retarded GF
        g_ip = gf.ipccsd_ao(ao_orbs, freqs.conj(), mf.mo_coeff, delta).conj()
        g_ea = gf.eaccsd_ao(ao_orbs, freqs, mf.mo_coeff, delta)
        gf = g_ip + g_ea
        gf = gf[np.newaxis, ...]

    return gf[:,:nao,:nao]

def ucc_gf(mf, freqs, delta, ao_orbs=None, gmres_tol=1e-4, nimp=None,
          cas=False, casno='ucc', composite=False, thresh=5e-3, nvir_act_a=None,
          nocc_act_a=None, nvir_act_b=None, nocc_act_b=None,
          save_gf=False, read_gf=False, load_cas=False):
    '''Calculate UCCSD GF matrix in the AO basis'''
    from fcdmft.solver import ucc_eri
    from fcdmft.solver import mpiuccgf as uccgf

    if ao_orbs is None:
        nmo = mf.mo_coeff[0].shape[0]
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    if nimp is None:
        nimp = nao

    if cas:
        if casno == 'ucc':
            assert(not composite)
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_uccsd(mf, freqs, delta, nimp, composite=False, thresh=thresh,
                               nvir_act_a=nvir_act_a, nocc_act_a=nocc_act_a,
                               nvir_act_b=nvir_act_b, nocc_act_b=nocc_act_b, load_cc=load_cas)
        else:
            raise NotImplementedError
        mycc = cc.UCCSD(mf_cas)
        # save nocc_act and nvir_act for later DMFT cycles
        mf.nocc_act_a = (mf_cas.mol.nelectron + mf_cas.mol.spin) // 2
        mf.nvir_act_a = len(mf_cas.mo_energy[0]) - mf.nocc_act_a
        mf.nocc_act_b = (mf_cas.mol.nelectron - mf_cas.mol.spin) // 2
        mf.nvir_act_b = len(mf_cas.mo_energy[1]) - mf.nocc_act_b
    else:
        mycc = cc.UCCSD(mf)
    mycc.conv_tol = 1e-8
    mycc.conv_tol_normt = 1e-5
    mycc.diis_space = 15
    mycc.max_cycle = 200
    if rank == 0:
        mycc.kernel()
        if mycc.converged is False:
            log = logger.Logger(sys.stdout, 4)
            log.warn('!!! Ground-state CCSD not converged !!!')
        mycc.solve_lambda()
        dm = mycc.make_rdm1()
        if not cas:
            dm_ao = np.zeros_like(dm)
            for s in range(2):
                dm_ao[s] = np.dot(mf.mo_coeff[s], np.dot(dm[s], mf.mo_coeff[s].T))
        else:
            dm_ao = np.zeros_like(dm_low)
            for s in range(2):
                dm_cc_cas = np.dot(mf_cas.mo_coeff[s], np.dot(dm[s], mf_cas.mo_coeff[s].T))
                ddm = dm_cc_cas - dm_low_cas[s]
                ddm = np.dot(no_coeff[s], np.dot(ddm, no_coeff[s].T))
                dm_ao[s] = dm_low[s] + ddm
        logger.info(mf, 'CC Nelec_up = %s, Nelec_dn = %s, Nelec = %s',
                    np.trace(dm_ao[0][:nimp,:nimp]),np.trace(dm_ao[1][:nimp,:nimp]),
                    np.trace(dm_ao[0][:nimp,:nimp])+np.trace(dm_ao[1][:nimp,:nimp]))
        logger.info(mf, 'CC 1-RDM up diag = \n %s', dm_ao[0][:nimp,:nimp].diagonal())
        logger.info(mf, 'CC 1-RDM dn diag = \n %s', dm_ao[1][:nimp,:nimp].diagonal())

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

    gf = uccgf.UCCGF(mycc, tol=gmres_tol)
    if cas:
        orbs = range(len(mf_cas.mo_energy[0]))
        if read_gf:
            fn = 'cc_gf.h5'
            feri = h5py.File(fn, 'r')
            gf_cc = np.asarray(feri['gf'])
            freqs0 = np.asarray(feri['freqs'])
            delta0 = np.asarray(feri['delta'])
            feri.close()
            assert(abs(delta0-delta)<1e-5)
            assert(np.max(np.abs(freqs0-freqs))<1e-5)
        else:
            g_ip = gf.ipccsd_ao(orbs, freqs.conj(), mf_cas.mo_coeff, delta).conj()
            g_ea = gf.eaccsd_ao(orbs, freqs, mf_cas.mo_coeff, delta)
            gf_cc = g_ip + g_ea
        if save_gf:
            if rank == 0:
                fn = 'cc_gf.h5'
                feri = h5py.File(fn, 'w')
                feri['gf'] = np.asarray(gf_cc)
                feri['freqs'] = np.asarray(freqs)
                feri['delta'] = delta
                feri.close()
            comm.Barrier()

        sigma_cas = get_sigma(gf_low_cas, gf_cc)
        sigma_full = np.zeros_like(gf_low)
        spin = gf_low.shape[0]
        for s in range(spin):
            for iw in range(len(freqs)):
                sigma_full[s,:,:,iw] = np.dot(no_coeff[s], np.dot(sigma_cas[s,:,:,iw], no_coeff[s].T))
        gf = np.zeros_like(gf_low)
        for s in range(spin):
            for iw in range(len(freqs)):
                gf[s,:,:,iw] = np.linalg.inv(np.linalg.inv(gf_low[s,:,:,iw]) - sigma_full[s,:,:,iw])
    else:
        # Note .conj()'s to make this the retarded GF
        g_ip = gf.ipccsd_ao(ao_orbs, freqs.conj(), mf.mo_coeff, delta).conj()
        g_ea = gf.eaccsd_ao(ao_orbs, freqs, mf.mo_coeff, delta)
        gf = g_ip + g_ea

    return gf[:,:nao,:nao]

def dmrg_gf(mf, freqs, delta, ao_orbs=None, n_threads=7, nimp=None,
          cas=False, casno='gw', composite=True, thresh=5e-3, nvir_act=None, nocc_act=None,
          reorder_method='gaopt', cc_gmres_tol=1e-3, gf_n_steps=10, gs_n_steps=20, gs_tol=1E-13,
          dmrg_verbose=3, gs_bond_dims=[400, 800, 1500], gf_bond_dims=[500], gf_tol=1E-4,
          cps_bond_dims=[2000], cps_noises=[0], cps_tol=1E-13, cps_n_steps=20,
          gmres_tol=1E-9, gs_noises=[1E-3, 1E-5, 1E-7, 1E-11, 0], gf_noises=[1E-4, 1E-7, 0],
          n_off_diag_cg=0, load_dir=None, save_dir=None, save_gf=False, read_gf=False, local=True,
          extra_freqs=None, extra_delta=None, load_cas=False, thresh2=None, dyn_corr_method=None,
          ncore=0, nvirt=0):
    '''Calculate the DMRG GF matrix in the AO basis'''
    from fcdmft.solver.gfdmrg import dmrg_mo_gf

    nmo = mf.mo_coeff.shape[0]
    if ao_orbs is None:
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    if nimp is None:
        nimp = nao

    if load_dir is not None:
        load_cas = True
    if save_dir is not None:
        save_cas = True
    else:
        save_cas = False

    if extra_delta is not None:
        freqs_cas = np.array(extra_freqs).reshape(-1)
        delta_cas = extra_delta
    else:
        freqs_cas = freqs
        delta_cas = delta

    if cas:
        if casno == 'gw':
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_gw(mf, freqs_cas, delta_cas, composite=composite, thresh=thresh,
                               nvir_act=nvir_act, nocc_act=nocc_act, local=local)
        elif casno == 'cc':
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_ccsd(mf, freqs_cas, delta_cas, nimp, composite=composite, thresh=thresh,
                               nvir_act=nvir_act, nocc_act=nocc_act,
                               cc_gmres_tol=cc_gmres_tol, local=local, load_cc=load_cas,
                               save_cas=save_cas, thresh2=thresh2,
                               nocc_act_low=ncore, nvir_act_high=nvirt)
        elif casno == 'ci':
            assert(not composite)
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_cisd(mf, freqs_cas, delta_cas, nimp, thresh=thresh, nvir_act=nvir_act,
                               nocc_act=nocc_act, local=local,
                               nocc_act_low=ncore, nvir_act_high=nvirt)
        elif casno == 'hf':
            assert(not composite)
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_hf(mf, freqs_cas, delta_cas, nimp, nvir_act=nvir_act, nocc_act=nocc_act)
        else:
            raise NotImplementedError
        mf_dmrg = mf_cas
        gf_orbs = range(len(mf_cas.mo_energy))
        # save nocc_act and nvir_act for later DMFT cycles
        mf.nocc_act = mf_cas.mol.nelectron // 2
        mf.nvir_act = len(mf_cas.mo_energy) - mf.nocc_act
    else:
        # Full DMRG
        mf.mol.symmetry = 'c1'
        mf.mol.nelectron = int(round(np.sum(mf.mo_occ)))
        mf_dmrg = mf
        gf_orbs = ao_orbs

    max_memory = mf.max_memory * 1E6

    # set scratch folder
    scratch = './tmp'
    if rank == 0:
        if not os.path.isdir(scratch):
            os.mkdir(scratch)
    comm.Barrier()
    os.environ['TMPDIR'] = scratch

    # set save_dir folder
    if save_dir is not None:
        if rank == 0:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
        comm.Barrier()

    if read_gf:
        fn = 'dmrg_gf.h5'
        feri = h5py.File(fn, 'r')
        gf = np.asarray(feri['gf'])
        dm = np.asarray(feri['dm'])
        freqs0 = np.asarray(feri['freqs'])
        delta0 = np.asarray(feri['delta'])
        feri.close()
        assert(abs(delta0-delta_cas)<1e-5)
        assert(np.max(np.abs(freqs0-freqs_cas))<1e-5)
    else:
        # NOTE: gmres_tol in DMRG-GF measures norm**2
        dm, gf = dmrg_mo_gf(mf_dmrg, freqs=freqs, delta=delta, ao_orbs=gf_orbs, mo_orbs=None,
                       extra_freqs=extra_freqs, extra_delta=extra_delta,
                       scratch=scratch, add_rem='+-', n_threads=n_threads, reorder_method=reorder_method,
                       memory=max_memory, gs_bond_dims=gs_bond_dims, gf_bond_dims=gf_bond_dims,
                       gf_n_steps=gf_n_steps, gs_n_steps=gs_n_steps, gs_tol=gs_tol, gf_noises=gf_noises,
                       gf_tol=gf_tol, gs_noises=gs_noises, gmres_tol=gmres_tol, load_dir=load_dir, save_dir=save_dir,
                       cps_bond_dims=cps_bond_dims, cps_noises=[0], cps_tol=gs_tol, cps_n_steps=gs_n_steps,
                       verbose=dmrg_verbose, mo_basis=False, ignore_ecore=False, n_off_diag_cg=n_off_diag_cg,
                       mpi=True, dyn_corr_method=dyn_corr_method, ncore=ncore, nvirt=nvirt)
        if gf.ndim == 3:
            gf = gf[np.newaxis, ...]

    if save_gf:
        if rank == 0:
            fn = 'dmrg_gf.h5'
            feri = h5py.File(fn, 'w')
            feri['gf'] = np.asarray(gf)
            feri['dm'] = np.asarray(dm)
            feri['freqs'] = np.asarray(freqs_cas)
            feri['delta'] = delta_cas
            feri.close()
        comm.Barrier()
    if cas:
        # assemble CASCI Green's function
        sigma_cas = get_sigma(gf_low_cas, gf)
        sigma_full = np.zeros_like(gf_low)
        spin = gf_low.shape[0]
        for s in range(spin):
            for iw in range(len(freqs_cas)):
                sigma_full[s,:,:,iw] = np.dot(no_coeff[s], np.dot(sigma_cas[s,:,:,iw], no_coeff[s].T))
        gf_full = np.zeros_like(gf_low)
        for s in range(spin):
            for iw in range(len(freqs_cas)):
                gf_full[s,:,:,iw] = np.linalg.inv(np.linalg.inv(gf_low[s,:,:,iw]) - sigma_full[s,:,:,iw])

        # assemble CASCI density matrix
        dm = dm[0] + dm[1]
        ddm = dm - dm_low_cas
        ddm = np.dot(no_coeff[0], np.dot(ddm, no_coeff[0].T))
        dm_cas = dm_low + ddm
    else:
        dm_cas = dm[0] + dm[1]
        gf_full = gf

    if rank == 0:
        logger.info(mf, 'DMRG Nelec = %s', np.trace(dm_cas[:nimp,:nimp]))
        logger.info(mf, 'DMRG 1-RDM diag = \n %s', dm_cas[:nimp,:nimp].diagonal())

    return gf_full[:,:nao,:nao]

def udmrg_gf(mf, freqs, delta, ao_orbs=None, n_threads=7, nimp=None,
          cas=False, casno='ucc', composite=True, thresh=5e-3, nvir_act_a=None,
          nocc_act_a=None, nvir_act_b=None, nocc_act_b=None,
          reorder_method='gaopt', cc_gmres_tol=1e-3, gf_n_steps=10, gs_n_steps=20, gs_tol=1E-13,
          dmrg_verbose=3, gs_bond_dims=[400, 800, 1500], gf_bond_dims=[500], gf_tol=1E-4,
          cps_bond_dims=[2000], cps_noises=[0], cps_tol=1E-13, cps_n_steps=20,
          gmres_tol=1E-9, gs_noises=[1E-3, 1E-5, 1E-7, 1E-11, 0], gf_noises=[1E-4, 1E-7, 0],
          n_off_diag_cg=0, load_dir=None, save_dir=None, save_gf=False, read_gf=False, local=True,
          extra_freqs=None, extra_delta=None, load_cas=False, thresh2=None):
    '''Calculate the spin-unrestricted DMRG GF matrix in the AO basis'''
    from fcdmft.solver.gfdmrg_sz import dmrg_mo_gf

    nmo = mf.mo_coeff[0].shape[0]
    if ao_orbs is None:
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    if nimp is None:
        nimp = nao

    if load_dir is not None:
        load_cas = True
    if save_dir is not None:
        save_cas = True
    else:
        save_cas = False

    if extra_delta is not None:
        freqs_cas = np.array(extra_freqs).reshape(-1)
        delta_cas = extra_delta
    else:
        freqs_cas = freqs
        delta_cas = delta

    if cas:
        if casno == 'ucc':
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_uccsd(mf, freqs_cas, delta_cas, nimp, composite=composite, thresh=thresh,
                               nvir_act_a=nvir_act_a, nocc_act_a=nocc_act_a, nvir_act_b=nvir_act_b,
                               nocc_act_b=nocc_act_b, cc_gmres_tol=cc_gmres_tol, local=local,
                               load_cc=load_cas, save_cas=save_cas, thresh2=thresh2)
        else:
            raise NotImplementedError
        ne_a = (mf_cas.mol.nelectron + mf_cas.mol.spin) // 2
        ne_b = (mf_cas.mol.nelectron - mf_cas.mol.spin) // 2
        mf_cas.mol.nelectron = (ne_a, ne_b)
        mf_dmrg = mf_cas
        gf_orbs = range(len(mf_cas.mo_energy[0]))
        # save nocc_act and nvir_act for later DMFT cycles
        mf.nocc_act_a = ne_a
        mf.nvir_act_a = len(mf_cas.mo_energy[0]) - mf.nocc_act_a
        mf.nocc_act_b = ne_b
        mf.nvir_act_b = len(mf_cas.mo_energy[1]) - mf.nocc_act_b
    else:
        # Full DMRG
        mf.mol.symmetry = 'c1'
        ne_a = int(round(np.sum(mf.mo_occ[0])))
        ne_b = int(round(np.sum(mf.mo_occ[1])))
        mf.mol.nelectron = (ne_a, ne_b)
        mf_dmrg = mf
        gf_orbs = ao_orbs

    max_memory = mf.max_memory * 1E6

    # set scratch folder
    scratch = './tmp'
    if rank == 0:
        if not os.path.isdir(scratch):
            os.mkdir(scratch)
    comm.Barrier()
    os.environ['TMPDIR'] = scratch

    # set save_dir folder
    if save_dir is not None:
        if rank == 0:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
        comm.Barrier()

    if read_gf:
        fn = 'dmrg_gf.h5'
        feri = h5py.File(fn, 'r')
        gf = np.asarray(feri['gf'])
        dm = np.asarray(feri['dm'])
        freqs0 = np.asarray(feri['freqs'])
        delta0 = np.asarray(feri['delta'])
        feri.close()
        assert(abs(delta0-delta_cas)<1e-5)
        assert(np.max(np.abs(freqs0-freqs_cas))<1e-5)
    else:
        # NOTE: gmres_tol in DMRG-GF measures norm**2
        dm, gf = dmrg_mo_gf(mf_dmrg, freqs=freqs, delta=delta, ao_orbs=gf_orbs, mo_orbs=None,
                       extra_freqs=extra_freqs, extra_delta=extra_delta,
                       scratch=scratch, add_rem='+-', n_threads=n_threads, reorder_method=reorder_method,
                       memory=max_memory, gs_bond_dims=gs_bond_dims, gf_bond_dims=gf_bond_dims,
                       gf_n_steps=gf_n_steps, gs_n_steps=gs_n_steps, gs_tol=gs_tol, gf_noises=gf_noises,
                       gf_tol=gf_tol, gs_noises=gs_noises, gmres_tol=gmres_tol, load_dir=load_dir, save_dir=save_dir,
                       cps_bond_dims=cps_bond_dims, cps_noises=[0], cps_tol=gs_tol, cps_n_steps=gs_n_steps,
                       verbose=dmrg_verbose, mo_basis=False, ignore_ecore=False, n_off_diag_cg=n_off_diag_cg, mpi=True)

    if save_gf:
        if rank == 0:
            fn = 'dmrg_gf.h5'
            feri = h5py.File(fn, 'w')
            feri['gf'] = np.asarray(gf)
            feri['dm'] = np.asarray(dm)
            feri['freqs'] = np.asarray(freqs_cas)
            feri['delta'] = delta_cas
            feri.close()
        comm.Barrier()

    if cas:
        # assemble CASCI Green's function
        sigma_cas = get_sigma(gf_low_cas, gf)
        sigma_full = np.zeros_like(gf_low)
        spin = gf_low.shape[0]
        for s in range(spin):
            for iw in range(len(freqs_cas)):
                sigma_full[s,:,:,iw] = np.dot(no_coeff[s], np.dot(sigma_cas[s,:,:,iw], no_coeff[s].T))
        gf_full = np.zeros_like(gf_low)
        for s in range(spin):
            for iw in range(len(freqs_cas)):
                gf_full[s,:,:,iw] = np.linalg.inv(np.linalg.inv(gf_low[s,:,:,iw]) - sigma_full[s,:,:,iw])

        # assemble CASCI density matrix
        ddm = dm - dm_low_cas
        ddm_full = np.zeros_like(dm_low)
        for s in range(spin):
            ddm_full[s] = np.dot(no_coeff[s], np.dot(ddm[s], no_coeff[s].T))
        dm_cas = dm_low + ddm_full
    else:
        dm_cas = dm
        gf_full = gf

    if rank == 0:
        logger.info(mf, 'DMRG Nelec_up = %s, Nelec_dn = %s, Nelec = %s',
                    np.trace(dm_cas[0][:nimp,:nimp]),np.trace(dm_cas[1][:nimp,:nimp]),
                    np.trace(dm_cas[0][:nimp,:nimp])+np.trace(dm_cas[1][:nimp,:nimp]))
        logger.info(mf, 'DMRG 1-RDM up diag = \n %s', dm_cas[0][:nimp,:nimp].diagonal())
        logger.info(mf, 'DMRG 1-RDM dn diag = \n %s', dm_cas[1][:nimp,:nimp].diagonal())

    return gf_full[:,:nao,:nao]

def cc_rdm(mf, ao_orbs=None, cas=False, casno='gw', composite=False,
           thresh=5e-3, nvir_act=None, nocc_act=None, load_cas=False):
    '''Calculate CCSD GF matrix in the AO basis'''
    if ao_orbs is None:
        nmo = mf.mo_coeff.shape[0]
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    nimp = nao

    freqs = np.array([0]); delta = 0.1
    if cas:
        if casno == 'gw':
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_gw(mf, freqs, delta, composite=composite, thresh=thresh,
                               nvir_act=nvir_act, nocc_act=nocc_act)
        elif casno == 'cc':
            assert(not composite)
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_ccsd(mf, freqs, delta, nimp, composite=composite, thresh=thresh,
                               nvir_act=nvir_act, nocc_act=nocc_act, load_cc=load_cas)
        elif casno == 'ci':
            assert(not composite)
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_cisd(mf, freqs, delta, nimp, thresh=thresh,
                               nvir_act=nvir_act, nocc_act=nocc_act)
        elif casno == 'hf':
            assert(not composite)
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_hf(mf, freqs, delta, nimp, nvir_act=nvir_act, nocc_act=nocc_act)
        else:
            raise NotImplementedError
        mycc = cc.RCCSD(mf_cas)
    else:
        mycc = cc.RCCSD(mf)
    mycc.conv_tol = 1e-8
    mycc.conv_tol_normt = 1e-5
    mycc.diis_space = 15
    mycc.max_cycle = 200
    mycc.verbose = 4
    rdm = None
    if rank == 0:
        mycc.kernel()
        mycc.solve_lambda()
        dm = mycc.make_rdm1()
        if not cas:
            rdm = np.dot(mf.mo_coeff, np.dot(dm, mf.mo_coeff.T))
        else:
            dm_cc_cas = np.dot(mf_cas.mo_coeff, np.dot(dm, mf_cas.mo_coeff.T))
            ddm = dm_cc_cas - dm_low_cas
            ddm = np.dot(no_coeff[0], np.dot(ddm, no_coeff[0].T))
            rdm = dm_low + ddm
    comm.Barrier()
    rdm = comm.bcast(rdm,root=0)
    return rdm[:nao,:nao]

def ucc_rdm(mf, ao_orbs=None, cas=False, casno='ucc', composite=False,
            thresh=5e-3, nvir_act_a=None, nocc_act_a=None,
            nvir_act_b=None, nocc_act_b=None, load_cas=False):
    from fcdmft.solver import ucc_eri
    if ao_orbs is None:
        nmo = mf.mo_coeff[0].shape[0]
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    nimp = nao

    freqs = np.array([0]); delta = 0.1
    if cas:
        if casno == 'ucc':
            assert(not composite)
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_uccsd(mf, freqs, delta, nimp, composite=composite, thresh=thresh,
                               nvir_act_a=nvir_act_a, nocc_act_a=nocc_act_a,
                               nvir_act_b=nvir_act_b, nocc_act_b=nocc_act_b, load_cc=load_cas)
        else:
            raise NotImplementedError
        mycc = cc.UCCSD(mf_cas)
    else:
        mycc = cc.UCCSD(mf)
    mycc.conv_tol = 1e-8
    mycc.conv_tol_normt = 1e-5
    mycc.diis_space = 15
    mycc.max_cycle = 200
    mycc.verbose = 4
    dm_ao = None
    if rank == 0:
        mycc.kernel()
        mycc.solve_lambda()
        dm = mycc.make_rdm1()
        if not cas:
            dm_ao = np.zeros_like(dm)
            for s in range(2):
                dm_ao[s] = np.dot(mf.mo_coeff[s], np.dot(dm[s], mf.mo_coeff[s].T))
        else:
            dm_ao = np.zeros_like(dm_low)
            for s in range(2):
                dm_cc_cas = np.dot(mf_cas.mo_coeff[s], np.dot(dm[s], mf_cas.mo_coeff[s].T))
                ddm = dm_cc_cas - dm_low_cas[s]
                ddm = np.dot(no_coeff[s], np.dot(ddm, no_coeff[s].T))
                dm_ao[s] = dm_low[s] + ddm
    comm.Barrier()
    dm_ao = comm.bcast(dm_ao,root=0)

    return dm_ao[0][:nao,:nao] + dm_ao[1][:nao,:nao]

def dmrg_rdm(mf, ao_orbs=None, n_threads=7, cas=False, casno='gw', composite=False,
             thresh=5e-3, nvir_act=None, nocc_act=None, reorder_method='gaopt',
             gs_n_steps=20, gs_tol=1E-13, gs_bond_dims=[400, 800, 1500],
             gs_noises=[1E-3, 1E-5, 1E-7, 1E-11, 0], local=True, load_dir=None, save_dir=None,
             load_cas=False, dyn_corr_method=None, ncore=0, nvirt=0):
    '''Calculate the DMRG GF matrix in the AO basis'''
    from fcdmft.solver.gfdmrg import dmrg_mo_pdm

    nmo = mf.mo_coeff.shape[0]
    if ao_orbs is None:
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    nimp = nao

    if load_dir is not None:
        load_cas = True
    if save_dir is not None:
        save_cas = True
    else:
        save_cas = False

    freqs = np.array([0.]); delta = 0.1
    if cas:
        if casno == 'gw':
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_gw(mf, freqs, delta, composite=False, thresh=thresh,
                               nvir_act=nvir_act, nocc_act=nocc_act, local=local)
        elif casno == 'cc':
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_ccsd(mf, freqs, delta, nimp, composite=False, thresh=thresh,
                               nvir_act=nvir_act, nocc_act=nocc_act,
                               local=local, load_cc=load_cas, save_cas=save_cas,
                               nocc_act_low=ncore, nvir_act_high=nvirt)
        elif casno == 'ci':
            assert(not composite)
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_cisd(mf, freqs, delta, nimp, thresh=thresh,
                               nvir_act=nvir_act, nocc_act=nocc_act, local=local,
                               nocc_act_low=ncore, nvir_act_high=nvirt)
        elif casno == 'hf':
            assert(not composite)
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_hf(mf, freqs, delta, nimp, nvir_act=nvir_act, nocc_act=nocc_act)
        else:
            raise NotImplementedError
        mf_dmrg = mf_cas
        dm_orbs = range(len(mf_cas.mo_energy))
    else:
        # Full DMRG
        mf.mol.symmetry = 'c1'
        mf.mol.nelectron = int(round(np.sum(mf.mo_occ)))
        mf_dmrg = mf
        dm_orbs = ao_orbs

    max_memory = mf.max_memory * 1E6

    # set scratch folder
    scratch = './tmp'
    if rank == 0:
        if not os.path.isdir(scratch):
            os.mkdir(scratch)
    comm.Barrier()
    os.environ['TMPDIR'] = scratch

    # set save_dir folder
    if save_dir is not None:
        if rank == 0:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
        comm.Barrier()

    dm = dmrg_mo_pdm(mf_dmrg, ao_orbs=dm_orbs, mo_orbs=None, scratch=scratch, reorder_method=reorder_method,
                     n_threads=n_threads, memory=max_memory, gs_bond_dims=gs_bond_dims,
                     gs_n_steps=gs_n_steps, gs_tol=gs_tol, gs_noises=gs_noises, load_dir=load_dir,
                     save_dir=save_dir, verbose=1, mo_basis=False, ignore_ecore=False, mpi=True,
                     dyn_corr_method=dyn_corr_method, ncore=ncore, nvirt=nvirt)

    if cas:
        # assemble CASCI density matrix
        dm = dm[0] + dm[1]
        ddm = dm - dm_low_cas
        ddm = np.dot(no_coeff[0], np.dot(ddm, no_coeff[0].T))
        dm_cas = dm_low + ddm
    else:
        dm_cas = dm[0] + dm[1]

    return dm_cas[:nao,:nao]

def udmrg_rdm(mf, ao_orbs=None, n_threads=7, cas=False, casno='ucc', composite=False,
             thresh=5e-3, nvir_act_a=None, nocc_act_a=None, nvir_act_b=None,
             nocc_act_b=None, reorder_method='gaopt', gs_n_steps=20,
             gs_tol=1E-13, gs_bond_dims=[400, 800, 1500], gs_noises=[1E-3, 1E-5, 1E-7, 1E-11, 0],
             load_dir=None, save_dir=None, local=True, load_cas=False):
    '''Calculate the DMRG density matrix in the AO basis'''
    from fcdmft.solver.gfdmrg_sz import dmrg_mo_pdm

    nmo = mf.mo_coeff[0].shape[0]
    if ao_orbs is None:
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    nimp = nao

    if load_dir is not None:
        load_cas = True
    if save_dir is not None:
        save_cas = True
    else:
        save_cas = False

    freqs = np.array([0.]); delta = 0.1
    if cas:
        if casno == 'ucc':
            mf_cas, no_coeff, gf_low, gf_low_cas, dm_low, dm_low_cas = \
                        cas_uccsd(mf, freqs, delta, nimp, composite=False, thresh=thresh,
                               nvir_act_a=nvir_act_a, nocc_act_a=nocc_act_a,
                               nvir_act_b=nvir_act_b, nocc_act_b=nocc_act_b,
                               local=local, load_cc=load_cas, save_cas=save_cas)
        else:
            raise NotImplementedError
        ne_a = (mf_cas.mol.nelectron + mf_cas.mol.spin) // 2
        ne_b = (mf_cas.mol.nelectron - mf_cas.mol.spin) // 2
        mf_cas.mol.nelectron = (ne_a, ne_b)
        mf_dmrg = mf_cas
        dm_orbs = range(len(mf_cas.mo_energy[0]))
    else:
        # Full DMRG
        mf.mol.symmetry = 'c1'
        ne_a = int(round(np.sum(mf.mo_occ[0])))
        ne_b = int(round(np.sum(mf.mo_occ[1])))
        mf.mol.nelectron = (ne_a, ne_b)
        mf_dmrg = mf
        dm_orbs = ao_orbs

    max_memory = mf.max_memory * 1E6

    # set scratch folder
    scratch = './tmp'
    if rank == 0:
        if not os.path.isdir(scratch):
            os.mkdir(scratch)
    comm.Barrier()
    os.environ['TMPDIR'] = scratch

    # set save_dir folder
    if save_dir is not None:
        if rank == 0:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
        comm.Barrier()

    dm = dmrg_mo_pdm(mf_dmrg, ao_orbs=dm_orbs, mo_orbs=None, scratch=scratch, reorder_method=reorder_method,
                     n_threads=n_threads, memory=max_memory, gs_bond_dims=gs_bond_dims,
                     gs_n_steps=gs_n_steps, gs_tol=gs_tol, gs_noises=gs_noises, load_dir=load_dir,
                     save_dir=save_dir, verbose=1, mo_basis=False, ignore_ecore=False, mpi=True)

    if cas:
        ddm = dm - dm_low_cas
        ddm_full = np.zeros_like(dm_low)
        for s in range(2):
            ddm_full[s] = np.dot(no_coeff[s], np.dot(ddm[s], no_coeff[s].T))
        dm_cas = dm_low + ddm_full
    else:
        dm_cas = dm

    return dm_cas[0,:nao,:nao] + dm_cas[1,:nao,:nao]


class FCIsol:
    def __init__(self, HamCheMPS2, theFCI, GSvector, GSenergy):
        assert (fci_)
        assert (isinstance(HamCheMPS2, PyCheMPS2.PyHamiltonian))
        self.HamCheMPS2 = HamCheMPS2
        assert (isinstance(theFCI, PyCheMPS2.PyFCI))
        self.FCI = theFCI
        self.GSvector = GSvector
        self.GSenergy = GSenergy

def fci_kernel(mf):
    norb = mf.mo_coeff.shape[0]
    h0 = 0.
    h1t = np.dot(mf.mo_coeff.T, \
                 np.dot(mf.get_hcore(), mf.mo_coeff))
    erit = ao2mo.incore.full(mf._eri, mf.mo_coeff, compact=False)
    erit = erit.reshape([norb,norb,norb,norb])

    Initializer = PyCheMPS2.PyInitialize()
    Initializer.Init()

    # Setting up the Hamiltonian
    Group = 0
    orbirreps = np.zeros((norb,), dtype=ctypes.c_int)
    HamCheMPS2 = PyCheMPS2.PyHamiltonian(norb, Group, orbirreps)
    HamCheMPS2.setEconst( h0 )
    for cnt1 in range(norb):
        for cnt2 in range(norb):
            HamCheMPS2.setTmat(cnt1, cnt2, h1t[cnt1,cnt2])
            for cnt3 in range(norb):
                for cnt4 in range(norb):
                    HamCheMPS2.setVmat(cnt1, cnt2, cnt3, cnt4, \
                                       erit[cnt1,cnt3,cnt2,cnt4])

    nel = np.count_nonzero(mf.mo_occ)*2
    assert( nel % 2 == 0 )
    Nel_up = nel / 2
    Nel_down = nel / 2
    Irrep = 0
    maxMemWorkMB = 100.0
    FCIverbose = 0
    theFCI = PyCheMPS2.PyFCI( HamCheMPS2, Nel_up, Nel_down, \
                              Irrep, maxMemWorkMB, FCIverbose )
    GSvector = np.zeros( [ theFCI.getVecLength() ], \
                         dtype=ctypes.c_double )
    GSvector[ theFCI.LowestEnergyDeterminant() ] = 1
    EnergyCheMPS2 = theFCI.GSDavidson( GSvector )
    if rank == 0:
        print ('FCI corr = %20.12f' % (EnergyCheMPS2-mf.e_tot))

    fcisol = FCIsol(HamCheMPS2, theFCI, GSvector, EnergyCheMPS2)
    return fcisol

def fci_gf(mf, freqs, delta, ao_orbs=None, gmres_tol=1e-4):
    if ao_orbs is None:
        nmo = mf.mo_coeff.shape[0]
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    nmo = mf.mo_coeff.shape[0]
    nw = len(freqs)
    gf = np.zeros([nmo, nmo, nw], np.complex128)

    orbsLeft  = np.arange(nmo, dtype=ctypes.c_int)
    orbsRight = np.arange(nmo, dtype=ctypes.c_int)

    fcisol = fci_kernel(mf)
    theFCI = fcisol.FCI
    energy_gs = fcisol.GSenergy
    gs_vector = fcisol.GSvector
    HamCheMPS2 = fcisol.HamCheMPS2
    for iw, w in enumerate(freqs):
        if np.iscomplex(w):
            wr = w.real
            wi = w.imag
        else:
            wr = w
            wi = 0.
        ReGF, ImGF = theFCI.GFmatrix_rem (wr-energy_gs, 1.0, wi+delta, \
                orbsLeft, orbsRight, 1, gs_vector, HamCheMPS2)
        gf_ = (ReGF.reshape((nmo,nmo), order='F') + \
               1j*ImGF.reshape((nmo,nmo), order='F')).T

        ReGF, ImGF = theFCI.GFmatrix_add (wr+energy_gs, -1.0, wi+delta, \
                orbsLeft, orbsRight, 1, gs_vector, HamCheMPS2)
        gf_ += ReGF.reshape((nmo,nmo), order='F') + \
               1j*ImGF.reshape((nmo,nmo), order='F')
        gf[:,:,iw] = np.dot(mf.mo_coeff, np.dot(gf_, mf.mo_coeff.T))
    return gf[:nao,:nao]

def fci_rdm(mf, ao_orbs=None):
    if ao_orbs is None:
        nmo = mf.mo_coeff.shape[0]
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    nmo = mf.mo_coeff.shape[0]
    fcisol = fci_kernel(mf)
    theFCI = fcisol.FCI
    gs_vector = fcisol.GSvector
    rdm2 = np.zeros(nmo**4) 
    theFCI.Fill2RDM(gs_vector, rdm2) 
    rdm2 = rdm2.reshape((nmo,nmo,nmo,nmo))
    rdm_mo = np.einsum('ikkj->ij', rdm2.transpose((0,1,3,2)))/(nmo-1)
    rdm = np.dot(mf.mo_coeff, np.dot(rdm_mo, mf.mo_coeff.T))

    return rdm[:nao,:nao]

def get_sigma(mf_gf, corr_gf):
    '''Get self-energy from correlated GF'''
    spin = mf_gf.shape[0]
    nw = mf_gf.shape[-1]
    sigma = np.zeros_like(mf_gf)
    for s in range(spin):
        for iw in range(nw):
            sigma[s,:,:,iw] = linalg.inv(mf_gf[s,:,:,iw]) - linalg.inv(corr_gf[s,:,:,iw])
    return sigma

def get_gf(hcore, sigma, freqs, delta):
    '''
    Green's function at a set of frequencies

    Args:
         hcore : (spin, nao, nao) ndarray
         sigma : (spin, nao, nao) ndarray
         freqs : (nw) ndarray
         delta : float

    Returns:
         gf : (spin, nao, nao, nw) ndarray

    '''
    nw  = len(freqs)
    spin, nao, _ = hcore.shape
    gf = np.zeros([spin,nao, nao, nw], np.complex128)
    for s in range(spin):
        for iw, w in enumerate(freqs):
            gf[s,:,:,iw] = linalg.inv((w+1j*delta)*np.eye(nao)-hcore[s]-sigma[s,:,:,iw])
    return gf
