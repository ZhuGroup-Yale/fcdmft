#!/usr/bin/python

import time, sys, os, h5py
import numpy as np
from scipy import linalg, optimize
from scipy.optimize import least_squares

from pyscf.lib import logger
from pyscf import lib
from fcdmft import solver
from fcdmft.solver import scf_mu as scf
from fcdmft.gw.pbc import krgw_gf
from fcdmft.gw.mol import gw_dc
from fcdmft.utils import write
from fcdmft.dmft.dmft_solver import mf_kernel, \
                cc_gf, ucc_gf, dmrg_gf, udmrg_gf, cc_rdm, ucc_rdm, \
                dmrg_rdm, udmrg_rdm, fci_gf, fci_rdm, get_gf, get_sigma
from mpi4py import MPI

einsum = lib.einsum

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

# ****************************************************************************
# core routines: kernel, mu_fit
# ****************************************************************************

def kernel(dmft, mu, wl=None, wh=None, occupancy=None, delta=None,
           conv_tol=None, opt_mu=False, dump_chk=True):
    '''DMFT self-consistency cycle at fixed mu'''
    cput0 = (time.process_time(), time.perf_counter())

    # set DMFT parameters
    if delta is None:
        delta = dmft.delta
    if conv_tol is None:
        conv_tol = dmft.conv_tol

    hcore_k = dmft.hcore_k
    JK_k = dmft.JK_k
    DM_k = dmft.DM_k
    eris = dmft.eris
    nval = dmft.nval
    ncore = dmft.ncore
    nb_per_e = dmft.nb_per_e

    spin, nkpts, nao, nao = hcore_k.shape
    hcore_cell = 1./nkpts * np.sum(hcore_k, axis=1)
    JK_cell = 1./nkpts * np.sum(JK_k, axis=1)
    DM_cell = 1./nkpts * np.sum(DM_k, axis=1)

    if np.iscomplexobj(hcore_cell):
        assert (np.max(np.abs(hcore_cell.imag)) < 1e-6)
        assert (np.max(np.abs(JK_cell.imag)) < 1e-6)
        assert (np.max(np.abs(DM_cell.imag)) < 1e-6)
        hcore_cell = hcore_cell.real
        JK_cell = JK_cell.real
        DM_cell = DM_cell.real

    # JK_00: double counting term (Hartree and exchange)
    JK_00 = scf._get_veff(DM_cell, eris)
    himp_cell = hcore_cell + JK_cell - JK_00
    dmft.JK_00 = JK_00

    nw = dmft.nbath
    if wl is None and wh is None:
        wl, wh = -0.4+mu, 0.4+mu
    else:
        wl, wh = wl+mu, wh+mu

    if dmft.disc_type == 'linear':
        freqs, wts = _get_linear_freqs(wl, wh, nw)
    elif dmft.disc_type == 'gauss':
        freqs, wts = _get_scaled_legendre_roots(wl, wh, nw)
    elif dmft.disc_type == 'direct':
        nw_org = nw
        wmult = 3
        nw = nw * wmult + 1
        freqs, wts = _get_linear_freqs(wl, wh, nw)
    elif dmft.disc_type == 'log':
        nw_org = nw
        wmult = 3
        nw = nw * wmult + 1
        freqs = _get_log_freqs(wl, wh, nw)
        wts = None
    elif dmft.disc_type == 'opt':
        nw_org = nw
        wmult = 3
        # stop optimizing bath energies after max_opt_cycle
        max_opt_cycle = 3
        # choose initial guess ('direct' or 'log') and fitting grids
        opt_init = dmft.opt_init_method
        nw = nw * wmult + 1
        if opt_init == 'direct':
            freqs, wts = _get_linear_freqs(wl, wh, nw)
        elif opt_init == 'log':
            freqs = _get_log_freqs(wl, wh, nw)
            wts = None

    dmft.freqs = freqs
    dmft.wts = wts
    if rank == 0:
        logger.info(dmft, 'bath discretization wl = %s, wh = %s', wl, wh)
        logger.info(dmft, 'discretization grids = \n %s', freqs)

    if dmft.gw_dmft:
        # Compute impurity GW self-energy (DC term)
        sigma_gw_imp = dmft.get_gw_sigma(freqs, delta)

        # Compute k-point GW self-energy at given freqs and delta
        sigma_kgw = dmft.get_kgw_sigma(freqs, delta)
        if dmft.twist_average:
            sigma_kgw_band = dmft.get_kgw_sigma_TA(freqs, delta)
    else:
        sigma_gw_imp = np.zeros((spin, nao, nao, nw), dtype=np.complex)
        sigma_kgw = np.zeros((spin, nkpts, nao, nao, nw), dtype=np.complex)
        if dmft.twist_average:
            nkpts_band = dmft.hcore_k_band.shape[1]
            sigma_kgw_band = np.zeros((spin, nkpts_band, nao, nao, nw), dtype=np.complex)

    # write GW self-energy (trace)
    tmpdir = 'dmft_tmp'
    if rank == 0:
        if not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)
        if dmft.gw_dmft:
            write.write_sigma(tmpdir+'/sigma_gw_dc', freqs, sigma_gw_imp)
            write.write_sigma(tmpdir+'/sigma_kgw', freqs, 1./nkpts * sigma_kgw.sum(axis=1))
    comm.Barrier()

    # turn off comment to read saved DMFT quantities
    '''
    if os.path.isfile('DMFT_chk.h5'):
        with h5py.File(dmft.chkfile, 'r') as fh5:
            sigma0 = np.array(fh5['dmft/sigma'])
        assert (sigma0.shape == (spin,nao,nao,nw))
        sigma = sigma0.copy()
    else:
        sigma = np.zeros([spin,nao,nao,nw], dtype=np.complex)
    '''
    sigma = np.zeros([spin,nao,nao,nw], dtype=np.complex)
    dmft.sigma = sigma

    # gf0_cell: local GF; gf_cell: lattice GF
    gf0_cell = get_gf((hcore_cell+JK_cell)[:,ncore:nval,ncore:nval],
                      sigma_gw_imp[:,ncore:nval,ncore:nval], freqs, delta)
    gf_cell = np.zeros([spin, nao, nao, nw], np.complex128)
    for k in range(nkpts):
        gf_cell += 1./nkpts * get_gf(hcore_k[:,k]+JK_k[:,k], sigma+sigma_kgw[:,k], freqs, delta)

    if dmft.twist_average:
        nkpts_band = dmft.hcore_k_band.shape[1]
        gf_cell = nkpts * gf_cell
        for k in range(nkpts_band):
            gf_cell += get_gf(dmft.hcore_k_band[:,k]+dmft.JK_k_band[:,k], sigma+sigma_kgw_band[:,k], freqs, delta)
        gf_cell = 1./(nkpts+nkpts_band) * gf_cell

    if dmft.band_interpolate:
        nkpts_band = dmft.hcore_k_band.shape[1]
        gf_cell = np.zeros([spin, nao, nao, nw], np.complex128)
        mem_now = lib.current_memory()[0]
        mem_band = spin * nao**2 * nw * nkpts_band * 16/1e6
        if mem_now+mem_band < 0.5 * dmft.max_memory:
            if dmft.gw_dmft:
                sigma_kgw_diff_band = dmft.get_kgw_sigma_interpolate(freqs, delta)
            else:
                sigma_kgw_diff_band = dmft.get_khf_sigma_interpolate(freqs, delta)
            for k in range(nkpts_band):
                gf_cell += 1./nkpts_band * get_gf(dmft.hcore_k_band[:,k]+dmft.JK_k_dft_band[:,k], \
                        sigma+sigma_kgw_diff_band[:,k], freqs, delta)
        else:
            # calculate sigma_kgw_diff_band over a slice of kpts each time to save memory 
            nslice = int(mem_band // (0.5*(dmft.max_memory - mem_now))) + 1
            nkpts_slice = (nkpts_band + nslice - 1) // nslice
            for i in range(nslice):
                k_range = np.arange(i*nkpts_slice, min((i+1)*nkpts_slice, nkpts_band))
                if dmft.gw_dmft:
                    sigma_kgw_diff_band = dmft.get_kgw_sigma_interpolate(freqs, delta, k_range)
                else:
                    sigma_kgw_diff_band = dmft.get_khf_sigma_interpolate(freqs, delta, k_range)
                for k in k_range:
                    gf_cell += 1./nkpts_band * get_gf(dmft.hcore_k_band[:,k]+dmft.JK_k_dft_band[:,k], \
                            sigma+sigma_kgw_diff_band[:,k-k_range[0]], freqs, delta)
        comm.Barrier()
        sigma_kgw_diff_band = None

    hyb = get_sigma(gf0_cell, gf_cell[:,ncore:nval,ncore:nval])

    if isinstance(dmft.diis, lib.diis.DIIS):
        dmft_diis = dmft.diis
    elif dmft.diis:
        dmft_diis = lib.diis.DIIS(dmft, dmft.diis_file)
        dmft_diis.space = dmft.diis_space
    else:
        dmft_diis = None
    diis_start_cycle = dmft.diis_start_cycle

    dmft_conv = False
    cycle = 0
    if rank == 0:
        cput1 = logger.timer(dmft, 'initialize DMFT', *cput0)
    while not dmft_conv and cycle < max(1, dmft.max_cycle):
        hyb_last = hyb

        if dmft.disc_type == 'direct' or dmft.disc_type == 'log':
            bath_v, bath_e = get_bath_direct(hyb, freqs, nw_org)
            if rank == 0:
                logger.info(dmft, 'bath energies = \n %s', bath_e[0][:nw_org])
        elif dmft.disc_type == 'opt':
            if cycle < max_opt_cycle:
                if cycle == 0:
                    bath_v, bath_e = get_bath_direct(hyb, freqs, nw_org)
                    bath_v = bath_v.reshape(spin,nval-ncore,nval-ncore,nw_org)
                    bath_v = bath_v[:,:,(nval-ncore-nb_per_e):,:].reshape(spin,nval-ncore,-1)
                    bath_e = bath_e.reshape(spin,nval-ncore,nw_org)
                    bath_e = bath_e[:,(nval-ncore-nb_per_e):,:].reshape(spin,-1)
                if rank == 0:
                    logger.info(dmft, 'initial bath energies = \n %s', bath_e[0][:nw_org])
                    bath_e, bath_v = opt_bath(bath_e, bath_v, hyb, freqs, delta, nw_org,
                                              diag_only=dmft.diag_only, orb_fit=dmft.orb_fit)
                    logger.info(dmft, 'optimized bath energies = \n %s', bath_e[0][:nw_org])
            else:
                if rank == 0:
                    bath_v = opt_bath_v_only(bath_e, bath_v, hyb, freqs, delta, nw_org,
                                            diag_only=dmft.diag_only, orb_fit=dmft.orb_fit)
            comm.Barrier()
            bath_e = comm.bcast(bath_e,root=0)
            bath_v = comm.bcast(bath_v,root=0)
        else:
            bath_v, bath_e = get_bath(hyb, freqs, wts)

        comm.Barrier()
        bath_e = comm.bcast(bath_e,root=0)
        bath_v = comm.bcast(bath_v,root=0)

        # construct embedding Hamiltonian
        himp, eri_imp = imp_ham(himp_cell, eris, bath_v, bath_e, ncore)

        # get initial guess of impurity 1-RDM
        nimp = himp.shape[1]
        dm0 = np.zeros((spin,nimp,nimp))
        if cycle == 0:
            dm0[:,:nao,:nao] = DM_cell.copy()
        else:
            dm0[:,:nao,:nao] = dm_last[:,:nao,:nao].copy()

        # optimize chemical potential for correct number of impurity electrons
        if opt_mu:
            mu = mu_fit(dmft, mu, occupancy, himp, eri_imp, dm0)
            comm.Barrier()
            mu = comm.bcast(mu, root=0)
            comm.Barrier()

        # run HF for embedding problem
        dmft._scf = mf_kernel(himp, eri_imp, mu, nao, dm0,
                              max_mem=dmft.max_memory, verbose=dmft.verbose)
        del eri_imp
        if dmft.max_cycle <= 1:
            break
        dm_last = dmft._scf.make_rdm1()
        if len(dm_last.shape) == 2:
            dm_last = dm_last[np.newaxis, ...]

        '''
        Run impurity solver calculation to get self-energy.
        When delta is small, CCSD imp self-energy can be non-causal if computed directly
        from imp CCSD-GF; instead, it is safer (but more expensive) to first compute
        imp+bath self-energy, then take the imp block of self-energy.
        '''
        # TODO: implement and test DMRG solver
        if delta >= 0.05:
            # imp GF -> imp sigma
            gf_imp = dmft.get_gf_imp(freqs, delta)
            if dmft.solver_type == 'cc' or dmft.solver_type == 'ucc':
                gf_imp = 0.5 * (gf_imp+gf_imp.transpose(0,2,1,3))

            sgdum = np.zeros((spin,nimp,nimp,nw))
            gf0_imp = get_gf(himp, sgdum, freqs, delta)
            gf0_imp = gf0_imp[:,:nao,:nao,:]
            sigma_imp = get_sigma(gf0_imp, gf_imp)
        else:
            # imp+bath GF -> imp+bath sigma -> imp sigma
            sigma_imp = dmft.get_sigma_imp(freqs, delta)
            sigma_imp = sigma_imp[:,:nao,:nao]

        if dmft.cas and cycle == 0:
            if spin == 1:
                dmft.nocc_act = dmft._scf.nocc_act
                dmft.nvir_act = dmft._scf.nvir_act
            else:
                dmft.nocc_act_a = dmft._scf.nocc_act_a
                dmft.nvir_act_a = dmft._scf.nvir_act_a
                dmft.nocc_act_b = dmft._scf.nocc_act_b
                dmft.nvir_act_b = dmft._scf.nvir_act_b

        # remove GW double counting term
        for w in range(nw):
            sigma[:,:,:,w] = sigma_imp[:,:,:,w] - JK_00
        sigma = sigma - sigma_gw_imp

        # update local and lattice GF
        gf0_cell = get_gf((himp_cell)[:,ncore:nval,ncore:nval], sigma_imp[:,ncore:nval,ncore:nval], freqs, delta)
        gf_cell = np.zeros([spin, nao, nao, nw], np.complex128)
        for k in range(nkpts):
            gf_cell += 1./nkpts * get_gf(hcore_k[:,k]+JK_k[:,k], sigma+sigma_kgw[:,k], freqs, delta)

        if dmft.twist_average:
            nkpts_band = dmft.hcore_k_band.shape[1]
            gf_cell = nkpts * gf_cell
            for k in range(nkpts_band):
                gf_cell += get_gf(dmft.hcore_k_band[:,k]+dmft.JK_k_band[:,k], sigma+sigma_kgw_band[:,k], freqs, delta)
            gf_cell = 1./(nkpts+nkpts_band) * gf_cell

        if dmft.band_interpolate:
            nkpts_band = dmft.hcore_k_band.shape[1]
            gf_cell = np.zeros([spin, nao, nao, nw], np.complex128)
            mem_now = lib.current_memory()[0]
            mem_band = spin * nao**2 * nw * nkpts_band * 16/1e6
            if mem_now+mem_band < 0.5 * dmft.max_memory:
                if dmft.gw_dmft:
                    sigma_kgw_diff_band = dmft.get_kgw_sigma_interpolate(freqs, delta)
                else:
                    sigma_kgw_diff_band = dmft.get_khf_sigma_interpolate(freqs, delta)
                for k in range(nkpts_band):
                    gf_cell += 1./nkpts_band * get_gf(dmft.hcore_k_band[:,k]+dmft.JK_k_dft_band[:,k], \
                            sigma+sigma_kgw_diff_band[:,k], freqs, delta)
            else:
                # calculate sigma_kgw_diff_band over a slice of kpts each time to save memory 
                nslice = int(mem_band // (0.5*(dmft.max_memory - mem_now))) + 1
                nkpts_slice = (nkpts_band + nslice - 1) // nslice
                for i in range(nslice):
                    k_range = np.arange(i*nkpts_slice, min((i+1)*nkpts_slice, nkpts_band))
                    if dmft.gw_dmft:
                        sigma_kgw_diff_band = dmft.get_kgw_sigma_interpolate(freqs, delta, k_range)
                    else:
                        sigma_kgw_diff_band = dmft.get_khf_sigma_interpolate(freqs, delta, k_range)
                    for k in k_range:
                        gf_cell += 1./nkpts_band * get_gf(dmft.hcore_k_band[:,k]+dmft.JK_k_dft_band[:,k], \
                                sigma+sigma_kgw_diff_band[:,k-k_range[0]], freqs, delta)
            comm.Barrier()
            sigma_kgw_diff_band = None

        hyb_new = get_sigma(gf0_cell, gf_cell[:,ncore:nval,ncore:nval])

        # write lattice GF and sigma during self-consistent loop
        if rank == 0:
            write.write_sigma(tmpdir+'/dmft_sigma_imp_iter', freqs, sigma)
            write.write_gf_to_dos(tmpdir+'/dmft_latt_dos_iter', freqs, gf_cell)
        comm.Barrier()

        damp = dmft.damp
        if rank == 0:
            if (abs(damp) > 1e-4 and
                (0 <= cycle < diis_start_cycle-1 or dmft_diis is None)):
                hyb_new = damp*hyb_new + (1-damp)*hyb
            hyb = dmft.run_diis(hyb_new, cycle, dmft_diis)
        comm.Barrier()
        hyb = comm.bcast(hyb,root=0)
        dmft.hyb = hyb
        dmft.sigma = sigma

        norm_dhyb = np.linalg.norm(hyb-hyb_last)
        if rank == 0:
            logger.info(dmft, 'cycle= %d  |dhyb|= %4.3g', cycle+1, norm_dhyb)

        if (norm_dhyb < conv_tol):
            dmft_conv = True
        if dump_chk and dmft.chkfile:
            if rank == 0:
                dmft.dump_chk()
        comm.Barrier()

        if rank == 0:
            cput1 = logger.timer(dmft, 'cycle= %d'%(cycle+1), *cput1)
        cycle += 1

    comm.Barrier()
    if rank == 0:
        logger.timer(dmft, 'DMFT_cycle', *cput0)

    if dmft.save_mf:
        if rank == 0:
            fn = 'dmft_scf.h5'
            feri = h5py.File(fn, 'w')
            feri['mo_coeff'] = np.asarray(dmft._scf.mo_coeff)
            feri['mo_energy'] = np.asarray(dmft._scf.mo_energy)
            feri['mo_occ'] = np.asarray(dmft._scf.mo_occ)
            feri['himp'] = np.asarray(himp)
            feri['eri'] = np.asarray(dmft._scf._eri)
            feri.close()
        comm.Barrier()

    if dmft.load_mf:
        from pyscf import ao2mo
        fn = 'dmft_scf.h5'
        feri = h5py.File(fn, 'r')
        dmft._scf.mo_coeff = np.array(feri['mo_coeff'])
        dmft._scf.mo_energy = np.array(feri['mo_energy'])
        dmft._scf.mo_occ = np.array(feri['mo_occ'])
        himp = np.array(feri['himp'])
        dmft._scf._eri = np.array(feri['eri'])
        feri.close()
        spin, n = himp.shape[0:2]
        if spin == 1:
            dmft._scf.get_hcore = lambda *args: himp[0]
        else:
            dmft._scf.get_hcore = lambda *args: himp

    return dmft_conv, mu


def mu_fit(dmft, mu0, occupancy, himp, eri_imp, dm0, step=0.03, trust_region=0.05,
           nelec_tol=3e-3, max_cycle=5):
    '''
    Fit chemical potential to find target impurity occupancy
    '''
    mu_cycle = 0
    dmu = 0
    record = []
    if rank == 0:
        logger.info(dmft, '### Start chemical potential fitting ###')

    while mu_cycle < max_cycle:
        # run HF for embedding problem
        mu = mu0 + dmu
        dmft._scf = mf_kernel(himp, eri_imp, mu, dmft.nao, dm0,
                              max_mem=dmft.max_memory, verbose=4)

        # run ground-state impurity solver to get 1-rdm
        rdm = dmft.get_rdm_imp()
        nelec = np.trace(rdm)
        if mu_cycle > 0:
            dnelec_old = dnelec
        dnelec = nelec - occupancy
        if abs(dnelec) < nelec_tol * occupancy:
            break
        if mu_cycle > 0:
            if abs(dnelec - dnelec_old) < 1e-3:
                if rank == 0:
                    logger.info(dmft, 'Electron number not affected by dmu, quit mu_fit')
                break
        record.append([dmu, dnelec])

        if mu_cycle == 0:
            if dnelec > 0:
                dmu = -1. * step
            else:
                dmu = step
        elif len(record) == 2:
            # linear fit
            dmu1 = record[0][0]; dnelec1 = record[0][1]
            dmu2 = record[1][0]; dnelec2 = record[1][1]
            dmu = (dmu1*dnelec2 - dmu2*dnelec1) / (dnelec2 - dnelec1)
        else:
            # linear fit
            dmu_fit = []
            dnelec_fit = []
            for rec in record:
                dmu_fit.append(rec[0])
                dnelec_fit.append(rec[1])
            dmu_fit = np.array(dmu_fit)
            dnelec_fit = np.array(dnelec_fit)
            idx = np.argsort(np.abs(dnelec_fit))[:2]
            dmu_fit = dmu_fit[idx]
            dnelec_fit = dnelec_fit[idx]
            a,b = np.polyfit(dmu_fit, dnelec_fit, deg=1)
            dmu = -b/a

        if abs(dmu) > trust_region:
            if dmu < 0:
                dmu = -trust_region
            else:
                dmu = trust_region
        if rank == 0:
            logger.info(dmft, 'mu_cycle = %s, mu = %s, nelec = %s, dmu = %s',
                        mu_cycle+1, mu, nelec, dmu)
        mu_cycle += 1

    if rank == 0:
        logger.info(dmft, 'Optimized mu = %s, Nelec = %s, Target = %s', mu, nelec, occupancy)

    return mu

# ****************************************************************************
# bath numerical optimization : opt_bath, opt_bath_v_only
# ****************************************************************************

def opt_bath(bath_e, bath_v, hyb, freqs, delta, nw_org, diag_only=False, orb_fit=None):
    '''
    Optimize bath energies and couplings for minimizing bath discretization error

    Args:
         bath_e : (spin, nb_per_e * nw_org) ndarray
         bath_v : (spin, nimp, nb_per_e * nw_org) ndarray
         hyb : (spin, nimp, nimp, nw) ndarray
         freqs : (nw) 1darray, fitting grids
         delta : float
         nw_org : interger, number of bath energies
         diag_only : bool, only fit diagonal hybridization
         orb_fit : list, orbitals with x5 weight in optimization

    Returns:
         bath_e_opt : (spin, nb_per_e * nw_org) ndarray
         bath_v_opt : (spin, nimp, nb_per_e * nw_org) ndarray
    '''
    # TODO: allow different number of bath orbitals at bath energies
    # TODO: choose to add optimization weights for different hyb elements (diag, 3d/4d)
    spin, nimp, nbath = bath_v.shape
    nb_per_e = nbath // nw_org
    v_opt = np.zeros((spin, nw_org+nimp*nbath))
    min_bound = []; max_bound = []
    for i in range(nw_org+nimp*nbath):
        if i < nw_org:
            min_bound.append(freqs[0])
            max_bound.append(freqs[-1])
        else:
            min_bound.append(-np.inf)
            max_bound.append(np.inf)

    for s in range(spin):
        if s == 0:
            v0 = np.concatenate([bath_e[s][:nw_org], bath_v[s].reshape(-1)])
            try:
                xopt = least_squares(bath_fit, v0, jac='2-point', method='trf', bounds=(min_bound,max_bound), xtol=1e-8,
                     gtol=1e-6, max_nfev=500, verbose=1, args=(hyb[s], bath_v[s], freqs, delta, nw_org, diag_only, orb_fit))
            except:
                xopt = least_squares(bath_fit, v0, jac='2-point', method='lm', xtol=1e-8,
                     gtol=1e-6, max_nfev=500, verbose=1, args=(hyb[s], bath_v[s], freqs, delta, nw_org, diag_only, orb_fit))
            v_opt[s] = xopt.x.copy()
        else:
            v0 = bath_v[s].reshape(-1)
            try:
                xopt = least_squares(bath_fit_v, v0, jac='2-point', method='trf', xtol=1e-8,
                             gtol=1e-6, max_nfev=500, verbose=1,
                             args=(v_opt[0][:nw_org], hyb[s], bath_v[s], freqs, delta, nw_org, diag_only, orb_fit))
            except:
                xopt = least_squares(bath_fit_v, v0, jac='2-point', method='lm', xtol=1e-8,
                             gtol=1e-6, max_nfev=500, verbose=1,
                             args=(v_opt[0][:nw_org], hyb[s], bath_v[s], freqs, delta, nw_org, diag_only, orb_fit))
            v_opt[s][nw_org:] = xopt.x.copy()
            v_opt[s][:nw_org] = v_opt[0][:nw_org]

    bath_e_opt = np.zeros_like(bath_e)
    bath_v_opt = np.zeros_like(bath_v)
    for s in range(spin):
        bath_v_opt[s] = v_opt[s][nw_org:].reshape(nimp, nbath)
        en = v_opt[s][:nw_org]
        for ip in range(nb_per_e):
            for iw in range(nw_org):
                bath_e_opt[s, ip*nw_org + iw] = en[iw]

    return bath_e_opt, bath_v_opt

def bath_fit(v, hyb, bath_v, omega, delta, nw_org, diag_only, orb_fit):
    '''
    Least square of hybridization fitting error
    '''
    nimp, nbath = bath_v.shape
    nb_per_e = nbath // nw_org
    en = v[:nw_org]
    v = v[nw_org:].reshape(nimp, nb_per_e, nw_org)
    w_en = 1./(omega[:,None] + 1j*delta - en[None,:])
    if not diag_only:
        J = einsum('ikn,jkn->ijn',v,v)
        hyb_now = einsum('ijn,wn->ijw',J,w_en)
        f = hyb_now - hyb
        if orb_fit:
            for i in orb_fit:
                f[i,i,:] = 5. * f[i,i,:]
    else:
        J = einsum('ikn,ikn->in',v,v)
        hyb_now = einsum('in,wn->iw',J,w_en)
        f = hyb_now - einsum('iiw->iw', hyb)
        if orb_fit:
            for i in orb_fit:
                f[i,:] = 5. * f[i,:]
    return np.array([f.real,f.imag]).reshape(-1)

def opt_bath_v_only(bath_e, bath_v, hyb, freqs, delta, nw_org, diag_only=False, orb_fit=None):
    '''
    Optimize bath couplings only for minimizing bath discretization error
    '''
    spin, nimp, nbath = bath_v.shape
    v_opt = np.zeros((spin, nimp*nbath))
    bath_v_opt = np.zeros_like(bath_v)
    for s in range(spin):
        v0 = bath_v[s].reshape(-1)
        try:
            xopt = least_squares(bath_fit_v, v0, jac='2-point', method='trf', xtol=1e-10,
                         gtol = 1e-10, max_nfev=500, verbose=1,
                         args=(bath_e[s][:nw_org], hyb[s], bath_v[s], freqs, delta, nw_org, diag_only, orb_fit))
        except:
            xopt = least_squares(bath_fit_v, v0, jac='2-point', method='lm', xtol=1e-10,
                         gtol = 1e-10, max_nfev=500, verbose=1,
                         args=(bath_e[s][:nw_org], hyb[s], bath_v[s], freqs, delta, nw_org, diag_only, orb_fit))
        v_opt[s] = xopt.x.copy()
        bath_v_opt[s] = v_opt[s].reshape(nimp, nbath)

    return bath_v_opt

def bath_fit_v(v, bath_e, hyb, bath_v, omega, delta, nw_org, diag_only, orb_fit):
    '''
    Least square of hybridization fitting error
    '''
    nval, nbath  = bath_v.shape
    nb_per_e = nbath // nw_org
    v = v.reshape(nval, nb_per_e, nw_org)
    w_en = 1./(omega[:,None] + 1j*delta - bath_e[None,:])
    if not diag_only:
        J = einsum('ikn,jkn->ijn',v,v)
        hyb_now = einsum('ijn,wn->ijw',J,w_en)
        f = hyb_now - hyb
        if orb_fit:
            for i in orb_fit:
                f[i,i,:] = 5. * f[i,i,:]
    else:
        J = einsum('ikn,ikn->in',v,v)
        hyb_now = einsum('in,wn->iw',J,w_en)
        f = hyb_now - einsum('iiw->iw', hyb)
        if orb_fit:
            for i in orb_fit:
                f[i,:] = 5. * f[i,:]
    return np.array([f.real,f.imag]).reshape(-1)

# ****************************************************************************
# bath discretization grids : legendre, linear, log 
# ****************************************************************************

def _get_scaled_legendre_roots(wl, wh, nw):
    '''
    Scale nw Legendre roots, which lie in the
    interval [-1, 1], so that they lie in [wl, wh]

    Returns:
        freqs : 1D ndarray
        wts : 1D ndarray
    '''
    freqs, wts = np.polynomial.legendre.leggauss(nw)
    freqs += 1
    freqs *= (wh - wl) / 2.
    freqs += wl
    wts *= (wh - wl) / 2.
    return freqs, wts

def _get_linear_freqs(wl, wh, nw):
    freqs = np.linspace(wl, wh, nw)
    wts = np.ones([nw]) * (wh - wl) / (nw - 1.)
    return freqs, wts

def _get_log_freqs(wl, wh, nw, expo=1.3):
    '''
    Scale nw logorithmic roots on [wl, wh],
    with a given exponent

    Returns:
        freqs : 1D ndarray
    '''
    if (nw % 2 == 1):
        n = nw // 2
        nlist = np.arange(n)
        wpos = 1./(expo ** (nlist))
        freqs1 = (wh + wl)/2. + wpos * (wh - wl)/2.
        freqs2 = (wh + wl)/2. - wpos * (wh - wl)/2.
        freqs = np.concatenate([freqs1,freqs2,[(wh+wl)/2.]])
        freqs = np.sort(freqs)
    if (nw % 2 == 0):
        n = nw // 2
        nlist = np.arange(n-1)
        nlist = np.append(nlist,[n+n//5])
        wpos = 1./(expo ** (nlist))
        freqs1 = (wh + wl)/2. + wpos * (wh - wl)/2.
        freqs2 = (wh + wl)/2. - wpos * (wh - wl)/2.
        freqs = np.concatenate([freqs1,freqs2])
        freqs = np.sort(freqs)
    return freqs

# ****************************************************************************
# Hamiltonian routines : imp_ham, get_bath
# ****************************************************************************

def imp_ham(hcore_cell, eri_cell, bath_v, bath_e, ncore):
    '''
    Construct impurity Hamiltonian

    Args:
         hcore_cell: (spin, nimp, nimp) ndarray
         eri_cell: (spin*(spin+1)/2, nimp*4) ndarray
         bath_v: (spin, nval, nval*nw) ndarray
         bath_e: (spin, nval*nw) ndarray
         ncore: interger

    Returns:
         himp: (spin, nimp+nb, nimp+nb) ndarray
         eri_imp: (spin*(spin+1)/2, (nimp+nb)*4) ndarray
    '''
    spin, nao = hcore_cell.shape[0:2]
    nbath = bath_e.shape[-1]
    nval = bath_v.shape[1] + ncore
    himp = np.zeros([spin, nao+nbath, nao+nbath])
    himp[:,:nao,:nao] = hcore_cell
    himp[:,ncore:nval,nao:] = bath_v
    himp[:,nao:,ncore:nval] = bath_v.transpose(0,2,1)
    for s in range(spin):
        himp[s,nao:,nao:] = np.diag(bath_e[s])

    eri_imp = np.zeros([spin*(spin+1)//2, nao+nbath, nao+nbath, nao+nbath, nao+nbath])
    eri_imp[:,:nao,:nao,:nao,:nao] = eri_cell
    return himp, eri_imp

def get_bath(hyb, freqs, wts):
    '''
    Convert hybridization function
    to bath couplings and energies,
    linear or gauss discretization

    Args:
        hyb : (spin, nimp, nimp, nw) ndarray
        freqs : (nw) ndarray
        wts : (nw) ndarray, wts at freq pts

    Returns:
        bath_v : (spin, nimp, nimp*nw) ndarray
        bath_e : (spin, nimp*nw) ndarray
    '''
    nw = len(freqs)
    wh = max(freqs)
    wl = min(freqs)
    spin, nimp = hyb.shape[0:2]

    dw = (wh - wl) / (nw - 1)
    # Eq. (6), arxiv:1507.07468
    v2 = -1./np.pi * np.imag(hyb)

    # simple discretization of bath, Eq. (9), arxiv:1507.07468
    v = np.empty_like(v2)

    for s in range(spin):
        for iw in range(nw):
            eig, vec = linalg.eigh(v2[s,:,:,iw])
            # although eigs should be positive, there
            # could be numerical-zero negative eigs: check this
            neg_eigs = [e for e in eig if e < 0]
            if rank == 0:
                if not np.allclose(neg_eigs, 0):
                    log = logger.Logger(sys.stdout, 4)
                    for neg_eig in neg_eigs:
                        log.warn('hyb eval = %.8f', neg_eig)
            # set negative eigs to 0
            for k in range(len(eig)):
                if eig[k] < 0:
                    eig[k] = 0.
            v[s,:,:,iw] = np.dot(vec, np.diag(np.sqrt(np.abs(eig)))) * \
                        np.sqrt(wts[iw])

    # bath_v[p,k_n] is the coupling btw impurity site p and bath orbital k
    # (total number nw=nbath) belonging to bath n (total number nimp)
    bath_v = v.reshape([spin, nimp, nimp*nw])
    bath_e = np.zeros([spin, nimp*nw])

    # bath_e is [nimp*nw] array, with entries
    # w1, w2 ... wn, w1, w2 ... wn, ...
    for s in range(spin):
        for ip in range(nimp):
            for iw in range(nw):
                bath_e[s, ip*nw + iw] = freqs[iw]

    return bath_v, bath_e

def get_bath_direct(hyb, freqs, nw_org):
    """
    Convert hybridization function
    to bath couplings and energies,
    log or direct discretization

    Args:
        hyb : (spin, nimp, nimp, nw) ndarray
        freqs : (nw) ndarray
        nw_org: integer, number of bath energies

    Returns:
        bath_v : (spin, nimp, nimp*nw_org) ndarray
        bath_e : (spin, nimp*nw_org) ndarray
    """
    nw = len(freqs)
    wmult = nw // nw_org
    wh = max(freqs)
    wl = min(freqs)

    # Eq. (6), arxiv:1507.07468
    v2 = -1./np.pi * np.imag(hyb)

    # direct discretization of bath, Eq. (7), arxiv:2003.06062 
    spin, nimp, nimp, nw = v2.shape
    J_int = np.zeros((spin, nimp,nimp,nw_org))
    for s in range(spin):
        for iw in range(nw_org):
            for j in range(wmult):
                J_int[s,:,:,iw] += (v2[s,:,:,iw*wmult+j] + v2[s,:,:,iw*wmult+j+1]) \
                                * (freqs[iw*wmult+j+1] - freqs[iw*wmult+j]) / 2

    v = np.empty_like(J_int)
    en = np.zeros((spin, nw_org))

    for s in range(spin):
        for iw in range(nw_org):
            eig, vec = linalg.eigh(J_int[s,:,:,iw])
            # although eigs should be positive, there
            # could be numerical-zero negative eigs: check this
            neg_eigs = [e for e in eig if e < 0]
            if rank == 0:
                if not np.allclose(neg_eigs, 0):
                    log = logger.Logger(sys.stdout, 4)
                    for neg_eig in neg_eigs:
                        log.warn('hyb eval = %.8f', neg_eig)
            # set negative eigs to 0
            for k in range(len(eig)):
                if eig[k] < 0:
                    eig[k] = 0.

            v[s,:,:,iw] = np.dot(vec, np.diag(np.sqrt(np.abs(eig))))
            e_sum = 0.
            for j in range(wmult):
                e_sum += (freqs[iw*wmult+j] * np.trace(v2[s,:,:,iw*wmult+j]) \
                        + freqs[iw*wmult+j+1] * np.trace(v2[s,:,:,iw*wmult+j+1])) \
                                * (freqs[iw*wmult+j+1] - freqs[iw*wmult+j]) / 2.
            en[s,iw] = e_sum / np.trace(J_int[s,:,:,iw])

    # bath_v[p,k_n] is the coupling btw impurity site p and bath orbital k
    # (total number nw_org=nbath) belonging to bath n (total number nimp)
    bath_v = v.reshape([spin, nimp, nimp*nw_org])
    bath_e = np.zeros([spin, nimp*nw_org])

    # bath_e is [nimp*nw_org] array, with entries
    # w1, w2 ... wn, w1, w2 ... wn, ...
    for s in range(spin):
        for ip in range(nimp):
            for iw in range(nw_org):
                bath_e[s, ip*nw_org + iw] = en[s,iw]

    return bath_v, bath_e


class DMFT(lib.StreamObject):
    '''
    List of DMFT class parameters (self-consistent iterations)

    max_cycle: max number of DMFT self-consistent iterations
    conv_tol: tolerance of hybridization that controls DMFT self-consistency
    damp: damping factor for first DMFT iteration
    gmres_tol: GMRES/GCROTMK convergence tolerance for imp solvers
    '''
    max_cycle = 10
    conv_tol = 1e-3
    damp = 0.7
    gmres_tol = 1e-3
    max_memory = 8000
    try:
        n_threads = int(os.environ['OMP_NUM_THREADS'])
    except:
        n_threads = 8

    # DIIS parameters for DMFT hybridization
    diis = True
    diis_space = 6
    diis_start_cycle = 1
    diis_file = None

    def __init__(self, hcore_k, JK_k, DM_k, eris, nval, ncore,
                 nbath, nb_per_e, disc_type='opt', solver_type='cc'):
        # account for both spin-restricted and spin-unrestricted cases
        if len(hcore_k.shape) == 3:
            hcore_k = hcore_k[np.newaxis, ...]
        if len(JK_k.shape) == 3:
            JK_k = JK_k[np.newaxis, ...]
        if len(DM_k.shape) == 3:
            DM_k = DM_k[np.newaxis, ...]
        if len(eris.shape) == 4:
            eris = eris[np.newaxis, ...]

        self.spin, self.nkpts, self.nao, _ = hcore_k.shape
        assert (hcore_k.shape == (self.spin, self.nkpts, self.nao, self.nao,))
        assert (JK_k.shape == (self.spin, self.nkpts, self.nao, self.nao,))
        assert (DM_k.shape == (self.spin, self.nkpts, self.nao, self.nao,))
        assert (eris.shape == (self.spin*(self.spin+1)//2, self.nao,
                               self.nao, self.nao, self.nao,))

        self.hcore_k = hcore_k
        self.JK_k = JK_k
        self.DM_k = DM_k
        self.eris = eris
        self.nbath = nbath
        self.nb_per_e = nb_per_e
        self.nval = nval
        self.ncore = ncore
        self.solver_type = solver_type
        self.disc_type = disc_type
        self.verbose = logger.NOTE
        self.chkfile = None
        self.diag_only = False
        self.orb_fit = None
        self.gw_dmft = True
        self.opt_init_method = 'direct'
        self.run_imagfreq = False
        self.twist_average = False
        self.band_interpolate = False

        self.mu = None
        self.JK_00 = None
        self.converged = False
        self.hyb = None
        self.sigma = None
        self.freqs = None
        self.wts = None

        # CAS specific parameters
        self.cas = False
        self.casno = 'gw'
        self.composite = False
        self.thresh = None
        self.thresh2 = None
        self.nvir_act = None
        self.nocc_act = None
        self.save_gf = False
        self.read_gf = False
        self.load_cas = False
        self.load_mf = False
        self.save_mf = False
        # CAS spin-unrestricted
        self.nvir_act_a = None
        self.nocc_act_a = None
        self.nvir_act_b = None
        self.nocc_act_b = None

        # DMRG specific parameters
        self.gs_n_steps = None
        self.gf_n_steps = None
        self.gs_tol = None
        self.gf_tol = None
        self.gs_bond_dims = None
        self.gs_noises = None
        self.gf_bond_dims = None
        self.gf_noises = None
        self.dmrg_gmres_tol = None
        self.dmrg_verbose = 1
        self.reorder_method = None
        self.dmrg_local = True
        self.n_off_diag_cg = 0
        self.load_dir = None
        self.save_dir = './gs_mps'
        self.extra_freqs = None
        self.extra_delta = None

    def dump_flags(self):
        if self.verbose < logger.INFO:
            return self

        if rank == 0:
            logger.info(self, '\n')
            logger.info(self, '******** %s flags ********', self.__class__)
            logger.info(self, 'impurity solver = %s', self.solver_type)
            logger.info(self, 'discretization method = %s', self.disc_type)
            logger.info(self, 'n impurity orbitals = %d', self.nao)
            logger.info(self, 'n core orbitals = %d', self.ncore)
            logger.info(self, 'n bath orbital energies = %d', self.nbath)
            logger.info(self, 'n bath orbitals per bath energy = %d', self.nb_per_e)
            logger.info(self, 'n bath orbitals total = %d', self.nbath*self.nb_per_e)
            logger.info(self, 'nkpts in lattice = %d', self.nkpts)
            if self.opt_mu:
                logger.info(self, 'mu will be optimized, init guess = %s, target occupancy = %s',
                            self.mu, self.occupancy)
            else:
                logger.info(self, 'mu is fixed, mu = %g', self.mu)
            logger.info(self, 'damping factor = %g', self.damp)
            logger.info(self, 'DMFT convergence tol = %g', self.conv_tol)
            logger.info(self, 'max. DMFT cycles = %d', self.max_cycle)
            logger.info(self, 'GMRES convergence tol = %g', self.gmres_tol)
            logger.info(self, 'delta for discretization = %g', self.delta)
            logger.info(self, 'using diis = %s', self.diis)
            if self.diis:
                logger.info(self, 'diis_space = %d', self.diis_space)
                logger.info(self, 'diis_start_cycle = %d', self.diis_start_cycle)
            if self.chkfile:
                logger.info(self, 'chkfile to save DMFT result = %s', self.chkfile)
        return self

    def dump_chk(self):
        if self.chkfile:
            with h5py.File(self.chkfile, 'w') as fh5:
                fh5['dmft/hyb'] = self.hyb
                fh5['dmft/sigma'] = self.sigma
                fh5['dmft/solver_type'] = self.solver_type
                fh5['dmft/disc_type'] = self.disc_type
                fh5['dmft/mu'] = self.mu
                fh5['dmft/delta'] = self.delta
                fh5['dmft/freqs'] = self.freqs
                fh5['dmft/wts'] = self.wts
        return self

    def get_kgw_sigma(self, freqs, eta):
        '''
        Get k-point GW-AC self-energy in LO basis
        '''
        fn = 'ac_coeff.h5'
        feri = h5py.File(fn, 'r')
        coeff = np.asarray(feri['coeff'])
        ef = np.asarray(feri['fermi'])
        omega_fit = np.asarray(feri['omega_fit'])
        feri.close()

        fn = 'C_mo_lo.h5'
        feri = h5py.File(fn, 'r')
        C_mo_lo = np.asarray(feri['C_mo_lo'])
        C_ao_lo = np.asarray(feri['C_ao_lo'])
        feri.close()

        nw = len(freqs)
        spin, nkpts, nao, nlo = C_mo_lo.shape
        sigma = np.zeros([spin,nkpts,nao,nao,nw], dtype=np.complex)
        if coeff.ndim == 4:
            coeff = coeff[np.newaxis, ...]
        for s in range(spin):
            for k in range(nkpts):
                for p in range(nao):
                    for q in range(nao):
                        sigma[s,k,p,q] = krgw_gf.pade_thiele(freqs-ef+1j*eta, omega_fit, coeff[s,k,:,p,q])

        sigma_lo = np.zeros([spin,nkpts,nlo,nlo,nw], dtype=np.complex)
        for s in range(spin):
            for iw in range(len(freqs)):
                for k in range(nkpts):
                    sigma_lo[s,k,:,:,iw] = np.dot(np.dot(C_mo_lo[s,k].T.conj(),
                                                   sigma[s,k,:,:,iw]), C_mo_lo[s,k])

        return sigma_lo

    def get_kgw_sigma_interpolate(self, freqs, eta, k_range=None):
        '''
        Get interpolated k-point GW-AC self-energy in LO basis
        NOTE: sigma = v_hf + sigma_gw - v_xc, must be used together with DFT Fock
        '''
        from fcdmft.utils import interpolate
        from pyscf.pbc import scf, dft
        from pyscf.pbc.lib import chkfile

        fn = 'ac_coeff.h5'
        feri = h5py.File(fn, 'r')
        coeff = np.asarray(feri['coeff'])
        ef = np.asarray(feri['fermi'])
        omega_fit = np.asarray(feri['omega_fit'])
        feri.close()

        fn = 'C_mo_lo.h5'
        feri = h5py.File(fn, 'r')
        C_mo_lo = np.asarray(feri['C_mo_lo'])
        C_ao_lo = np.asarray(feri['C_ao_lo'])
        feri.close()

        fn = 'vxc.h5'
        feri = h5py.File(fn, 'r')
        vk = np.array(feri['vk'])
        v_mf = np.array(feri['v_mf'])
        feri.close()

        fn = 'hcore_JK_iao_k_dft_band.h5'
        feri = h5py.File(fn, 'r')
        kpts_band = np.array(feri['kpts'])
        feri.close()

        # load cell and kmf
        cell = chkfile.load_cell('cell.chk')
        kpts = [[0,0,0]]
        kmf = dft.KRKS(cell, kpts).density_fit()
        data = chkfile.load('kmf.chk', 'scf')
        kmf.__dict__.update(data)

        nw = len(freqs)
        spin, nkpts, nao, nlo = C_mo_lo.shape
        sigma = np.zeros([spin,nkpts,nao,nao,nw], dtype=complex)
        if coeff.ndim == 4:
            coeff = coeff[np.newaxis, ...]
        if vk.ndim == 3:
            vk = vk[np.newaxis, ...]
            v_mf = v_mf[np.newaxis, ...]
        for s in range(spin):
            for k in range(nkpts):
                for p in range(nao):
                    for q in range(nao):
                        sigma[s,k,p,q] = krgw_gf.pade_thiele(freqs-ef+1j*eta, omega_fit, coeff[s,k,:,p,q])
                        sigma[s,k,p,q] += vk[s,k,p,q] - v_mf[s,k,p,q]

        if k_range is not None:
            kpts_band = kpts_band[k_range]
        nkpts_band = len(kpts_band)
        if spin==1:
            sigma_lo = interpolate.interpolate_selfenergy(kmf, kpts_band, sigma[0], C_ao_lo=C_ao_lo[0])
            sigma_lo = sigma_lo.reshape(1, *sigma_lo.shape)
        else:
            sigma_lo = np.zeros([spin,nkpts_band,nlo,nlo,nw], dtype=complex)
            for s in range(spin):
                sigma_lo[s] = interpolate.interpolate_selfenergy(kmf, kpts_band, sigma[s], C_ao_lo=C_ao_lo[s])

        return sigma_lo

    def get_khf_sigma_interpolate(self, freqs, eta, k_range=None):
        '''
        Get interpolated k-point HF self-energy in LO basis
        NOTE: sigma = v_hf - v_xc, must be used together with DFT Fock
        '''
        from fcdmft.utils import interpolate
        from pyscf.pbc import scf, dft
        from pyscf.pbc.lib import chkfile

        fn = 'C_mo_lo.h5'
        feri = h5py.File(fn, 'r')
        C_mo_lo = np.asarray(feri['C_mo_lo'])
        C_ao_lo = np.asarray(feri['C_ao_lo'])
        feri.close()

        fn = 'vxc.h5'
        feri = h5py.File(fn, 'r')
        vk = np.array(feri['vk'])
        v_mf = np.array(feri['v_mf'])
        feri.close()

        fn = 'hcore_JK_iao_k_dft_band.h5'
        feri = h5py.File(fn, 'r')
        kpts_band = np.array(feri['kpts'])
        feri.close()

        # load cell and kmf
        cell = chkfile.load_cell('cell.chk')
        kpts = [[0,0,0]]
        kmf = dft.KRKS(cell, kpts).density_fit()
        data = chkfile.load('kmf.chk', 'scf')
        kmf.__dict__.update(data)

        nw = len(freqs)
        spin, nkpts, nao, nlo = C_mo_lo.shape
        sigma = np.zeros([spin,nkpts,nao,nao,1], dtype=complex)
        if vk.ndim == 3:
            vk = vk[np.newaxis, ...]
            v_mf = v_mf[np.newaxis, ...]
        sigma[:,:,:,:,0] = vk - v_mf

        if k_range is not None:
            kpts_band = kpts_band[k_range]
        nkpts_band = len(kpts_band)
        if spin==1:
            sigma_lo = interpolate.interpolate_selfenergy(kmf, kpts_band, sigma[0], C_ao_lo=C_ao_lo[0])
            sigma_lo = sigma_lo.reshape(1, *sigma_lo.shape)
        else:
            sigma_lo = np.zeros([spin,nkpts_band,nlo,nlo,1], dtype=complex)
            for s in range(spin):
                sigma_lo[s] = interpolate.interpolate_selfenergy(kmf, kpts_band, sigma[s], C_ao_lo=C_ao_lo[s])

        sigma_lo_w = np.zeros([spin,nkpts_band,nlo,nlo,nw], dtype=complex)
        for iw in range(nw):
            sigma_lo_w[:,:,:,:,iw] = sigma_lo[:,:,:,:,0]

        return sigma_lo_w

    def get_kgw_sigma_TA(self, freqs, eta):
        '''
        Get twist average k-point GW-AC self-energy in LO basis
        '''
        center_list = self.center_list
        sigma_TA = []
        for i in range(len(center_list)):
            center = center_list[i]

            fn = 'ac_coeff_%d_%d_%d.h5'%(center[0],center[1],center[2])
            feri = h5py.File(fn, 'r')
            coeff = np.asarray(feri['coeff'])
            ef = np.asarray(feri['fermi'])
            omega_fit = np.asarray(feri['omega_fit'])
            feri.close()

            fn = 'C_mo_lo_%d_%d_%d.h5'%(center[0],center[1],center[2])
            feri = h5py.File(fn, 'r')
            C_mo_lo = np.asarray(feri['C_mo_lo'])
            feri.close()

            nw = len(freqs)
            spin, nkpts, nao, nlo = C_mo_lo.shape
            sigma = np.zeros([spin,nkpts,nao,nao,nw], dtype=np.complex)
            if coeff.ndim == 4:
                coeff = coeff[np.newaxis, ...]
            for s in range(spin):
                for k in range(nkpts):
                    for p in range(nao):
                        for q in range(nao):
                            sigma[s,k,p,q] = krgw_gf.pade_thiele(freqs-ef+1j*eta, omega_fit, coeff[s,k,:,p,q])
 
            sigma_lo = np.zeros([spin,nkpts,nlo,nlo,nw], dtype=np.complex)
            for s in range(spin):
                for iw in range(len(freqs)):
                    for k in range(nkpts):
                        sigma_lo[s,k,:,:,iw] = np.dot(np.dot(C_mo_lo[s,k].T.conj(),
                                                       sigma[s,k,:,:,iw]), C_mo_lo[s,k])
            sigma_TA.append(sigma_lo)

        sigma_TA = np.array(sigma_TA)
        sigma_TA = sigma_TA.transpose(1,0,2,3,4,5).reshape(spin, len(center_list)*nkpts, nlo, nlo, nw)

        return sigma_TA

    def get_gw_sigma(self, freqs, eta):
        '''
        Get local GW double counting self-energy
        '''
        spin, nao, nbath = self.spin, self.nao, self.nbath
        nw = len(freqs)

        fn = 'imp_ac_coeff.h5'
        feri = h5py.File(fn, 'r')
        coeff = np.asarray(feri['coeff'])
        ef = np.asarray(feri['fermi'])
        omega_fit = np.asarray(feri['omega_fit'])
        feri.close()

        sigma = np.zeros([spin,nao,nao,nw], dtype=np.complex)
        if coeff.ndim == 3:
            coeff = coeff[np.newaxis, ...]
        for s in range(spin):
            for p in range(nao):
                for q in range(nao):
                    sigma[s,p,q] = gw_dc.pade_thiele(freqs-ef+1j*eta, omega_fit, coeff[s,:,p,q])

        return sigma

    def kernel(self, mu0, wl=None, wh=None, occupancy=None, delta=0.1,
               conv_tol=None, opt_mu=False, dump_chk=True):
        '''
        main routine for DMFT

        Args:
            mu0 : float
                Chemical potential or an initial guess if opt_mu=True

        Kwargs:
            wl, wh : None or float
                Hybridization discretization range
            occupancy : None or float
                Target average occupancy (1 is half filling)
            delta : float
                Broadening used during self-consistency
            conv_tol : float
                Convergence tolerance on the hybridization
            opt_mu : bool
                Whether to optimize the chemical potential
            dump_chk : bool
                Whether to dump DMFT chkfile
        '''

        cput0 = (time.process_time(), time.perf_counter())
        self.mu = mu0
        self.occupancy = occupancy
        self.delta = delta
        if conv_tol:
            self.conv_tol = conv_tol
        self.opt_mu = opt_mu
        if opt_mu:
            assert(self.occupancy is not None)

        self.dump_flags()

        self.converged, self.mu = kernel(self, mu0, wl=wl, wh=wh, occupancy=occupancy, delta=delta,
                                         conv_tol=conv_tol, opt_mu=opt_mu, dump_chk=dump_chk)

        if rank == 0:
            self._finalize()
            logger.timer(self, 'DMFT', *cput0)

    def dmft(self, **kwargs):
        return self.kernel(**kwargs)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if self.converged:
            logger.info(self, '%s converged', self.__class__.__name__)
        else:
            logger.note(self, '%s not converged', self.__class__.__name__)
        return self

    def run_diis(self, hyb, istep, adiis):
        if (adiis and istep >= self.diis_start_cycle):
            hyb = adiis.update(hyb)
            logger.debug1(self, 'DIIS for step %d', istep)
        return hyb

    def get_rdm_imp(self):
        '''Calculate the interacting local RDM from the impurity problem'''
        if self.solver_type == 'cc':
            return cc_rdm(self._scf, ao_orbs=range(self.nao), cas=self.cas, casno=self.casno,
                          composite=self.composite, thresh=self.thresh, nvir_act=self.nvir_act,
                          nocc_act=self.nocc_act, load_cas=self.load_cas)
        elif self.solver_type == 'ucc':
            return ucc_rdm(self._scf, ao_orbs=range(self.nao), cas=self.cas, casno=self.casno,
                          composite=self.composite, thresh=self.thresh, nvir_act_a=self.nvir_act_a,
                          nocc_act_a=self.nocc_act_a, nvir_act_b=self.nvir_act_b,
                          nocc_act_b=self.nocc_act_b)
        elif self.solver_type == 'fci':
            return fci_rdm(self._scf, ao_orbs=range(self.nao))
        elif self.solver_type == 'dmrg':
            return dmrg_rdm(self._scf, ao_orbs=range(self.nao), n_threads=self.n_threads,
                            cas=self.cas, casno=self.casno, composite=self.composite, thresh=self.thresh,
                            nvir_act=self.nvir_act, nocc_act=self.nocc_act,
                            reorder_method=self.reorder_method, gs_n_steps=self.gs_n_steps,
                            gs_tol=self.gs_tol, gs_bond_dims=self.gs_bond_dims, gs_noises=self.gs_noises,
                            local=self.dmrg_local, load_dir=self.load_dir, save_dir=self.save_dir,
                            dyn_corr_method=self.dyn_corr_method, ncore=self.nocc_act_low,
                            nvirt=self.nvir_act_high, load_cas=self.load_cas)
        elif self.solver_type == 'dmrgsz':
            return udmrg_rdm(self._scf, ao_orbs=range(self.nao), n_threads=self.n_threads,
                            cas=self.cas, casno=self.casno, composite=self.composite, thresh=self.thresh,
                            nvir_act_a=self.nvir_act_a, nocc_act_a=self.nocc_act_a,
                            nvir_act_b=self.nvir_act_b, nocc_act_b=self.nocc_act_b,
                            reorder_method=self.reorder_method, gs_n_steps=self.gs_n_steps,
                            gs_tol=self.gs_tol, gs_bond_dims=self.gs_bond_dims, gs_noises=self.gs_noises,
                            local=self.dmrg_local, load_dir=self.load_dir, save_dir=self.save_dir,
                            load_cas=self.load_cas)

    def get_gf_imp(self, freqs, delta, extra_freqs=None, extra_delta=None):
        '''Calculate the interacting local GF from the impurity problem'''
        if self.solver_type == 'cc':
            return cc_gf(self._scf, freqs, delta, ao_orbs=range(self.nao), gmres_tol=self.gmres_tol,
                         cas=self.cas, casno=self.casno, composite=self.composite, thresh=self.thresh,
                         nvir_act=self.nvir_act, nocc_act=self.nocc_act, load_cas=self.load_cas)
        elif self.solver_type == 'ucc':
            return ucc_gf(self._scf, freqs, delta, ao_orbs=range(self.nao), gmres_tol=self.gmres_tol,
                         cas=self.cas, casno=self.casno, composite=self.composite, thresh=self.thresh,
                         nvir_act_a=self.nvir_act_a, nocc_act_a=self.nocc_act_a, nvir_act_b=self.nvir_act_b,
                         nocc_act_b=self.nocc_act_b)
        elif self.solver_type == 'fci':
            # TODO: CASCI
            return fci_gf(self._scf, freqs, delta, ao_orbs=range(self.nao),
                          gmres_tol=self.gmres_tol)
        elif self.solver_type == 'dmrg':
            return dmrg_gf(self._scf, freqs, delta, ao_orbs=range(self.nao), n_threads=self.n_threads,
                    cas=self.cas, casno=self.casno, composite=self.composite,
                    thresh=self.thresh, nvir_act=self.nvir_act, nocc_act=self.nocc_act,
                    reorder_method=self.reorder_method, cc_gmres_tol=self.gmres_tol, gf_n_steps=self.gf_n_steps,
                    gs_n_steps=self.gs_n_steps, gs_tol=self.gs_tol, dmrg_verbose=self.dmrg_verbose,
                    gs_bond_dims=self.gs_bond_dims, gf_bond_dims=self.gf_bond_dims, gf_tol=self.gf_tol,
                    gmres_tol=self.dmrg_gmres_tol, gs_noises=self.gs_noises, gf_noises=self.gf_noises,
                    n_off_diag_cg=self.n_off_diag_cg, local=self.dmrg_local, extra_freqs=extra_freqs,
                    extra_delta=extra_delta, load_cas=self.load_cas, thresh2=self.thresh2,
                    dyn_corr_method=self.dyn_corr_method, ncore=self.nocc_act_low, nvirt=self.nvir_act_high)
        elif self.solver_type == 'dmrgsz':
            return udmrg_gf(self._scf, freqs, delta, ao_orbs=range(self.nao), n_threads=self.n_threads,
                    cas=self.cas, casno=self.casno, composite=self.composite,
                    thresh=self.thresh, nvir_act_a=self.nvir_act_a, nocc_act_a=self.nocc_act_a,
                    nvir_act_b=self.nvir_act_b, nocc_act_b=self.nocc_act_b,
                    reorder_method=self.reorder_method, cc_gmres_tol=self.gmres_tol, gf_n_steps=self.gf_n_steps,
                    gs_n_steps=self.gs_n_steps, gs_tol=self.gs_tol, dmrg_verbose=self.dmrg_verbose,
                    gs_bond_dims=self.gs_bond_dims, gf_bond_dims=self.gf_bond_dims, gf_tol=self.gf_tol,
                    gmres_tol=self.dmrg_gmres_tol, gs_noises=self.gs_noises, gf_noises=self.gf_noises,
                    n_off_diag_cg=self.n_off_diag_cg, local=self.dmrg_local, extra_freqs=extra_freqs,
                    extra_delta=extra_delta, load_cas=self.load_cas, thresh2=self.thresh2)

    def get_gf0_imp(self, freqs, delta):
        '''Calculate the noninteracting local GF from the impurity problem'''
        himp = self._scf.get_hcore()
        if len(himp.shape) == 2:
            himp = himp[np.newaxis, ...]
        spin, nb = himp.shape[0:2]
        nw = len(freqs)
        sig_dum = np.zeros((spin,nb,nb,nw,))
        gf = get_gf(himp, sig_dum, freqs, delta)
        return gf

    def get_sigma_imp(self, freqs, delta, load_dir=None, save_dir=None, save_gf=False,
                      read_gf=False, extra_freqs=None, extra_delta=None):
        '''Calculate the local self-energy from the impurity problem'''
        spin = self.spin
        if spin == 1:
            nmo = len(self._scf.mo_energy)
        else:
            nmo = len(self._scf.mo_energy[0])
        nao = self.nao
        if extra_delta is not None:
            freqs_comp = np.array(extra_freqs).reshape(-1)
            gf0 = self.get_gf0_imp(freqs_comp, extra_delta)
        else:
            freqs_comp = freqs
            gf0 = self.get_gf0_imp(freqs, delta)

        if self.solver_type == 'cc':
            gf = cc_gf(self._scf, freqs, delta, ao_orbs=range(nmo), gmres_tol=self.gmres_tol,
                       nimp=self.nao, cas=self.cas, casno=self.casno, composite=self.composite,
                       thresh=self.thresh, nvir_act=self.nvir_act, nocc_act=self.nocc_act,
                       save_gf=save_gf, read_gf=read_gf, load_cas=self.load_cas)
        elif self.solver_type == 'ucc':
            gf = ucc_gf(self._scf, freqs, delta, ao_orbs=range(nmo), gmres_tol=self.gmres_tol,
                       nimp=self.nao, cas=self.cas, casno=self.casno, composite=self.composite,
                       thresh=self.thresh, nvir_act_a=self.nvir_act_a, nocc_act_a=self.nocc_act_a,
                       nvir_act_b=self.nvir_act_b, nocc_act_b=self.nocc_act_b,
                       save_gf=save_gf, read_gf=read_gf)
        elif self.solver_type == 'fci':
            # TODO: CASCI
            gf = self.get_gf_imp(freqs, delta)
        elif self.solver_type == 'dmrg':
            gf = dmrg_gf(self._scf, freqs, delta, ao_orbs=range(nmo), n_threads=self.n_threads,
                    nimp=self.nao, cas=self.cas, casno=self.casno, composite=self.composite,
                    thresh=self.thresh, nvir_act=self.nvir_act, nocc_act=self.nocc_act,
                    reorder_method=self.reorder_method, cc_gmres_tol=self.gmres_tol, gf_n_steps=self.gf_n_steps,
                    gs_n_steps=self.gs_n_steps, gs_tol=self.gs_tol, dmrg_verbose=self.dmrg_verbose,
                    gs_bond_dims=self.gs_bond_dims, gf_bond_dims=self.gf_bond_dims, gf_tol=self.gf_tol,
                    gmres_tol=self.dmrg_gmres_tol, gs_noises=self.gs_noises, gf_noises=self.gf_noises,
                    save_gf=save_gf, read_gf=read_gf, load_dir=load_dir, save_dir=save_dir,
                    n_off_diag_cg=self.n_off_diag_cg, local=self.dmrg_local, extra_freqs=extra_freqs,
                    extra_delta=extra_delta, load_cas=self.load_cas, thresh2=self.thresh2,
                    dyn_corr_method=self.dyn_corr_method, ncore=self.nocc_act_low, nvirt=self.nvir_act_high)
        elif self.solver_type == 'dmrgsz':
            gf = udmrg_gf(self._scf, freqs, delta, ao_orbs=range(nmo), n_threads=self.n_threads,
                    nimp=self.nao, cas=self.cas, casno=self.casno, composite=self.composite,
                    thresh=self.thresh, nvir_act_a=self.nvir_act_a, nocc_act_a=self.nocc_act_a,
                    nvir_act_b=self.nvir_act_b, nocc_act_b=self.nocc_act_b,
                    reorder_method=self.reorder_method, cc_gmres_tol=self.gmres_tol, gf_n_steps=self.gf_n_steps,
                    gs_n_steps=self.gs_n_steps, gs_tol=self.gs_tol, dmrg_verbose=self.dmrg_verbose,
                    gs_bond_dims=self.gs_bond_dims, gf_bond_dims=self.gf_bond_dims, gf_tol=self.gf_tol,
                    gmres_tol=self.dmrg_gmres_tol, gs_noises=self.gs_noises, gf_noises=self.gf_noises,
                    save_gf=save_gf, read_gf=read_gf, load_dir=load_dir, save_dir=save_dir,
                    n_off_diag_cg=self.n_off_diag_cg, local=self.dmrg_local, extra_freqs=extra_freqs,
                    extra_delta=extra_delta, load_cas=self.load_cas, thresh2=self.thresh2)
 
        if self.solver_type == 'cc' or self.solver_type == 'ucc':
            gf = 0.5 * (gf+gf.transpose(0,2,1,3))

        tmpdir = 'dmft_dos'
        if rank == 0:
            if not os.path.isdir(tmpdir):
                os.mkdir(tmpdir)
            if isinstance(freqs_comp[-1], float):
                write.write_gf_to_dos(tmpdir+'/dmft_imp_dos', freqs_comp, gf)

        return get_sigma(gf0, gf)

    def get_ldos_imp(self, freqs, delta):
        '''Calculate the local DOS from the impurity problem'''
        nao = self.nao
        gf = self.get_gf_imp(freqs, delta, extra_freqs=self.extra_freqs, extra_delta=self.extra_delta)
        ldos = -1./np.pi*np.trace(gf[:,:nao,:nao,:].imag,axis1=1,axis2=2)
        return ldos

    def get_ldos_latt(self, freqs, delta, sigma=None):
        '''Calculate local DOS from the lattice problem'''
        if self.extra_delta is not None:
            freqs_comp = np.array(self.extra_freqs).reshape(-1)
            delta_comp = self.extra_delta
        else:
            freqs_comp = freqs
            delta_comp = delta
        nw = len(freqs_comp)
        nao = self.nao
        nkpts = self.nkpts
        spin = self.spin
        nval = self.nval

        if sigma is None:
            sigma = self.get_sigma_imp(freqs, delta, save_gf=self.save_gf, read_gf=self.read_gf,
                                       load_dir=self.load_dir, save_dir=self.save_dir,
                                       extra_freqs=self.extra_freqs, extra_delta=self.extra_delta)
        nb = self.nbath
        sigma = sigma[:,:nao,:nao,:]
        JK_00 = self.JK_00

        if self.gw_dmft:
            # Compute impurity GW self-energy (DC term)
            sigma_gw_imp = self.get_gw_sigma(freqs_comp, delta_comp)

            # Compute k-point GW self-energy at given freqs and delta
            sigma_kgw = self.get_kgw_sigma(freqs_comp, delta_comp)
            if self.twist_average:
                sigma_kgw_band = self.get_kgw_sigma_TA(freqs_comp, delta_comp)
        else:
            sigma_gw_imp = np.zeros((spin, nao, nao, nw), dtype=np.complex)
            sigma_kgw = np.zeros((spin, nkpts, nao, nao, nw), dtype=np.complex)
            if self.twist_average:
                nkpts_band = self.hcore_k_band.shape[1]
                sigma_kgw_band = np.zeros((spin, nkpts_band, nao, nao, nw), dtype=np.complex)

        # remove GW double counting
        for w in range(nw):
            sigma[:,:,:,w] = sigma[:,:,:,w] - JK_00
        sigma = sigma - sigma_gw_imp

        tmpdir = 'dmft_dos'
        if rank == 0:
            write.write_sigma(tmpdir+'/dmft_sigma_imp_prod', freqs_comp, sigma)
            fn = 'sigma_nb-%d_eta-%0.2f_w-%.3f-%.3f.h5'%(nb, delta_comp*27.211386,
                                                     freqs_comp[0]*27.211386, freqs_comp[-1]*27.211386)
            feri = h5py.File(fn, 'w')
            feri['omegas'] = np.asarray(freqs_comp)
            feri['sigma'] = np.asarray(sigma)
            feri.close()
        comm.Barrier()

        # k-point GW GF
        gf_loc_gw = np.zeros([spin, nao, nao, nw], np.complex128)
        for k in range(nkpts):
            gf = get_gf(self.hcore_k[:,k]+self.JK_k[:,k], sigma_kgw[:,k], freqs_comp, delta_comp)
            gf_loc_gw += 1./nkpts * gf
            if rank == 0:
                if self.gw_dmft:
                    write.write_gf_to_dos(tmpdir+'/gw_dos_k-%d'%(k), freqs_comp, gf)
                else:
                    write.write_gf_to_dos(tmpdir+'/hf_dos_k-%d'%(k), freqs_comp, gf)
            comm.Barrier()

        if self.twist_average:
            nkpts_band = self.hcore_k_band.shape[1]
            gf_loc_gw = nkpts * gf_loc_gw
            for k in range(nkpts_band):
                gf = get_gf(self.hcore_k_band[:,k]+self.JK_k_band[:,k], sigma_kgw_band[:,k], freqs_comp, delta_comp)
                gf_loc_gw += gf
                if rank == 0:
                    if self.gw_dmft:
                        write.write_gf_to_dos(tmpdir+'/gw_band_dos_k-%d'%(k), freqs_comp, gf)
                    else:
                        write.write_gf_to_dos(tmpdir+'/hf_band_dos_k-%d'%(k), freqs_comp, gf)
                comm.Barrier()
            gf_loc_gw = 1./(nkpts+nkpts_band) * gf_loc_gw

        # DMFT GF
        gf_loc = np.zeros([spin, nao, nao, nw], np.complex128)
        for k in range(nkpts):
            gf = get_gf(self.hcore_k[:,k]+self.JK_k[:,k], sigma+sigma_kgw[:,k], freqs_comp, delta_comp)
            gf_loc += 1./nkpts * gf
            if rank == 0:
                write.write_gf_to_dos(tmpdir+'/dmft_dos_prod_k-%d'%(k), freqs_comp, gf)
            comm.Barrier()

        if self.twist_average:
            nkpts_band = self.hcore_k_band.shape[1]
            gf_loc = nkpts * gf_loc
            for k in range(nkpts_band):
                gf = get_gf(self.hcore_k_band[:,k]+self.JK_k_band[:,k], sigma+sigma_kgw_band[:,k], freqs_comp, delta_comp)
                gf_loc += gf
                if rank == 0:
                    write.write_gf_to_dos(tmpdir+'/dmft_band_dos_prod_k-%d'%(k), freqs_comp, gf)
                comm.Barrier()
            gf_loc = 1./(nkpts+nkpts_band) * gf_loc

        if self.band_interpolate:
            nkpts_band = self.hcore_k_band.shape[1]
            gf_loc_gw = np.zeros([spin, nao, nao, nw], np.complex128)
            gf_loc = np.zeros([spin, nao, nao, nw], np.complex128)
            mem_now = lib.current_memory()[0]
            mem_band = spin * nao**2 * nw * nkpts_band * 16/1e6
            if mem_now+mem_band < 0.5 * self.max_memory:
                if self.gw_dmft:    
                    sigma_kgw_diff_band = self.get_kgw_sigma_interpolate(freqs_comp, delta_comp)
                else:
                    sigma_kgw_diff_band = self.get_khf_sigma_interpolate(freqs_comp, delta_comp)
                for k in range(nkpts_band):
                    gf_gw = get_gf(self.hcore_k_band[:,k]+self.JK_k_dft_band[:,k], sigma_kgw_diff_band[:,k], freqs_comp, delta_comp)
                    gf_loc_gw += 1./nkpts_band * gf_gw
                    gf = get_gf(self.hcore_k_band[:,k]+self.JK_k_dft_band[:,k], sigma+sigma_kgw_diff_band[:,k], freqs_comp, delta_comp)
                    gf_loc += 1./nkpts_band * gf
                    if rank == 0:
                        if self.gw_dmft:
                            write.write_gf_to_dos(tmpdir+'/gw_band_dos_k-%d'%(k), freqs_comp, gf_gw)
                        else:
                            write.write_gf_to_dos(tmpdir+'/hf_band_dos_k-%d'%(k), freqs_comp, gf_gw)
                        write.write_gf_to_dos(tmpdir+'/dmft_band_dos_k-%d'%(k), freqs_comp, gf)
                    comm.Barrier()
            else:
                nslice = int(mem_band // (0.5*(self.max_memory - mem_now))) + 1
                nkpts_slice = (nkpts_band + nslice - 1) // nslice
                for i in range(nslice):
                    k_range = np.arange(i*nkpts_slice, min((i+1)*nkpts_slice, nkpts_band))
                    if self.gw_dmft:
                        sigma_kgw_diff_band = self.get_kgw_sigma_interpolate(freqs_comp, delta_comp, k_range)
                    else:
                        sigma_kgw_diff_band = self.get_khf_sigma_interpolate(freqs_comp, delta_comp, k_range)
                    for k in k_range:
                        gf_gw = get_gf(self.hcore_k_band[:,k]+self.JK_k_dft_band[:,k], \
                                sigma_kgw_diff_band[:,k-k_range[0]], freqs_comp, delta_comp)
                        gf_loc_gw += 1./nkpts_band * gf_gw
                        gf = get_gf(self.hcore_k_band[:,k]+self.JK_k_dft_band[:,k], \
                                sigma+sigma_kgw_diff_band[:,k-k_range[0]], freqs_comp, delta_comp)
                        gf_loc += 1./nkpts_band * gf
                        if rank == 0:
                            if self.gw_dmft:
                                write.write_gf_to_dos(tmpdir+'/gw_band_dos_k-%d'%(k), freqs_comp, gf_gw)
                            else:
                                write.write_gf_to_dos(tmpdir+'/hf_band_dos_k-%d'%(k), freqs_comp, gf_gw)
                            write.write_gf_to_dos(tmpdir+'/dmft_band_dos_k-%d'%(k), freqs_comp, gf)
                        comm.Barrier()
            comm.Barrier()
            sigma_kgw_diff_band = None

        ldos_gw = -1./np.pi * np.trace(gf_loc_gw.imag,axis1=1,axis2=2)
        if rank == 0:
            for i in range(nao):
                ldos_orb = -1./np.pi * gf_loc_gw[:,i,i,:].imag
                if self.gw_dmft:
                    write.write_dos(tmpdir+'/gw_dos_orb-%d'%(i), freqs_comp, ldos_orb)
                else:
                    write.write_dos(tmpdir+'/hf_dos_orb-%d'%(i), freqs_comp, ldos_orb)
        comm.Barrier()

        ldos = -1./np.pi * np.trace(gf_loc.imag,axis1=1,axis2=2)
        if rank == 0:
            for i in range(nao):
                ldos_orb = -1./np.pi * gf_loc[:,i,i,:].imag
                write.write_dos(tmpdir+'/dmft_dos_prod_orb-%d'%(i), freqs_comp, ldos_orb)
        comm.Barrier()

        return ldos, ldos_gw

