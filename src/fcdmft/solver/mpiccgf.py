import time
import sys

import numpy as np
import scipy
from fcdmft.solver import gmres

import pyscf
import pyscf.cc
from pyscf.lib import logger
from pyscf.cc.eom_rccsd import amplitudes_to_vector_ip, amplitudes_to_vector_ea

from mpi4py import MPI


'''
MPI CCSD Green's function
'''

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

def greens_b_singles_ea_rhf(t1, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        return -t1[p,:]
    else:
        p = p-nocc
        result = np.zeros((nvir,), dtype=ds_type)
        result[p] = 1.0
        return result


def greens_b_doubles_ea_rhf(t2, p):
    nocc, _, nvir, _ = t2.shape
    ds_type = t2.dtype
    if p < nocc:
        return -t2[p,:,:,:]
    else:
        return np.zeros((nocc,nvir,nvir), dtype=ds_type)


def greens_b_vector_ea_rhf(cc, p):
    return amplitudes_to_vector_ea(
        greens_b_singles_ea_rhf(cc.t1, p),
        greens_b_doubles_ea_rhf(cc.t2, p),
    )


def greens_e_singles_ea_rhf(t1, t2, l1, l2, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        return l1[p, :]
    else:
        p = p-nocc
        result = np.zeros((nvir,), dtype=ds_type)
        result[p] = -1.0
        result += np.einsum('ia,i->a', l1, t1[:,p])
        result += 2*np.einsum('klca,klc->a', l2, t2[:,:,:,p])
        result -= np.einsum('klca,lkc->a', l2, t2[:,:,:,p])
        return result


def greens_e_doubles_ea_rhf(t1, l1, l2, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        return 2*l2[p,:,:,:] - l2[:,p,:,:]
    else:
        p = p-nocc
        result = np.zeros((nocc,nvir,nvir), dtype=ds_type)
        result[:,p,:] += -2*l1
        result[:,:,p] += l1
        result += 2*np.einsum('k,jkba->jab', t1[:,p], l2)
        result -= np.einsum('k,jkab->jab', t1[:,p], l2)
        return result


def greens_e_vector_ea_rhf(cc, p):
    return amplitudes_to_vector_ea(
        greens_e_singles_ea_rhf(cc.t1, cc.t2, cc.l1, cc.l2, p),
        greens_e_doubles_ea_rhf(cc.t1, cc.l1, cc.l2, p),
    )


def greens_b_singles_ip_rhf(t1, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        result = np.zeros((nocc,), dtype=ds_type)
        result[p] = 1.0
        return result
    else:
        p = p-nocc
        return t1[:,p]


def greens_b_doubles_ip_rhf(t2, p):
    nocc, _, nvir, _ = t2.shape
    ds_type = t2.dtype
    if p < nocc:
        return np.zeros((nocc,nocc,nvir), dtype=ds_type)
    else:
        p = p-nocc
        return t2[:,:,p,:]


def greens_b_vector_ip_rhf(cc, p):
    return amplitudes_to_vector_ip(
        greens_b_singles_ip_rhf(cc.t1, p),
        greens_b_doubles_ip_rhf(cc.t2, p),
    )


def greens_e_singles_ip_rhf(t1, t2, l1, l2, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        result = np.zeros((nocc,), dtype=ds_type)
        result[p] = -1.0
        result += np.einsum('ia,a->i', l1, t1[p,:])
        result += 2*np.einsum('ilcd,lcd->i', l2, t2[p,:,:,:])
        result -= np.einsum('ilcd,ldc->i', l2, t2[p,:,:,:])
        return result
    else:
        p = p-nocc
        return -l1[:,p]


def greens_e_doubles_ip_rhf(t1, l1, l2, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        result = np.zeros((nocc, nocc, nvir), dtype=ds_type)
        result[p, :, :] += -2*l1
        result[:, p, :] += l1
        result += 2*np.einsum('c,ijcb->ijb', t1[p,:], l2)
        result -= np.einsum('c,jicb->ijb', t1[p,:], l2)
        return result
    else:
        p = p-nocc
        return -2*l2[:,:,p,:] + l2[:,:,:,p]


def greens_e_vector_ip_rhf(cc, p):
    return amplitudes_to_vector_ip(
        greens_e_singles_ip_rhf(cc.t1, cc.t2, cc.l1, cc.l2, p),
        greens_e_doubles_ip_rhf(cc.t1, cc.l1, cc.l2, p),
    )


def greens_func_multiply(ham, vector, linear_part, **kwargs):
    return np.array(ham(vector, **kwargs) + linear_part * vector)


def ip_shape(cc):
    nocc, nvir = cc.t1.shape
    return nocc + nocc*nocc*nvir


def ea_shape(cc):
    nocc, nvir = cc.t1.shape
    return nvir + nocc*nvir*nvir


class CCGF(object):
    def __init__(self, mycc, tol=1e-4, verbose=None):
        self._cc = mycc
        self.tol = tol
        if verbose:
            self.verbose = verbose
        else:
            self.verbose = self._cc.verbose
        self.stdout = sys.stdout

    def ipccsd_ao(self, ps, omega_list, mo_coeff, broadening):
        '''
        Compute IP-CCSD-GF in AO basis (parallelize over orbitals)
        '''
        eomip = pyscf.cc.eom_rccsd.EOMIP(self._cc)
        eomip_imds = eomip.make_imds()
        diag = eomip.get_diag()

        nmo = mo_coeff.shape[1]
        e_vector_mo = np.zeros([nmo, ip_shape(self._cc)], dtype=np.complex128)
        b_vector_mo = np.zeros([ip_shape(self._cc), nmo], dtype=np.complex128)
        for i in range(nmo):
            e_vector_mo[i,:] = greens_e_vector_ip_rhf(self._cc, i)
        e_vector_ao = np.einsum("pi,ix->px", mo_coeff[ps,:], e_vector_mo)
        for i in range(nmo):
            b_vector_mo[:,i] = greens_b_vector_ip_rhf(self._cc, i)
        b_vector_ao = np.einsum("xi,ip->xp", b_vector_mo, mo_coeff.T[:,ps])
        comm.Barrier()

        segsize = len(ps) // size
        if rank >= size-(len(ps)-segsize*size):
            start = rank * segsize + rank-(size-(len(ps)-segsize*size))
            stop = min(len(ps), start+segsize+1)
        else:
            start = rank * segsize
            stop = min(len(ps), start+segsize)

        gf_ao = np.zeros((stop-start, len(ps), len(omega_list)), dtype=np.complex128)
        for ip in range(start,stop):
            p = ps[ip]
            x0 = None
            for iomega in range(len(omega_list))[::-1]:
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomip.matvec, vector, curr_omega - 1j * broadening, imds=eomip_imds)

                diag_w = diag + curr_omega-1j*broadening
                if x0 is None:
                    x0 = b_vector_ao[:,ip]/diag_w
                solver = gmres.GMRES(matr_multiply, b_vector_ao[:,ip], x0, diag_w, tol=self.tol)
                cput1 = (time.process_time(), time.perf_counter())
                sol = solver.solve().reshape(-1)
                cput1 = logger.timer(self, 'IPGF orbital p = %d/%d, freq w = %d/%d (%d iterations) @ Rank %d'%(
                    ip+1,len(ps),iomega+1,len(omega_list),solver.niter,rank), *cput1)
                x0 = sol
                for iq, q in enumerate(ps):
                    gf_ao[ip-start,iq,iomega] = -np.dot(e_vector_ao[iq,:], sol)
        comm.Barrier()
        gf_ao_gather = comm.gather(gf_ao)
        if rank == 0:
            gf_ao_gather = np.vstack(gf_ao_gather)
        comm.Barrier()
        gf_ao_gather = comm.bcast(gf_ao_gather,root=0)
        return gf_ao_gather

    def eaccsd_ao(self, ps, omega_list, mo_coeff, broadening):
        '''
        Compute EA-CCSD-GF in AO basis (parallelize over orbitals)
        '''
        eomea = pyscf.cc.eom_rccsd.EOMEA(self._cc)
        eomea_imds = eomea.make_imds()
        diag = eomea.get_diag()

        nmo = mo_coeff.shape[1]
        e_vector_mo = np.zeros([nmo, ea_shape(self._cc)], dtype=np.complex128)
        b_vector_mo = np.zeros([ea_shape(self._cc), nmo], dtype=np.complex128)
        for i in range(nmo):
            e_vector_mo[i,:] = greens_e_vector_ea_rhf(self._cc, i)
        e_vector_ao = np.einsum("pi,ix->px", mo_coeff[ps,:], e_vector_mo)
        for i in range(nmo):
            b_vector_mo[:,i] = greens_b_vector_ea_rhf(self._cc, i)
        b_vector_ao = np.einsum("xi,ip->xp", b_vector_mo, mo_coeff.T[:,ps])
        comm.Barrier()

        segsize = len(ps) // size
        if rank < len(ps)-segsize*size:
            start = rank * segsize + rank
            stop = min(len(ps), start+segsize+1)
        else:
            start = rank * segsize + len(ps)-segsize*size
            stop = min(len(ps), start+segsize)

        gf_ao = np.zeros((len(ps), stop-start, len(omega_list)), dtype=np.complex128)
        for iq in range(start,stop):
            q = ps[iq]
            x0 = None
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomea.matvec, vector, -curr_omega - 1j * broadening, imds=eomea_imds)

                diag_w = diag + (-curr_omega-1j*broadening)
                if x0 is None:
                    x0 = b_vector_ao[:,iq]/diag_w
                solver = gmres.GMRES(matr_multiply, b_vector_ao[:,iq], x0, diag_w, tol=self.tol)
                cput1 = (time.process_time(), time.perf_counter())
                sol = solver.solve().reshape(-1)
                cput1 = logger.timer(self, 'EAGF orbital q = %d/%d, freq w = %d/%d (%d iterations) @ Rank %d'%(
                    iq+1,len(ps),iomega+1,len(omega_list),solver.niter,rank), *cput1)
                x0 = sol
                for ip, p in enumerate(ps):
                    gf_ao[ip,iq-start,iomega] = np.dot(e_vector_ao[ip,:], sol)
        comm.Barrier()
        gf_ao_gather = comm.gather(gf_ao.transpose(1,0,2))
        if rank == 0:
            gf_ao_gather = np.vstack(gf_ao_gather).transpose(1,0,2)
        comm.Barrier()
        gf_ao_gather = comm.bcast(gf_ao_gather,root=0)
        return gf_ao_gather

    def ipccsd_mo(self, ps, qs, omega_list, broadening):
        '''
        Compute IP-CCSD-GF in MO basis
        '''
        eomip = pyscf.cc.eom_rccsd.EOMIP(self._cc)
        eomip_imds = eomip.make_imds()
        diag = eomip.get_diag()
        e_vector = list()
        for q in qs:
            e_vector.append(greens_e_vector_ip_rhf(self._cc, q))

        segsize = len(ps) // size
        if rank < len(ps)-segsize*size:
            start = rank * segsize + rank
            stop = min(len(ps), start+segsize+1)
        else:
            start = rank * segsize + len(ps)-segsize*size
            stop = min(len(ps), start+segsize)

        gfvals = np.zeros((stop-start, len(qs), len(omega_list)), dtype=np.complex128)
        for ip in range(start,stop):
            p = ps[ip]
            b_vector = greens_b_vector_ip_rhf(self._cc, p)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomip.matvec, vector, curr_omega - 1j * broadening, imds=eomip_imds)

                diag_w = diag + curr_omega-1j*broadening
                x0 = b_vector/diag_w
                solver = gmres.GMRES(matr_multiply, b_vector, x0, diag_w, tol=self.tol)
                cput1 = (time.process_time(), time.perf_counter())
                sol = solver.solve().reshape(-1)
                cput1 = logger.timer(self, 'IPGF orbital p = %d/%d, freq w = %d/%d (%d iterations) @ Rank %d'%(
                    ip+1,len(ps),iomega+1,len(omega_list),solver.niter,rank), *cput1)
                x0 = sol
                for iq, q in enumerate(qs):
                    gfvals[ip-start,iq,iomega] = -np.dot(e_vector[iq], sol)
        comm.Barrier()
        gfvals_gather = comm.gather(gfvals)
        if rank == 0:
            gfvals_gather = np.vstack(gfvals_gather)
        comm.Barrier()
        gfvals_gather = comm.bcast(gfvals_gather,root=0)
        return gfvals_gather

    def eaccsd_mo(self, ps, qs, omega_list, broadening):
        '''
        Compute EA-CCSD-GF in MO basis
        '''
        eomea = pyscf.cc.eom_rccsd.EOMEA(self._cc)
        eomea_imds = eomea.make_imds()
        diag = eomea.get_diag()
        e_vector = list()
        for p in ps:
            e_vector.append(greens_e_vector_ea_rhf(self._cc, p))

        segsize = len(qs) // size
        if rank < len(qs)-segsize*size:
            start = rank * segsize + rank
            stop = min(len(qs), start+segsize+1)
        else:
            start = rank * segsize + len(qs)-segsize*size
            stop = min(len(qs), start+segsize)

        gfvals = np.zeros((len(ps), stop-start, len(omega_list)), dtype=np.complex128)
        for iq in range(start,stop):
            q = qs[iq]
            b_vector = greens_b_vector_ea_rhf(self._cc, q)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomea.matvec, vector, -curr_omega - 1j * broadening, imds=eomea_imds)

                diag_w = diag + (-curr_omega-1j*broadening)
                x0 = b_vector/diag_w
                solver = gmres.GMRES(matr_multiply, b_vector, x0, diag_w, tol=self.tol)
                cput1 = (time.process_time(), time.perf_counter())
                sol = solver.solve().reshape(-1)
                cput1 = logger.timer(self, 'EAGF orbital q = %d/%d, freq w = %d/%d (%d iterations) @ Rank %d'%(
                    iq+1,len(qs),iomega+1,len(omega_list),solver.niter,rank), *cput1)
                x0 = sol
                for ip, p in enumerate(ps):
                    gfvals[ip,iq-start,iomega] = np.dot(e_vector[ip], sol)
        comm.Barrier()
        gfvals_gather = comm.gather(gfvals.transpose(1,0,2))
        if rank == 0:
            gfvals_gather = np.vstack(gfvals_gather).transpose(1,0,2)
        comm.Barrier()
        gfvals_gather = comm.bcast(gfvals_gather,root=0)
        return gfvals_gather

    def get_gf(self, p, q, omega_list, broadening):
        return (self.ipccsd_mo(p, q, omega_list, broadening),
                self.eaccsd_mo(p, q, omega_list, broadening))

