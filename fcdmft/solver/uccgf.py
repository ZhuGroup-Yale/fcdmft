import time
import sys

import numpy as np
import scipy
from fcdmft.solver import gmres

import pyscf
import pyscf.cc
from pyscf.lib import logger
from pyscf.cc.eom_uccsd import amplitudes_to_vector_ip, amplitudes_to_vector_ea
from pyscf.cc.eom_uccsd import vector_to_amplitudes_ip, vector_to_amplitudes_ea
from pyscf import lib

'''
UCCSD Green's function
'''


def greens_b_singles_ea_alpha(t1, p):
    t1a, t1b = t1
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    ds_type = t1a.dtype
    if p < nocca:
        return -t1a[p,:]
    else:
        p = p-nocca
        result = np.zeros((nvira,), dtype=ds_type)
        result[p] = 1.0
        return result

def greens_b_singles_ea_beta(t1, p):
    t1a, t1b = t1
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    ds_type = t1a.dtype
    if p < noccb:
        return -t1b[p,:]
    else:
        p = p-noccb
        result = np.zeros((nvirb,), dtype=ds_type)
        result[p] = 1.0
        return result

def greens_b_doubles_ea_alpha(t2, p):
    t2aa, t2ab, t2bb = t2
    nocca, _, nvira, _ = t2aa.shape
    noccb, _, nvirb, _ = t2bb.shape
    ds_type = t2aa.dtype
    if p < nocca:
        return -t2aa[p,:,:,:], -t2ab[p,:,:,:]
    else:
        return np.zeros((nocca,nvira,nvira), dtype=ds_type), \
                np.zeros((noccb,nvira,nvirb), dtype=ds_type)

def greens_b_doubles_ea_beta(t2, p):
    t2aa, t2ab, t2bb = t2
    nocca, _, nvira, _ = t2aa.shape
    noccb, _, nvirb, _ = t2bb.shape
    ds_type = t2aa.dtype
    if p < noccb:
        return -t2ab[:,p,:,:].transpose(0,2,1), -t2bb[p,:,:,:]
    else:
        return np.zeros((nocca,nvirb,nvira), dtype=ds_type), \
                np.zeros((noccb,nvirb,nvirb), dtype=ds_type)

def greens_b_vector_ea_uhf(cc, p):
    b1a = greens_b_singles_ea_alpha(cc.t1, p)
    b1b = greens_b_singles_ea_beta(cc.t1, p)
    b2aaa, b2bab = greens_b_doubles_ea_alpha(cc.t2, p)
    b2aba, b2bbb = greens_b_doubles_ea_beta(cc.t2, p)
    return amplitudes_to_vector_ea(
        [b1a, b1b], [b2aaa, b2aba, b2bab, b2bbb],
    )

def greens_b_fullvector_ea_uhf(cc, p):
    b1a = greens_b_singles_ea_alpha(cc.t1, p)
    b1b = greens_b_singles_ea_beta(cc.t1, p)
    b2aaa, b2bab = greens_b_doubles_ea_alpha(cc.t2, p)
    b2aba, b2bbb = greens_b_doubles_ea_beta(cc.t2, p)
    return b1a,b1b,b2aaa,b2bab,b2aba,b2bbb

def greens_e_singles_ea_alpha(t1, t2, l1, l2, p):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    ds_type = t1a.dtype
    if p < nocca:
        return l1a[p, :]
    else:
        p = p-nocca
        result = np.zeros((nvira,), dtype=ds_type)
        result[p] = -1.0
        result += lib.einsum('ia,i->a', l1a, t1a[:,p])
        result += 0.5*lib.einsum('klca,klc->a', l2aa, t2aa[:,:,:,p])
        result += lib.einsum('lkac,lkc->a', l2ab, t2ab[:,:,p,:])
        return result

def greens_e_singles_ea_beta(t1, t2, l1, l2, p):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    ds_type = t1a.dtype
    if p < noccb:
        return l1b[p, :]
    else:
        p = p-noccb
        result = np.zeros((nvirb,), dtype=ds_type)
        result[p] = -1.0
        result += lib.einsum('ia,i->a', l1b, t1b[:,p])
        result += 0.5*lib.einsum('klca,klc->a', l2bb, t2bb[:,:,:,p])
        result += lib.einsum('klca,klc->a', l2ab, t2ab[:,:,:,p])
        return result

def greens_e_doubles_ea_alpha(t1, l1, l2, p):
    t1a, t1b = t1
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    ds_type = t1a.dtype
    if p < nocca:
        return 0.5*l2aa[p,:,:,:], l2ab[p,:,:,:]
    else:
        p = p-nocca
        evec_aaa = np.zeros((nocca,nvira,nvira), dtype=ds_type)
        evec_aaa[:,p,:] += -0.5*l1a
        evec_aaa[:,:,p] += 0.5*l1a
        evec_aaa += 0.5*lib.einsum('k,jkba->jab',t1a[:,p], l2aa)
        evec_bab = np.zeros((noccb,nvira,nvirb), dtype=ds_type)
        evec_bab[:,p,:] += -l1b
        evec_bab += lib.einsum('k,kjab->jab',t1a[:,p], l2ab)
        return evec_aaa, evec_bab

def greens_e_doubles_ea_beta(t1, l1, l2, p):
    t1a, t1b = t1
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    ds_type = t1a.dtype
    if p < noccb:
        return l2ab[:,p,:,:].transpose(0,2,1), 0.5*l2bb[p,:,:,:]
    else:
        p = p-noccb
        evec_bbb = np.zeros((noccb,nvirb,nvirb), dtype=ds_type)
        evec_bbb[:,p,:] += -0.5*l1b
        evec_bbb[:,:,p] += 0.5*l1b
        evec_bbb += 0.5*lib.einsum('k,jkba->jab',t1b[:,p], l2bb)
        evec_aba = np.zeros((nocca,nvirb,nvira), dtype=ds_type)
        evec_aba[:,p,:] += -l1a
        evec_aba += lib.einsum('k,jkba->jab',t1b[:,p], l2ab)
        return evec_aba, evec_bbb


def greens_e_vector_ea_uhf(cc, p):
    e1a = greens_e_singles_ea_alpha(cc.t1, cc.t2, cc.l1, cc.l2, p)
    e1b = greens_e_singles_ea_beta(cc.t1, cc.t2, cc.l1, cc.l2, p)
    e2aaa, e2bab = greens_e_doubles_ea_alpha(cc.t1, cc.l1, cc.l2, p)
    e2aba, e2bbb = greens_e_doubles_ea_beta(cc.t1, cc.l1, cc.l2, p)
    return amplitudes_to_vector_ea(
        [e1a, e1b], [e2aaa, e2aba, e2bab, e2bbb],
    )

def greens_e_fullvector_ea_uhf(cc, p):
    e1a = greens_e_singles_ea_alpha(cc.t1, cc.t2, cc.l1, cc.l2, p)
    e1b = greens_e_singles_ea_beta(cc.t1, cc.t2, cc.l1, cc.l2, p)
    e2aaa, e2bab = greens_e_doubles_ea_alpha(cc.t1, cc.l1, cc.l2, p)
    e2aba, e2bbb = greens_e_doubles_ea_beta(cc.t1, cc.l1, cc.l2, p)
    return e1a,e1b,e2aaa,e2bab,e2aba,e2bbb


def greens_b_singles_ip_alpha(t1, p):
    t1a, t1b = t1
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    ds_type = t1a.dtype
    if p < nocca:
        result = np.zeros((nocca,), dtype=ds_type)
        result[p] = 1.0
        return result
    else:
        p = p-nocca
        return t1a[:,p]

def greens_b_singles_ip_beta(t1, p):
    t1a, t1b = t1
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    ds_type = t1a.dtype
    if p < noccb:
        result = np.zeros((noccb,), dtype=ds_type)
        result[p] = 1.0
        return result
    else:
        p = p-noccb
        return t1b[:,p]

def greens_b_doubles_ip_alpha(t2, p):
    t2aa, t2ab, t2bb = t2
    nocca, _, nvira, _ = t2aa.shape
    noccb, _, nvirb, _ = t2bb.shape
    ds_type = t2aa.dtype
    if p < nocca:
        return np.zeros((nocca,nocca,nvira), dtype=ds_type), \
               np.zeros((nocca,noccb,nvirb), dtype=ds_type)
    else:
        p = p-nocca
        return -t2aa[:,:,p,:], -t2ab[:,:,p,:]

def greens_b_doubles_ip_beta(t2, p):
    t2aa, t2ab, t2bb = t2
    nocca, _, nvira, _ = t2aa.shape
    noccb, _, nvirb, _ = t2bb.shape
    ds_type = t2aa.dtype
    if p < noccb:
        return np.zeros((noccb,nocca,nvira), dtype=ds_type), \
               np.zeros((noccb,noccb,nvirb), dtype=ds_type)
    else:
        p = p-noccb
        return -t2ab[:,:,:,p].transpose(1,0,2), -t2bb[:,:,p,:]

def greens_b_vector_ip_uhf(cc, p):
    b1a = greens_b_singles_ip_alpha(cc.t1, p)
    b1b = greens_b_singles_ip_beta(cc.t1, p)
    b2aaa, b2abb = greens_b_doubles_ip_alpha(cc.t2, p)
    b2baa, b2bbb = greens_b_doubles_ip_beta(cc.t2, p)
    return amplitudes_to_vector_ip(
        [b1a, b1b], [b2aaa, b2baa, b2abb, b2bbb],
    )

def greens_b_fullvector_ip_uhf(cc, p):
    b1a = greens_b_singles_ip_alpha(cc.t1, p)
    b1b = greens_b_singles_ip_beta(cc.t1, p)
    b2aaa, b2abb = greens_b_doubles_ip_alpha(cc.t2, p)
    b2baa, b2bbb = greens_b_doubles_ip_beta(cc.t2, p)
    return b1a,b1b,b2aaa,b2abb,b2baa,b2bbb

def greens_e_singles_ip_alpha(t1, t2, l1, l2, p):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    ds_type = t1a.dtype
    if p < nocca:
        result = np.zeros((nocca,), dtype=ds_type)
        result[p] = -1.0
        result += lib.einsum('ic,c->i',l1a,t1a[p,:])
        result += 0.5*lib.einsum('ilcd,lcd->i',l2aa,t2aa[p,:,:,:])
        result += lib.einsum('ilcd,lcd->i',l2ab,t2ab[p,:,:,:])
        return result
    else:
        p = p-nocca
        return -l1a[:,p]

def greens_e_singles_ip_beta(t1, t2, l1, l2, p):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    ds_type = t1a.dtype
    if p < noccb:
        result = np.zeros((noccb,), dtype=ds_type)
        result[p] = -1.0
        result += lib.einsum('ic,c->i',l1b,t1b[p,:])
        result += 0.5*lib.einsum('ilcd,lcd->i',l2bb,t2bb[p,:,:,:])
        result += lib.einsum('lidc,ldc->i',l2ab,t2ab[:,p,:,:])
        return result
    else:
        p = p-noccb
        return -l1b[:,p]

def greens_e_doubles_ip_alpha(t1, l1, l2, p):
    t1a, t1b = t1
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    ds_type = t1a.dtype
    if p < nocca:
        evec_aaa = np.zeros((nocca, nocca, nvira), dtype=ds_type)
        evec_aaa[p, :, :] += -0.5*l1a
        evec_aaa[:, p, :] += 0.5*l1a
        evec_aaa += 0.5*lib.einsum('c,ijcb->ijb',t1a[p,:],l2aa)
        evec_abb = np.zeros((nocca, noccb, nvirb), dtype=ds_type)
        evec_abb[p, :, :] += -l1b
        evec_abb += lib.einsum('c,ijcb->ijb',t1a[p,:],l2ab)
        return -evec_aaa, -evec_abb
    else:
        p = p-nocca
        return 0.5*l2aa[:,:,p,:], l2ab[:,:,p,:]

def greens_e_doubles_ip_beta(t1, l1, l2, p):
    t1a, t1b = t1
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    ds_type = t1a.dtype
    if p < noccb:
        evec_bbb = np.zeros((noccb, noccb, nvirb), dtype=ds_type)
        evec_bbb[p, :, :] += -0.5*l1b
        evec_bbb[:, p, :] += 0.5*l1b
        evec_bbb += 0.5*lib.einsum('c,ijcb->ijb',t1b[p,:],l2bb)
        evec_baa = np.zeros((noccb, nocca, nvira), dtype=ds_type)
        evec_baa[p, :, :] += -l1a
        evec_baa += lib.einsum('c,jibc->ijb',t1b[p,:],l2ab)
        return -evec_baa, -evec_bbb
    else:
        p = p-noccb
        return l2ab[:,:,:,p].transpose(1,0,2), 0.5*l2bb[:,:,p,:]


def greens_e_vector_ip_uhf(cc, p):
    e1a = greens_e_singles_ip_alpha(cc.t1, cc.t2, cc.l1, cc.l2, p)
    e1b = greens_e_singles_ip_beta(cc.t1, cc.t2, cc.l1, cc.l2, p)
    e2aaa, e2abb = greens_e_doubles_ip_alpha(cc.t1, cc.l1, cc.l2, p)
    e2baa, e2bbb = greens_e_doubles_ip_beta(cc.t1, cc.l1, cc.l2, p)
    return amplitudes_to_vector_ip(
        [e1a, e1b], [e2aaa, e2baa, e2abb, e2bbb],
    )

def greens_e_fullvector_ip_uhf(cc, p):
    e1a = greens_e_singles_ip_alpha(cc.t1, cc.t2, cc.l1, cc.l2, p)
    e1b = greens_e_singles_ip_beta(cc.t1, cc.t2, cc.l1, cc.l2, p)
    e2aaa, e2abb = greens_e_doubles_ip_alpha(cc.t1, cc.l1, cc.l2, p)
    e2baa, e2bbb = greens_e_doubles_ip_beta(cc.t1, cc.l1, cc.l2, p)
    return e1a,e1b,e2aaa,e2abb,e2baa,e2bbb

def greens_func_multiply(ham, vector, linear_part, **kwargs):
    return np.array(ham(vector, **kwargs) + linear_part * vector)


class UCCGF(object):
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
        Compute IP-CCSD-GF in AO basis
        '''
        eomip = pyscf.cc.eom_uccsd.EOMIP(self._cc)
        eomip_imds = eomip.make_imds()
        diag = eomip.get_diag()

        t1a, t1b = self._cc.t1
        nocca, nvira = t1a.shape
        noccb, nvirb = t1b.shape
        nmoa = nocca+nvira
        nmob = noccb+nvirb
        nmo = nmoa

        shapea = nocca + nocca*nocca*nvira + nocca*noccb*nvirb
        shapeb = noccb + noccb*noccb*nvirb + noccb*nocca*nvira
        e_vector_moa = np.zeros([nmo,shapea],dtype=np.complex128)
        e_vector_mob = np.zeros([nmo,shapeb],dtype=np.complex128)
        for i in range(nmo):
            e_vector_mo = greens_e_vector_ip_uhf(self._cc, i)
            e1, e2 = vector_to_amplitudes_ip(e_vector_mo, (nmoa,nmob), (nocca,noccb))
            e1a, e1b = e1
            e2aaa, e2baa, e2abb, e2bbb = e2
            e_vector_moa[i,:] = np.hstack((e1a, e2aaa.ravel(), e2abb.ravel()))
            e_vector_mob[i,:] = np.hstack((e1b, e2baa.ravel(), e2bbb.ravel()))
        e_vector_aoa = lib.einsum("pi,ix->px", mo_coeff[0][ps,:], e_vector_moa)
        e_vector_aob = lib.einsum("pi,ix->px", mo_coeff[1][ps,:], e_vector_mob)

        b_vector_moa = np.zeros([shapea,nmo],dtype=np.complex128)
        b_vector_mob = np.zeros([shapeb,nmo],dtype=np.complex128)
        for i in range(nmo):
            b_vector_mo = greens_b_vector_ip_uhf(self._cc, i)
            b1, b2 = vector_to_amplitudes_ip(b_vector_mo, (nmoa,nmob), (nocca,noccb))
            b1a, b1b = b1
            b2aaa, b2baa, b2abb, b2bbb = b2
            b_vector_moa[:,i] = np.hstack((b1a, b2aaa.ravel(), b2abb.ravel()))
            b_vector_mob[:,i] = np.hstack((b1b, b2baa.ravel(), b2bbb.ravel()))
        b_vector_aoa = lib.einsum("xi,ip->xp", b_vector_moa, mo_coeff[0].T[:,ps])
        b_vector_aob = lib.einsum("xi,ip->xp", b_vector_mob, mo_coeff[1].T[:,ps])

        gf_ao = np.zeros((2, len(ps), len(ps), len(omega_list)), dtype=np.complex128)
        for ip, p in enumerate(ps):
            b1a = b_vector_aoa[:,ip][:nocca]
            b2aaa = b_vector_aoa[:,ip][nocca:nocca+nocca*nocca*nvira].reshape(nocca,nocca,nvira)
            b2abb = b_vector_aoa[:,ip][nocca+nocca*nocca*nvira:].reshape(nocca,noccb,nvirb)
            b1b = b_vector_aob[:,ip][:noccb]
            b2baa = b_vector_aob[:,ip][noccb:noccb+noccb*nocca*nvira].reshape(noccb,nocca,nvira)
            b2bbb = b_vector_aob[:,ip][noccb+noccb*nocca*nvira:].reshape(noccb,noccb,nvirb)
            b1 = (b1a, b1b)
            b2 = (b2aaa, b2baa, b2abb, b2bbb)
            b_vector_ao = amplitudes_to_vector_ip(b1, b2)

            x0 = None
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomip.matvec, vector, curr_omega - 1j * broadening, imds=eomip_imds)

                diag_w = diag + curr_omega-1j*broadening
                if x0 is None:
                    x0 = b_vector_ao/diag_w
                solver = gmres.GMRES(matr_multiply, b_vector_ao, x0, diag_w, tol=self.tol)
                cput1 = (time.process_time(), time.perf_counter())
                sol = solver.solve().reshape(-1)
                cput1 = logger.timer(self, 'IPGF GMRES orbital p = %d/%d, freq w = %d/%d (%d iterations)'%(
                    ip+1,len(ps),iomega+1,len(omega_list),solver.niter), *cput1)
                x0 = sol

                sol1, sol2 = vector_to_amplitudes_ip(sol, (nmoa,nmob), (nocca,noccb))
                sol1a, sol1b = sol1
                sol2aaa, sol2baa, sol2abb, sol2bbb = sol2
                sol_a = np.hstack((sol1a, sol2aaa.ravel(), sol2abb.ravel()))
                sol_b = np.hstack((sol1b, sol2baa.ravel(), sol2bbb.ravel()))

                for iq, q in enumerate(ps):
                    gf_ao[0,ip,iq,iomega] = -np.dot(e_vector_aoa[iq,:], sol_a)
                    gf_ao[1,ip,iq,iomega] = -np.dot(e_vector_aob[iq,:], sol_b)
        return gf_ao

    def eaccsd_ao(self, ps, omega_list, mo_coeff, broadening):
        '''
        Compute EA-CCSD-GF in AO basis
        '''
        eomea = pyscf.cc.eom_uccsd.EOMEA(self._cc)
        eomea_imds = eomea.make_imds()
        diag = eomea.get_diag()

        t1a, t1b = self._cc.t1
        nocca, nvira = t1a.shape
        noccb, nvirb = t1b.shape
        nmoa = nocca+nvira
        nmob = noccb+nvirb
        nmo = nmoa

        shapea = nvira + nvira*nocca*nvira + nvira*noccb*nvirb
        shapeb = nvirb + nvirb*nocca*nvira + nvirb*noccb*nvirb
        e_vector_moa = np.zeros([nmo,shapea],dtype=np.complex128)
        e_vector_mob = np.zeros([nmo,shapeb],dtype=np.complex128)
        for i in range(nmo):
            e_vector_mo = greens_e_vector_ea_uhf(self._cc, i)
            e1, e2 = vector_to_amplitudes_ea(e_vector_mo, (nmoa,nmob), (nocca,noccb))
            e1a, e1b = e1
            e2aaa, e2aba, e2bab, e2bbb = e2
            e_vector_moa[i,:] = np.hstack((e1a, e2aaa.ravel(), e2bab.ravel()))
            e_vector_mob[i,:] = np.hstack((e1b, e2aba.ravel(), e2bbb.ravel()))
        e_vector_aoa = lib.einsum("pi,ix->px", mo_coeff[0][ps,:], e_vector_moa)
        e_vector_aob = lib.einsum("pi,ix->px", mo_coeff[1][ps,:], e_vector_mob)

        b_vector_moa = np.zeros([shapea,nmo],dtype=np.complex128)
        b_vector_mob = np.zeros([shapeb,nmo],dtype=np.complex128)
        for i in range(nmo):
            b_vector_mo = greens_b_vector_ea_uhf(self._cc, i)
            b1, b2 = vector_to_amplitudes_ea(b_vector_mo, (nmoa,nmob), (nocca,noccb))
            b1a, b1b = b1
            b2aaa, b2aba, b2bab, b2bbb = b2
            b_vector_moa[:,i] = np.hstack((b1a, b2aaa.ravel(), b2bab.ravel()))
            b_vector_mob[:,i] = np.hstack((b1b, b2aba.ravel(), b2bbb.ravel()))
        b_vector_aoa = lib.einsum("xi,ip->xp", b_vector_moa, mo_coeff[0].T[:,ps])
        b_vector_aob = lib.einsum("xi,ip->xp", b_vector_mob, mo_coeff[1].T[:,ps])

        gf_ao = np.zeros((2, len(ps), len(ps), len(omega_list)), dtype=np.complex128)
        for ip, p in enumerate(ps):
            b1a = b_vector_aoa[:,ip][:nvira]
            b2aaa = b_vector_aoa[:,ip][nvira:nvira+nvira*nocca*nvira].reshape(nocca,nvira,nvira)
            b2bab = b_vector_aoa[:,ip][nvira+nvira*nocca*nvira:].reshape(noccb,nvira,nvirb)
            b1b = b_vector_aob[:,ip][:nvirb]
            b2aba = b_vector_aob[:,ip][nvirb:nvirb+nocca*nvirb*nvira].reshape(nocca,nvirb,nvira)
            b2bbb = b_vector_aob[:,ip][nvirb+nocca*nvirb*nvira:].reshape(noccb,nvirb,nvirb)
            b1 = (b1a, b1b)
            b2 = (b2aaa, b2aba, b2bab, b2bbb)
            b_vector_ao = amplitudes_to_vector_ea(b1, b2)

            x0 = None
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomea.matvec, vector, -curr_omega - 1j * broadening, imds=eomea_imds)

                diag_w = diag + (-curr_omega-1j*broadening)
                if x0 is None:
                    x0 = b_vector_ao/diag_w
                solver = gmres.GMRES(matr_multiply, b_vector_ao, x0, diag_w, tol=self.tol)
                cput1 = (time.process_time(), time.perf_counter())
                sol = solver.solve().reshape(-1)
                cput1 = logger.timer(self, 'EAGF GMRES orbital q = %d/%d, freq w = %d/%d (%d iterations)'%(
                    ip+1,len(ps),iomega+1,len(omega_list),solver.niter), *cput1)
                x0 = sol

                sol1, sol2 = vector_to_amplitudes_ea(sol, (nmoa,nmob), (nocca,noccb))
                sol1a, sol1b = sol1
                sol2aaa, sol2aba, sol2bab, sol2bbb = sol2
                sol_a = np.hstack((sol1a, sol2aaa.ravel(), sol2bab.ravel()))
                sol_b = np.hstack((sol1b, sol2aba.ravel(), sol2bbb.ravel()))
                for iq, q in enumerate(ps):
                    gf_ao[0,iq,ip,iomega] = np.dot(e_vector_aoa[iq,:], sol_a)
                    gf_ao[1,iq,ip,iomega] = np.dot(e_vector_aob[iq,:], sol_b)
        return gf_ao

    def ipccsd_mo(self, ps, qs, omega_list, broadening):
        '''
        Compute IP-CCSD-GF in MO basis
        '''
        eomip = pyscf.cc.eom_uccsd.EOMIP(self._cc)
        eomip_imds = eomip.make_imds()
        diag = eomip.get_diag()
        e_vector = list()
        t1a, t1b = self._cc.t1
        nocca, nvira = t1a.shape
        noccb, nvirb = t1b.shape
        nmoa = nocca+nvira
        nmob = noccb+nvirb
        for q in qs:
            e_vector.append(greens_e_vector_ip_uhf(self._cc, q))

        gfvals = np.zeros((2, len(ps), len(qs), len(omega_list)), dtype=complex)
        for ip, p in enumerate(ps):
            b_vector = greens_b_vector_ip_uhf(self._cc, p)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomip.matvec, vector, curr_omega - 1j * broadening, imds=eomip_imds)

                diag_w = diag + curr_omega - 1j*broadening
                x0 = b_vector/diag_w
                solver = gmres.GMRES(matr_multiply, b_vector, x0, diag_w, tol=self.tol)
                cput1 = (time.process_time(), time.perf_counter())
                sol = solver.solve().reshape(-1)
                cput1 = logger.timer(self, 'IPGF GMRES orbital p = %d/%d, freq w = %d/%d (%d iterations)'%(
                    ip+1,len(ps),iomega+1,len(omega_list),solver.niter), *cput1)
                x0 = sol

                sol1, sol2 = vector_to_amplitudes_ip(sol, (nmoa,nmob), (nocca,noccb))
                sol1a, sol1b = sol1
                sol2aaa, sol2baa, sol2abb, sol2bbb = sol2
                sol_a = np.hstack((sol1a, sol2aaa.ravel(), sol2abb.ravel()))
                sol_b = np.hstack((sol1b, sol2baa.ravel(), sol2bbb.ravel()))

                for iq, q in enumerate(qs):
                    e1, e2 = vector_to_amplitudes_ip(e_vector[iq], (nmoa,nmob), (nocca,noccb))
                    e1a, e1b = e1
                    e2aaa, e2baa, e2abb, e2bbb = e2
                    e_vector_a = np.hstack((e1a, e2aaa.ravel(), e2abb.ravel()))
                    e_vector_b = np.hstack((e1b, e2baa.ravel(), e2bbb.ravel()))
                    gfvals[0,ip,iq,iomega] = -np.dot(e_vector_a, sol_a)
                    gfvals[1,ip,iq,iomega] = -np.dot(e_vector_b, sol_b)
        return gfvals

    def eaccsd_mo(self, ps, qs, omega_list, broadening):
        '''
        Compute EA-CCSD-GF in MO basis
        '''
        eomea = pyscf.cc.eom_uccsd.EOMEA(self._cc)
        eomea_imds = eomea.make_imds()
        diag = eomea.get_diag()
        e_vector = list()
        t1a, t1b = self._cc.t1
        nocca, nvira = t1a.shape
        noccb, nvirb = t1b.shape
        nmoa = nocca+nvira
        nmob = noccb+nvirb
        for p in ps:
            e_vector.append(greens_e_vector_ea_uhf(self._cc, p))

        gfvals = np.zeros((2, len(ps), len(qs), len(omega_list)), dtype=complex)
        for iq, q in enumerate(qs):
            b_vector = greens_b_vector_ea_uhf(self._cc, q)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomea.matvec, vector, -curr_omega - 1j * broadening, imds=eomea_imds)

                diag_w = diag + (-curr_omega - 1j*broadening)
                x0 = b_vector/diag_w
                solver = gmres.GMRES(matr_multiply, b_vector, x0, diag_w, tol=self.tol)
                cput1 = (time.process_time(), time.perf_counter())
                sol = solver.solve().reshape(-1)
                cput1 = logger.timer(self, 'EAGF GMRES orbital q = %d/%d, freq w = %d/%d (%d iterations)'%(
                    iq+1,len(ps),iomega+1,len(omega_list),solver.niter), *cput1)
                x0 = sol

                sol1, sol2 = vector_to_amplitudes_ea(sol, (nmoa,nmob), (nocca,noccb))
                sol1a, sol1b = sol1
                sol2aaa, sol2aba, sol2bab, sol2bbb = sol2
                sol_a = np.hstack((sol1a, sol2aaa.ravel(), sol2bab.ravel()))
                sol_b = np.hstack((sol1b, sol2aba.ravel(), sol2bbb.ravel()))

                for ip, p in enumerate(ps):
                    e1, e2 = vector_to_amplitudes_ea(e_vector[ip], (nmoa,nmob), (nocca,noccb))
                    e1a, e1b = e1
                    e2aaa, e2aba, e2bab, e2bbb = e2
                    e_vector_a = np.hstack((e1a, e2aaa.ravel(), e2bab.ravel()))
                    e_vector_b = np.hstack((e1b, e2aba.ravel(), e2bbb.ravel()))
                    gfvals[0,ip,iq,iomega] = np.dot(e_vector_a, sol_a)
                    gfvals[1,ip,iq,iomega] = np.dot(e_vector_b, sol_b)
        return gfvals

    def get_gf(self, p, q, omega_list, broadening):
        return (self.ipccsd_mo(p, q, omega_list, broadening), 
                self.eaccsd_mo(p, q, omega_list, broadening))

