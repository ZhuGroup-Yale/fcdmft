#!/usr/bin/python

import numpy
from pyscf.lib import logger
import pyscf.scf as scf
import numpy as np
from pyscf import ao2mo

'''
HF with fluctuating occupantion and smearing
'''

class RHF(scf.hf.RHF):
    __doc__ = scf.hf.SCF.__doc__

    def __init__(self, mol, mu, smearing=None):
        self.mu = mu
        self.smearing = smearing
        scf.hf.SCF.__init__(self, mol)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        if self.smearing:
            for n,e in enumerate(mo_energy):
                mo_occ[n] = 2./(numpy.exp((e-self.mu)/self.smearing)+1)
        else:
            mo_occ[mo_energy<=self.mu] = 2.
        nmo = mo_energy.size
        nocc = int(numpy.sum(mo_occ) // 2)
        if self.verbose >= logger.INFO and nocc < nmo:
            if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
                logger.warn(self, 'HOMO %.15g == LUMO %.15g',
                            mo_energy[nocc-1], mo_energy[nocc])
            else:
                logger.info(self, '  nelec = %d', nocc*2)
                logger.info(self, '  HOMO = %.15g  LUMO = %.15g',
                            mo_energy[nocc-1], mo_energy[nocc])

        if self.verbose >= logger.DEBUG:
            numpy.set_printoptions(threshold=nmo)
            logger.debug(self, '  mo_energy =\n%s', mo_energy)
            numpy.set_printoptions(threshold=1000)
        return mo_occ

class UHF(scf.uhf.UHF):
    __doc__ = scf.uhf.UHF.__doc__

    def __init__(self, mol, mu, smearing=None):
        self.mu = mu
        self.smearing = smearing
        scf.uhf.UHF.__init__(self, mol)
        self._keys = self._keys.union(['h1e', 'ovlp'])
        self.h1e = None
        self.ovlp = None

    def get_occ(self, mo_energy=None, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        if self.smearing:
            for i in range(2):
                for n,e in enumerate(mo_energy[i]):
                    mo_occ[i][n] = 1./(numpy.exp((e-self.mu)/self.smearing)+1)
        else:
            for i in range(2):
                mo_occ[i][mo_energy[i]<=self.mu] = 1.
        nmo = mo_energy[0].size
        nocca = int(numpy.sum(mo_occ[0]))
        noccb = int(numpy.sum(mo_occ[1]))

        if self.verbose >= logger.INFO and nocca < nmo and noccb > 0 and noccb < nmo:
            if mo_energy[0][nocca-1]+1e-3 > mo_energy[0][nocca]:
                logger.warn(self, 'alpha HOMO %.15g == LUMO %.15g',
                            mo_energy[0][nocca-1], mo_energy[0][nocca])
            else:
                logger.info(self, '  alpha nelec = %d', nocca)
                logger.info(self, '  alpha HOMO = %.15g  LUMO = %.15g',
                            mo_energy[0][nocca-1], mo_energy[0][nocca])

            if mo_energy[1][noccb-1]+1e-3 > mo_energy[1][noccb]:
                logger.warn(self, 'beta HOMO %.15g == LUMO %.15g',
                            mo_energy[1][noccb-1], mo_energy[1][noccb])
            else:
                logger.info(self, '  beta nelec = %d', noccb)
                logger.info(self, '  beta HOMO = %.15g  LUMO = %.15g',
                            mo_energy[1][noccb-1], mo_energy[1][noccb])

        if self.verbose >= logger.DEBUG:
            numpy.set_printoptions(threshold=nmo)
            logger.debug(self, '  mo_energy =\n%s', mo_energy)
            numpy.set_printoptions(threshold=1000)

        if mo_coeff is not None and self.verbose >= logger.DEBUG:
            ss, s = self.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                mo_coeff[1][:,mo_occ[1]>0]), self.get_ovlp())
            logger.debug(self, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
        return mo_occ

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        '''Coulomb (J) and exchange (K)

        Args:
            dm : a list of 2D arrays or a list of 3D arrays
                (alpha_dm, beta_dm) or (alpha_dms, beta_dms)
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self._eri is not None or mol.incore_anyway or self._is_mem_enough():
            if self._eri is None:
                log.error("SCF eri is not initialized.")
                self._eri = mol.intor('int2e', aosym='s8')

            #vj, vk = hf.dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
            vj, vk = _get_jk(dm, self._eri)
        else:
            log.error("Direct SCF not implemented")
            vj, vk = hf.SCF.get_jk(self, mol, dm, hermi, with_j, with_k)
        return vj, vk

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None:
            h1e = self.get_hcore()
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            dm = numpy.array((dm*.5, dm*.5))
        if vhf is None:
            vhf = self.get_veff(self.mol, dm)
        e1 = numpy.einsum('ij,ji', h1e[0], dm[0])
        e1+= numpy.einsum('ij,ji', h1e[1], dm[1])
        e_coul =(numpy.einsum('ij,ji', vhf[0], dm[0]) +
                 numpy.einsum('ij,ji', vhf[1], dm[1])) * .5
        logger.debug(self, 'E1 = %s  Ecoul = %s', e1, e_coul.real)
        return (e1+e_coul).real, e_coul

    def get_hcore(self, *args):
        return self.h1e

    def get_ovlp(self, *args):
        return self.ovlp

class UHFNOMU(scf.uhf.UHF):
    __doc__ = scf.uhf.UHF.__doc__

    def __init__(self, mol, smearing=None):
        scf.uhf.UHF.__init__(self, mol)
        self._keys = self._keys.union(['h1e', 'ovlp'])
        self.h1e = None
        self.ovlp = None

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        '''Coulomb (J) and exchange (K)

        Args:
            dm : a list of 2D arrays or a list of 3D arrays
                (alpha_dm, beta_dm) or (alpha_dms, beta_dms)
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self._eri is not None or mol.incore_anyway or self._is_mem_enough():
            if self._eri is None:
                log.error("SCF eri is not initialized.")
                self._eri = mol.intor('int2e', aosym='s8')

            #vj, vk = hf.dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
            vj, vk = _get_jk(dm, self._eri)
        else:
            log.error("Direct SCF not implemented")
            vj, vk = hf.SCF.get_jk(self, mol, dm, hermi, with_j, with_k)
        return vj, vk

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None:
            h1e = self.get_hcore()
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            dm = numpy.array((dm*.5, dm*.5))
        if vhf is None:
            vhf = self.get_veff(self.mol, dm)
        e1 = numpy.einsum('ij,ji', h1e[0], dm[0])
        e1+= numpy.einsum('ij,ji', h1e[1], dm[1])
        e_coul =(numpy.einsum('ij,ji', vhf[0], dm[0]) +
                 numpy.einsum('ij,ji', vhf[1], dm[1])) * .5
        logger.debug(self, 'E1 = %s  Ecoul = %s', e1, e_coul.real)
        return (e1+e_coul).real, e_coul

    def get_hcore(self, *args):
        return self.h1e

    def get_ovlp(self, *args):
        return self.ovlp

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


