"""
Spin-restricted G0W0 method based the contour deformation scheme.
The scaling N^4 for valence and N^5 for core states.
GW-CD is particularly recommended for accurate core and high-energy states.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    J. Chem. Theory Comput. 14, 4856-4869 (2018)
"""

import os
import time
import numpy as np
import h5py
from scipy.optimize import newton
import scipy.linalg as sla

from fcdmft.utils import arraymath
from fcdmft.gw.mol.gw_ac import GWAC, get_rho_response, set_frozen_orbs, _mo_energy_without_core, _mo_without_core
from fcdmft.ac.grids import _get_scaled_legendre_roots
from fcdmft.utils.arraymath import mkslice

from pyscf import lib
from pyscf.lib import logger, temporary_env

einsum = lib.einsum


def kernel(gw, load_chk=None):
    """GWCD kernel.

    Parameters
    ----------
    gw : GWCD
        instance of GWCD class
    load_chk : str, optional
        name of chkfile to load

    Returns
    -------
    tuple(bool, ndarray, ndarray)
        conv : bool
            True if converged
        conv_inds : ndarray
            conv_inds[p] = 1 if quasiparticle energy for orbital p is converged
        mo_energy : ndarray
            quasiparticle energies
    """
    # local variables for convenience
    mf = gw._scf
    nocc = gw.nocc

    # set frozen orbitals
    set_frozen_orbs(gw)
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz

    # get non-frozen quantities
    mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
    mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)

    if gw.Lpq is None and (not gw.outcore):
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            gw.Lpq = gw.ao2mo(mo_coeff_frz)
    Lpq = gw.Lpq

    nmo = gw.nmo
    mo_energy = np.zeros_like(gw._scf.mo_energy)
    conv_inds = np.zeros(shape=mo_energy.shape, dtype=int)

    if load_chk is not None:
        with h5py.File(load_chk, 'r') as f:
            g = f['gwcd']
            mo_energy = g['mo_energy'][()]
            conv_inds = g['conv_inds'][()]
        if mo_energy.shape != (nmo,):
            raise RuntimeError(f'{load_chk}: mo_energy shape mismatch')
        if conv_inds.shape != mo_energy.shape:
            raise RuntimeError(f'{load_chk}: conv_inds shape mismatch')

    chkfile = gw.chkfile
    if chkfile is not None:
        with h5py.File(chkfile, 'a') as f:
            if 'gwcd' in f:
                del f['gwcd']
            g = f.create_group('gwcd')
            g.create_dataset('mo_energy', data=mo_energy)
            g.create_dataset('conv_inds', data=conv_inds)

    # mean-field exchange-correlation
    with temporary_env(mf, verbose=0):
        v_mf = mo_coeff_frz.T @ (mf.get_veff() - mf.get_j()) @ mo_coeff_frz
    gw.vxc = v_mf

    # exchange self-energy
    if gw.vhf_df and gw.outcore is False:
        # TODO: support smearing
        vk = -einsum('Lpi,Liq->pq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
    else:
        vk = gw.get_sigma_exchange(mo_coeff=mo_coeff_frz)
    gw.vk = vk

    # set up Fermi level
    gw.ef = gw.get_ef(mo_energy=mf.mo_energy)

    # grids for integration on imaginary axis
    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)

    # Compute Wmn(iw) on imaginary axis
    logger.debug(gw, 'Computing the imaginary part')
    Lia = np.ascontiguousarray(gw.Lpq[:, :nocc, nocc:])
    Wmn = get_WmnI_diag(gw, gw.orbs, gw.Lpq, Lia, quad_freqs, mo_energy_frz)

    conv = True
    for p_in_frz, p_in_all in zip(orbs_frz, orbs):
        if gw.qpe_linearized:
            # FIXME
            logger.warn(gw, 'linearization with CD leads to wrong quasiparticle energy')
            raise NotImplementedError

        def quasiparticle(omega):
            sigma = get_sigma_diag(
                gw.ef, omega, p_in_frz, mo_energy_frz, gw.Lpq, Lia, Wmn[:, p_in_frz], quad_freqs, quad_wts, gw.eta
            ).real
            return (
                omega - gw._scf.mo_energy[p_in_all] - (sigma.real + vk[p_in_frz, p_in_frz] - v_mf[p_in_frz, p_in_frz])
            )

        try:
            if p_in_frz < nocc:
                delta = -1e-2
            else:
                delta = 1e-2
            e, result = newton(
                quasiparticle,
                gw._scf.mo_energy[p_in_frz] + delta,
                tol=gw.qpe_tol,
                maxiter=gw.qpe_max_iter,
                full_output=True,
            )
            logger.debug(gw, f'QP energy for orb {p_in_all}: {e}')
            logger.debug(gw, f'Number of iterations: {result.iterations}')
            mo_energy[p_in_all] = e
            conv_inds[p_in_all] = 1
            if chkfile is not None:
                with h5py.File(chkfile, 'r+') as f:
                    g = f['gwcd']
                    g['mo_energy'][()] = mo_energy[()]
                    g['conv_inds'][()] = conv_inds[()]
        except RuntimeError:
            conv = False

    if gw.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=nmo)
        logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)
        np.set_printoptions(threshold=1000)

    return conv, conv_inds, mo_energy


def get_sigma_diag(ef, ep, p, mo_energy, Lpq, Lia, Wmn, freqs, wts, eta):
    """
    Compute self-energy on real axis using contour deformation
    """
    sign = np.sign(ef - mo_energy)

    emo = ep - 1j * eta * sign - mo_energy
    g0 = wts[None, :] * emo[:, None] / ((emo**2)[:, None] + (freqs**2)[None, :])
    sigmaI = -einsum('mw,wm', g0, Wmn) / np.pi

    sigmaR = get_sigmaR_diag(mo_energy, ep, p, ef, Lpq, Lia, eta)
    return sigmaI + sigmaR


def get_sigmaR_diag(mo_energy, omega, orbp, ef, Lpq, Lia, eta):
    """
    Compute self-energy for poles inside contour
    (more and more expensive away from Fermi surface)
    """
    if omega > ef:
        fm = 1.0
        idx = np.where((mo_energy < omega) & (mo_energy > ef))[0]
    else:
        fm = -1.0
        idx = np.where((mo_energy > omega) & (mo_energy < ef))[0]

    nocc = Lia.shape[1]
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]

    sigmaR = 0j
    if len(idx) > 0:
        for m in idx:
            em = mo_energy[m] - omega
            Pi = get_rho_response_R(eia, abs(em), Lia, eta)
            Pi = arraymath.get_id_minus_pi(Pi)
            vec = sla.solve(Pi.T, Lpq[:, orbp, m], check_finite=False, overwrite_a=True, assume_a='sym')
            vec -= Lpq[:, orbp, m]
            sigmaR += fm * np.dot(Lpq[:, m, orbp], vec)
    return sigmaR


def get_WmnI_diag(gw, orbs, Lpq, Lia, quad_freqs, mo_energy):
    """Calculate Wmn on imaginary axis.

    Parameters
    ----------
    gw : GWCD
        GWCD object
    orbs : list
        list of orbital indexes
    Lpq : double 3d array
        three-center density-fitting matrix in MO space
    Lia : double 3d array
        three-center density-fitting matrix in MO space, O-V block
    quad_freqs : double 1d array
        position of imaginary frequency grids used for integration
    mo_energy : double 1d array
        MO energies

    Returns
    -------
    complex 3d array
        Wmn on imaginary axis, shape (nw, norbs, nmo)
    """
    naux, nmo, _ = Lpq.shape
    nw = len(quad_freqs)
    l_slice = Lpq[:, mkslice(orbs), :].reshape(naux, -1)

    naux_ones = np.ones((1, naux))

    norbs = len(orbs)
    Wmn = np.empty((nw, norbs, nmo))
    Pi = np.empty((naux, naux))

    logger.info(gw, f'Computing Wmn on imaginary axis ({nw} points)')

    for w in range(nw):
        if gw.verbose >= 4:
            gw.stdout.write(f'{w:4d} ')
            if w % 12 == 11:
                gw.stdout.write('\n')
            gw.stdout.flush()

        #  Pi_inv = (I - Pi)^-1 - I.
        Pi = get_rho_response(quad_freqs[w], mo_energy, Lia, out=Pi)
        Pi_inv = arraymath.get_id_minus_pi_inv_minus_id(Pi, overwrite_input=True)

        # These lines compute
        # Wmn = einsum('Pmn, Qmn, PQ -> mn', Lpq[:, :, mkslice(orbs)], Lpq[:,  :, mkslice(orbs)], Pi_inv)
        Qmn = np.matmul(Pi_inv, l_slice)

        try:
            import numexpr as ne

            Qmn = ne.evaluate('Qmn * l_slice', out=Qmn)
        except ImportError:
            Qmn *= l_slice
        Wmn[w] = (naux_ones @ Qmn).reshape(norbs, nmo)

    gw.stdout.write('\n')
    gw.stdout.flush()

    # Simple but slow version:
    # for w in range(nw):
    #     Pi = get_rho_response(freqs[w], mo_energy, Lpq[:,:nocc,nocc:])
    #     Pi_inv = np.linalg.inv(np.eye(naux)-Pi)-np.eye(naux)
    #     Qnm = einsum('Pnm,PQ->Qnm',Lpq[:,orbs,:],Pi_inv)
    #     Wmn[:,:,w] = einsum('Qnm,Qmn->mn',Qnm,Lpq[:,:,orbs])

    return Wmn


def get_rho_response_R(eia, omega, Lia, eta):
    """
    Compute density response function in auxiliary basis at poles
    """
    naux, nocc, nvir = Lia.shape

    # Simple but slow version:
    # eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    # eia = 1./(omega+eia+2j*gw.eta) + 1./(-omega+eia)
    # #Pia = einsum('Pia,ia->Pia',Lia,eia)

    # # Response from both spin-up and spin-down density
    # Pi = 2. * einsum('Pia,Qia->PQ',Pia,Lia)
    # return Pi

    naux, nocc, nvir = Lia.shape
    try:
        import numexpr as ne

        eia2_times2 = ne.evaluate('2./(omega+eia+2j*eta) + 2./(-omega+eia)')

        eiaR = np.ascontiguousarray(eia2_times2.real)
        eiaI = np.ascontiguousarray(eia2_times2.imag)

        Pia_R = ne.evaluate('Lia * eiaR')

        Pi_R = Pia_R.reshape(naux, nocc * nvir) @ Lia.reshape(naux, nocc * nvir).T
        del Pia_R

        Pia_I = ne.evaluate('Lia * eiaI')
        Pi_I = Pia_I.reshape(naux, nocc * nvir) @ Lia.reshape(naux, nocc * nvir).T
        del Pia_I
        return Pi_R + 1j * Pi_I
    except ImportError:
        # plain numpy, numexpr not available
        eia = 1.0 / (omega + eia + 2j * eta) + 1.0 / (-omega + eia)
        eiaR = np.ascontiguousarray(eia.real)
        eiaI = np.ascontiguousarray(eia.imag)

        # Response from both spin-up and spin-down density
        PiaR = Lia * (eiaR * 2.0)
        Pi_R = PiaR.reshape(naux, nocc * nvir) @ Lia.reshape(naux, nocc * nvir).T
        del PiaR
        PiaI = Lia * (eiaI * 2.0)
        Pi_I = PiaI.reshape(naux, nocc * nvir) @ Lia.reshape(naux, nocc * nvir).T
        del PiaI
        return Pi_R + 1j * Pi_I


class GWCD(GWAC):
    def __init__(self, mf, frozen=None, auxbasis=None, chkfile=None):
        super().__init__(mf, frozen=frozen, auxbasis=auxbasis)
        self.chkfile = chkfile
        self.eta = 1.0e-3

    def kernel(self):
        """Do one-shot GW calculation using contour deformation."""
        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.converged, self.conv_inds, self.mo_energy = kernel(self)

        logger.timer(self, 'GWCD', *cput0)
        return


if __name__ == '__main__':
    from pyscf import gto, dft, tddft

    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel()

    nocc = mol.nelectron // 2
    nmo = mf.mo_energy.size
    nvir = nmo - nocc

    gw = GWCD(mf)
    gw.orbs = range(0, nocc + 3)
    gw.kernel()
    print(gw.mo_energy)
    assert abs(gw.mo_energy[nocc - 1] - -0.41284735) < 1e-5
    assert abs(gw.mo_energy[nocc] - 0.16574524) < 1e-5
    assert abs(gw.mo_energy[0] - -19.53387986) < 1e-5

    gw = GWCD(mf, chkfile='gwcd_test_test_test.chk')
    gw.orbs = range(0, nocc + 3)
    gw.kernel()
    print(gw.mo_energy)

    with h5py.File('gwcd_test_test_test.chk', 'r') as f:
        g = f['gwcd']
        chk_mo_energy = g['mo_energy'][()]
        chk_conv_inds = g['conv_inds'][()]
    os.remove('gwcd_test_test_test.chk')
    assert abs(chk_mo_energy[nocc - 1] - -0.41284912) < 1e-5
    assert abs(chk_mo_energy[nocc] - 0.16574467) < 1e-5
    assert abs(chk_mo_energy[0] - -19.533888) < 1e-5
    assert np.all(chk_conv_inds[: nocc + 3] == 1)
    print('passed tests!')
