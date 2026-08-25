"""
Spin-restricted space-time G0W0 method.

References:
    J. Phys. Chem. Lett. 9, 2, 306-312 (2018)
    J. Chem. Theory Comput. 17, 3, 1662-1677 (2021)
    Phys. Rev. B 94, 165109 (2016)
"""

import numpy as np
from scipy.optimize import newton
import time

from pyscf import dft, scf
from pyscf.lib import einsum, logger, temporary_env

from fcdmft.ac.pade import PadeAC
from fcdmft.ac.two_pole import TwoPoleAC
from fcdmft.gw.minimax import fourier_transform
from fcdmft.gw.minimax.minimax_rpa import get_frequency_minimax_grid
from fcdmft.gw.minimax.minimax_exp_gw import get_time_minimax_grid
from fcdmft.gw.mol.gw_ac import GWAC, get_g0


def kernel(gw):
    """Run one-shot space-time GW calculation.

    Parameters
    ----------
    gw : GWSpaceTime
        gw object
    mo_energy : double 1d array
        orbital energy
    mo_coeff : double 2d array
        coefficient from AO to MO
    Lpq : double 3d array
        3-center density-fitting matrix in AO space
    """
    # local variables for convenience
    nmo = gw.nmo
    nocc = gw.nocc
    mf = gw._scf
    Lpq = gw.Lpq
    mo_energy = gw.mo_energy
    mo_coeff = gw.mo_coeff

    # mean-field exchange-correlation potential
    with temporary_env(mf, verbose=0):
        gw.vxc = vxc = mo_coeff.T @ (mf.get_veff() - mf.get_j()) @ mo_coeff

    # exchange self-energy
    dm = mf.make_rdm1()
    with temporary_env(gw.mol, verbose=0):
        # keep this condition for embedding calculations
        if (not isinstance(gw._scf, dft.rks.RKS)) and isinstance(gw._scf, scf.hf.RHF):
            rhf = gw._scf
        else:
            rhf = gw._scf.to_hf()
        gw.vk = vk = mo_coeff.T @ (rhf.get_veff(gw.mol, dm) - rhf.get_j(gw.mol, dm)) @ mo_coeff

    # get minimax grid
    gw.initialize_grid()
    t, w = gw.t, gw.w
    wt_t_to_w_cos, wt_t_to_w_sin, wt_w_to_t_cos = gw.wt_t_to_w_cos, gw.wt_t_to_w_sin, gw.wt_w_to_t_cos
    if gw.fix_ef is not None:
        gw.ef = ef = gw.fix_ef
    else:
        gw.ef = ef = gw.get_ef()

    # Green's function in AO on it and -it
    if gw.g0_t_pos_ao is None and gw.g0_t_neg_ao is None:
        g0_t_pos_ao, g0_t_neg_ao = get_g0_t(nocc=nocc, mo_energy=mo_energy, t=t, ef=ef, mo_coeff=mo_coeff)
    else:
        g0_t_pos_ao, g0_t_neg_ao = gw.g0_t_pos_ao, gw.g0_t_neg_ao

    # P(it) = G(it)G(-it), Pi(it)=P(it)V
    Pi_t = get_pi_t(g_t_pos=g0_t_pos_ao, g_t_neg=g0_t_neg_ao, Lpq=Lpq)

    # Pi(iw) = FT[Pi(it)]
    Pi_w = einsum('tPQ,tw->wPQ', Pi_t, np.cos(w[None, :] * t[:, None]) * wt_t_to_w_cos)

    # Wc^-1(iw) = [1-Pi(iw)]^-1 - 1
    wc_w = get_wc_w(Pi_w=Pi_w)
    wc_t = einsum('wPQ,wt->tPQ', wc_w, np.cos(w[:, None] * t[None, :]) * wt_w_to_t_cos)
    gw.FT_test_bosonic(wc_w, 'Wc(iw)<->Wc(it)')

    # Sigma(it)=G(it)W(it), Sigma(iw)=FT[Sigma(it)]
    sigma_t_pos, sigma_t_neg = get_imag_sigma_t(
        g_t_pos=g0_t_pos_ao, g_t_neg=g0_t_neg_ao, wc_t=wc_t, Lpq=Lpq, mo_coeff=mo_coeff, fullsigma=gw.fullsigma,
        fast_diag=gw.fast_diag)
    gw.sigma_t_pos, gw.sigma_t_neg = sigma_t_pos, sigma_t_neg
    sigma_t_cos = 0.5 * (sigma_t_pos + sigma_t_neg)
    sigma_t_sin = 0.5 * (sigma_t_neg - sigma_t_pos)
    if gw.fullsigma is True:
        sigma_w_cos = einsum('tpq,tw->wpq', sigma_t_cos, np.cos(w[None, :] * t[:, None]) * wt_t_to_w_cos)
        sigma_w_sin = einsum('tpq,tw->wpq', sigma_t_sin, np.sin(w[None, :] * t[:, None]) * wt_t_to_w_sin)
    else:
        sigma_w_cos = einsum('tp,tw->wp', sigma_t_cos, np.cos(w[None, :] * t[:, None]) * wt_t_to_w_cos)
        sigma_w_sin = einsum('tp,tw->wp', sigma_t_sin, np.sin(w[None, :] * t[:, None]) * wt_t_to_w_sin)
    gw.sigma_w = sigma_w_cos + 1j * sigma_w_sin

    if gw.fullsigma is True:
        gw.FT_test_fermionic(gw.sigma_w, 'Sigma(iw)<->Sigma(it)')

    # analytical continuation: sigma(iw)->sigma(w)
    if gw.ac == 'twopole':
        acobj = TwoPoleAC(range(nmo), nocc)
    elif gw.ac == 'pade':
        acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
    acobj.ac_fit(gw.sigma_w, 1j * w, axis=0)

    # get GW quasiparticle energy
    if gw.fullsigma:
        diag_acobj = acobj.diagonal(axis1=0, axis2=1)
    else:
        diag_acobj = acobj

    mo_energy = np.zeros_like(gw._scf.mo_energy)
    for p in range(nmo):
        if gw.qpe_linearized is True:
            de = 1e-6
            sigmaR = diag_acobj[p].ac_eval(gw._scf.mo_energy[p] - ef).real
            dsigma = diag_acobj[p].ac_eval(gw._scf.mo_energy[p] - ef + de).real - sigmaR.real
            zn = 1.0 / (1.0 - dsigma / de)
            mo_energy[p] = gw._scf.mo_energy[p] + zn * (sigmaR.real + vk[p, p] - vxc[p, p])
        else:
            def quasiparticle(omega):
                sigmaR = diag_acobj[p].ac_eval(omega - ef).real
                return omega - gw._scf.mo_energy[p] - (sigmaR.real + vk[p, p] - vxc[p, p])

            try:
                mo_energy[p] = newton(quasiparticle, gw._scf.mo_energy[p], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter)
            except RuntimeError:
                logger.warn(gw, 'QPE for orbital=%d not converged!', p)

    # save GW results
    gw.acobj = acobj
    gw.mo_energy = mo_energy
    with np.printoptions(threshold=len(mo_energy)):
        logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)
    logger.warn(gw, 'GW QP energies may not be sorted from min to max')

    return


def get_g0_t(nocc, mo_energy, t, ef, mo_coeff=None):
    r"""Get non-interacting Green's function in the imaginary time domain.
    [G_mo]_{pq} (t) = \delta_{pq} \Theta(F-p) exp((mo_energy[p]-mu)t)   t>0
                                  \Theta(p-F)-exp((mo_energy[p]-mu)t)   t<0
    the prefactor i is not included.
    if mo_coeff is not None, G_mo is rotated to G_ao.

    Parameters
    ----------
    nocc : int
        number of occupied orbitals
    mo_energy : double 1d array
        orbital energy
    t : double 1d array
        time grid
    ef : double
        Fermi level
    mo_coeff : double 2d array, optional
        coefficient from AO to MO, by default None

    Returns
    -------
    g0_t_pos : double 3d array
        G0 on the positive imaginary time in MO/AO space
    g0_t_neg : double 3d array
        G0 on the negative imaginary time in MO/AO space
    """
    assert t.all() > 0
    assert mo_energy[nocc - 1] < ef and mo_energy[nocc] > ef
    nmo = mo_energy.shape[0]
    nt = t.shape[0]
    g0_t_pos = np.zeros(shape=[nt, nmo, nmo], dtype=np.double)
    g0_t_neg = np.zeros_like(g0_t_pos)
    for it in range(nt):
        exp_pos = np.exp((mo_energy[:nocc] - ef) * t[it])
        exp_neg = -np.exp((ef - mo_energy[nocc:]) * t[it])

        g0_t_pos[it, :nocc, :nocc] = np.diag(exp_pos)
        g0_t_neg[it, nocc:, nocc:] = np.diag(exp_neg)

    if mo_coeff is not None:
        g0_t_pos = mo_coeff @ g0_t_pos @ mo_coeff.T
        g0_t_neg = mo_coeff @ g0_t_neg @ mo_coeff.T
    return g0_t_pos, g0_t_neg


def get_pi_t(g_t_pos, g_t_neg, Lpq):
    """Get Pi(t)=P(t)V, where the response function P(t)=G(t)G(-t).
    G and Lpq must be in the same space (AO, MO or LO).

    Parameters
    ----------
    g_t_pos : double 3d array
        Green's function on positive imaginary time
    g_t_neg : double 3d array
        Green's function on negative imaginary time
    Lpq : double 3d array
        three-center density-fitting matrix

    Returns
    -------
    Pi_t : double 3d array
        Pi on positive imaginary time domain in auxiliary space
    """
    naux = Lpq.shape[0]
    nt = g_t_pos.shape[0]
    Pi_t = np.zeros(shape=[nt, naux, naux], dtype=np.double)
    for it in range(nt):
        Pkj = einsum('Pij,ki->Pkj', Lpq, g_t_pos[it]).reshape(naux, -1)
        Qkj = einsum('Qkl,lj->Qkj', Lpq, g_t_neg[it]).reshape(naux, -1)
        Pi_t[it] = 2.0 * Pkj @ Qkj.T  # factor 2.0 is for two spin channels
    return Pi_t


def get_wc_w(Pi_w):
    """Get the correlation part screened interaction W in auxiliary orbital space and imaginary frequency space.
    Wc(w)=[I-Pi(w)]^-1 - I

    Parameters
    ----------
    Pi_w : double 3d array
        Pi on positive imaginary time domain in auxiliary space

    Returns
    -------
    wc_w : double 3d array
        Wc on positive imaginary time domain in auxiliary space
    """
    naux = Pi_w.shape[-1]
    identity = np.eye(naux)
    wc_w = np.linalg.inv(identity - Pi_w) - identity[None, :, :]
    return wc_w


def get_imag_sigma_t(g_t_pos, g_t_neg, wc_t, Lpq, mo_coeff=None, fullsigma=False, fast_diag=True):
    """Get the correlation self-energy in the imaginary time domain.
    this function is similar to equation (28) in 10.1021/acs.jctc.0c01282
    but we get sigma_c in AO first, then transform to the MO space

    Parameters
    ----------
    g_t_pos : double 3d array
        Green's function on the positive imaginary time
    g_t_neg : double 3d array
        Green's function on the negative imaginary time
    wc_t : double 3d array
        Wc matrix in the auxiliary space
    Lpq : double 3d array
        3-center density-fitting matrix in the same space of G
    mo_coeff : double 2d array, optional
        coefficient from initial space to final space, by default None
    fullsigma : bool, optional
        calculate off-diagonal self-energy, by default False
    fast_diag : bool, optional
        double the memory cost to speed up diagonal-only calculation, by default True

    Returns
    -------
    sigma_t_pos : double 2d or 3d array
        self-energy on the positive imaginary time in MO
    sigma_t_neg : double 2d or 3d array
        self-energy on the negative imaginary time in MO
    """
    nmo = Lpq.shape[1]
    nt = g_t_pos.shape[0]
    if fullsigma is False:
        sigma_t_pos = np.zeros(shape=[nt, nmo], dtype=np.double)
        sigma_t_neg = np.zeros_like(sigma_t_pos)
    else:
        sigma_t_pos = np.zeros(shape=[nt, nmo, nmo], dtype=np.double)
        sigma_t_neg = np.zeros_like(sigma_t_pos)
    if fullsigma is False and fast_diag is True:
        assert mo_coeff is not None
        Lma = einsum('Pmn,mi->Pin', Lpq, mo_coeff)  # indexed as in aux/mo/ao space
    for it in range(nt):
        if fullsigma is False:
            if fast_diag is True:
                t1_pos = einsum('ij,Pki->Pkj', g_t_pos[it], Lma)
                t1_neg = einsum('ij,Pki->Pkj', g_t_neg[it], Lma)
                t2 = einsum('PQ,Qij->Pij', wc_t[it], Lma)
                sigma_t_pos[it] = -einsum('Pik,Pik->i', t1_pos, t2)
                sigma_t_neg[it] = -einsum('Pik,Pik->i', t1_neg, t2)
            else:
                t1_pos = einsum('ij,Pki->Pkj', g_t_pos[it], Lpq)
                t1_neg = einsum('ij,Pki->Pkj', g_t_neg[it], Lpq)
                t2 = einsum('PQ,Qij->Pij', wc_t[it], Lpq)
                sigma_pos_ao = einsum('Pik,Pjk->ij', t1_pos, t2)
                sigma_neg_ao = einsum('Pik,Pjk->ij', t1_neg, t2)
                sigma_t_pos[it] = -(mo_coeff.T @ sigma_pos_ao @ mo_coeff).diagonal()
                sigma_t_neg[it] = -(mo_coeff.T @ sigma_neg_ao @ mo_coeff).diagonal()
        else:
            t1_pos = einsum('ij,Pki->Pkj', g_t_pos[it], Lpq)
            t1_neg = einsum('ij,Pki->Pkj', g_t_neg[it], Lpq)
            t2 = einsum('PQ,Qij->Pij', wc_t[it], Lpq)
            sigma_t_pos[it] = -einsum('Pik,Pjk->ij', t1_pos, t2)
            sigma_t_neg[it] = -einsum('Pik,Pjk->ij', t1_neg, t2)
            if mo_coeff is not None:
                sigma_t_pos[it] = mo_coeff.T @ sigma_t_pos[it] @ mo_coeff
                sigma_t_neg[it] = mo_coeff.T @ sigma_t_neg[it] @ mo_coeff
    return sigma_t_pos, sigma_t_neg


def get_minimax_grid(k, E_min, E_max):
    """Get minimax grid for time and frequency.

    Parameters
    ----------
    k : int
        number of time/frequency grids
    E_min : double
        minimum frequency range
    E_max : double
        maximum frequency range

    Returns
    -------
    t : double 1d array
        scaled time grid
    t_weight : double 1d array
        scaled time grid weight
    tw : double 1d array
        scaled frequency grid
    w_weight : double 1d array
        scaled frequency grid weight
    """
    E_range = E_max / E_min
    # if k > 20 and E_range < 100:
    #    print("You requested a large minimax grid (> 20 points) for a small minimax range R (R < 100).")
    #    print("That may lead to numerical instabilities when computing minimax grid weights.\n")
    t, t_weight = get_time_minimax_grid(k, E_range)
    w, w_weight = get_frequency_minimax_grid(k, E_range)
    # scaling grid and weight, equation (26) and (27) in 10.1021/ct5001268
    t /= 2.0 * E_min
    t_weight /= 2.0 * E_min
    w *= E_min
    w_weight *= E_min
    return t, t_weight, w, w_weight


def get_ft_weight(E_min, E_max, t, w, num=200, return_error=False):
    """Get Fourier transform weights between time and frequency domain.

    Parameters
    ----------
    E_min : double
        minimum frequency range
    E_max : double
        maximum frequency range
    t : double 1d array
        scaled time grid
    w : double 1d array
        scaled frequency grid
    num : int, optional
        number of points per magnitude in least square fitting, by default 200
    return_error : bool, optional
        return least square error for FT weights, by default False

    Returns
    -------
    wt_t_to_w_cos : double 2d array
        cosine t to w weight, [nt,nw]
    wt_t_to_w_sin : double 2d array
        sine t to w weight, [nt,nw]
    wt_w_to_t_cos : double 2d array
        cosine w to t weight, [nw,nt]
    wt_w_to_t_sin : double 2d array
        sine w to t weight, [nw,nt]
    error_t_to_w_cos : double
        cosine t to w error
    error_t_to_w_sin : double
        sine t to w error
    error_w_to_t_cos : double
        cosine w to t error
    error_w_to_t_sin : double
        sine w to t error
    """
    wt_t_to_w_cos, error_t_to_w_cos = fourier_transform.get_l_sq_wghts_cos_tf_t_to_w(
        t, w, E_min, E_max, num_points_per_magnitude=num
    )
    wt_t_to_w_sin, error_t_to_w_sin = fourier_transform.get_l_sq_wghts_sin_tf_t_to_w(
        t, w, E_min, E_max, num_points_per_magnitude=num
    )
    wt_w_to_t_cos, error_w_to_t_cos = fourier_transform.get_l_sq_wghts_cos_tf_w_to_t(
        t, w, E_min, E_max, num_points_per_magnitude=num
    )
    wt_w_to_t_sin, error_w_to_t_sin = fourier_transform.get_l_sq_wghts_sin_tf_w_to_t(
        t, w, E_min, E_max, num_points_per_magnitude=num
    )

    if return_error is False:
        return wt_t_to_w_cos.T, wt_t_to_w_sin.T, wt_w_to_t_cos.T, wt_w_to_t_sin.T
    else:
        return wt_t_to_w_cos.T, wt_t_to_w_sin.T, wt_w_to_t_cos.T, wt_w_to_t_sin.T,\
            error_t_to_w_cos, error_t_to_w_sin, error_w_to_t_cos, error_w_to_t_sin


def FT_test_bosonic(gw, m_w, message):
    """Test accuracy of Fourier transform for Bosonic functions.
    Grids need to be in the last dimension.
    See equation 4 and 6 in 10.1103/PhysRevB.94.165109.
    Typical error magnitudes:
    1 cycle ~ 1.0e-6
    5 cycles ~ 1.0e-5
    10 cycles ~ 1.0e-3

    Parameters
    ----------
    gw : SCGW
        gw object
    m_w : double 3d array
        Bosonic function on positive imaginary frequency
    message : str
        message to print

    Returns
    -------
    first_cycle_error : double
        error in first Fourier transform cycle
    """
    ERROR_THRESHOLD = 0.1
    if gw.FT_cycle <= 0:
        return

    logger.info(gw, '\ntest FT accuracy: %s', message)
    m_w_FT = m_w.copy()
    first_cycle_error = 0
    for i in range(gw.FT_cycle):
        m_t_FT = einsum('wPQ,wt->tPQ', m_w_FT, np.cos(gw.w[:, None] * gw.t[None, :]) * gw.wt_w_to_t_cos)
        m_w_FT = einsum('tPQ,tw->wPQ', m_t_FT, np.cos(gw.w[None, :] * gw.t[:, None]) * gw.wt_t_to_w_cos)
        max_error = np.max(np.abs(m_w - m_w_FT))
        logger.info(gw, '%d cycles error = %.6f', i + 1, max_error)
        if i == 0:
            first_cycle_error = max_error
        if gw.FT_error_stop is True and i == 0 and max_error > ERROR_THRESHOLD:
            logger.error(gw, 'large error detected, exit.')
    return first_cycle_error


def FT_test_fermionic(gw, m_w, message):
    """Test accuracy of Fourier transform for Fermionic functions.
    Grids need to be in the last dimension.
    Frequency to time: equation 13 and 14 in 10.1103/PhysRevB.98.155143.
    Time to frequency: equation 67-71 in 10.1103/PhysRevB.94.165109.
    Typical error magnitudes:
    1 cycle ~ 1.0e-6
    5 cycles ~ 1.0e-4
    10 cycles ~ 1.0e-1

    Parameters
    ----------
    gw : SCGW
        gw object
    m_w : complex 3d array
        Fermionic function on positive imaginary frequency
    message : str
        message to print

    Returns
    -------
    first_cycle_error : double
        error in first Fourier transform cycle
    """
    ERROR_THRESHOLD = 0.1
    if gw.FT_cycle <= 0:
        return

    logger.info(gw, '\ntest FT accuracy: %s', message)
    m_w_FT = m_w.copy()
    first_cycle_error = 0
    for i in range(gw.FT_cycle):
        m_t_pos_FT = einsum('wpq,wt->tpq', m_w_FT.real, np.cos(gw.w[:, None] * gw.t[None, :]) * gw.wt_w_to_t_cos)
        m_t_pos_FT -= einsum('wpq,wt->tpq', m_w_FT.imag, np.sin(gw.w[:, None] * gw.t[None, :]) * gw.wt_w_to_t_sin)
        m_t_neg_FT = einsum('wpq,wt->tpq', m_w_FT.real, np.cos(gw.w[:, None] * gw.t[None, :]) * gw.wt_w_to_t_cos)
        m_t_neg_FT += einsum('wpq,wt->tpq', m_w_FT.imag, np.sin(gw.w[:, None] * gw.t[None, :]) * gw.wt_w_to_t_sin)
        m_t_cos_FT = 0.5 * (m_t_pos_FT + m_t_neg_FT)
        m_t_sin_FT = 0.5 * (m_t_neg_FT - m_t_pos_FT)
        m_w_cos_FT = einsum('tpq,tw->wpq', m_t_cos_FT, np.cos(gw.w[None, :] * gw.t[:, None]) * gw.wt_t_to_w_cos)
        m_w_sin_FT = einsum('tpq,tw->wpq', m_t_sin_FT, np.sin(gw.w[None, :] * gw.t[:, None]) * gw.wt_t_to_w_sin)
        m_w_FT = m_w_cos_FT + 1j * m_w_sin_FT
        max_error = np.max(np.abs(m_w - m_w_FT))
        logger.info(gw, '%d cycles error = %.6f', i + 1, max_error)
        if i == 0:
            first_cycle_error = max_error
        if gw.FT_error_stop is True and i == 0 and max_error > ERROR_THRESHOLD:
            logger.error(gw, 'large error detected, exit.')
    return first_cycle_error


class GWSpaceTime(GWAC):
    def __init__(self, mf, auxbasis=None):
        GWAC.__init__(self, mf, auxbasis=auxbasis)
        self.ngrid = 30  # number of minimax grids
        self.fast_diag = True  # if double the memory cost to speed up diagonal only calculation
        self.fix_ef = None  # fix Fermi level
        self.g0_t_pos_ao = None  # input Green's function in AO for t > 0
        self.g0_t_neg_ao = None  # input Green's function in AO for t < 0
        self.E_min_scale = 1.0  # scale factor for E_min in minimax grids
        self.E_max_scale = 1.0  # scale factor for E_max in minimax grids
        self.FT_cycle = 0  # number of cycles to test accuracy of Fourier transform
        self.FT_error_stop = False  # exit if large error found in Fourier transform

        # don't modify the following attributes, they are not input options
        self.E_min = None  # minimum gap for minimax grids
        self.E_max = None  # maximum gap for minimax grids
        self.t = None  # time grid
        self.t_wt = None  # weight of time grid
        self.w = None  # frequency grid
        self.w_wt = None  # weight of frequency grid
        self.wt_t_to_w_cos = None  # weight of cosine Fourier transform from time to frequency
        self.wt_t_to_w_sin = None  # weight of sine Fourier transform from time to frequency
        self.wt_w_to_t_cos = None  # weight of cosine Fourier transform from frequency to time
        self.wt_w_to_t_sin = None  # weight of sine Fourier transform from frequency to time
        self.sigma_t_pos = None  # self-energy on positive imaginary time
        self.sigma_t_neg = None  # self-energy on negative imaginary time
        self.sigma_w = None  # self-energy on positive imaginary frequency

        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        log.info('nocc = %-d, nvir = %-d', self.nocc, self.nmo - self.nocc)
        log.info('nmo = %-d', self.nmo)
        log.info('number of minimax grids = %-d', self.ngrid)
        log.info('use perturbative linearized QP equation = %s', self.qpe_linearized)
        log.info('analytic continuation method = %s', self.ac)
        log.info('off-diagonal self-energy = %s', self.fullsigma)
        if self.fullsigma is False:
            log.info('fast diagonal self-energy calculation = %s', self.fast_diag)
        if self.fix_ef is not None:
            log.info('fixed Fermi level = %.6f', self.fix_ef)
        if self.g0_t_pos_ao is not None and self.g0_t_neg_ao is not None:
            log.info('start with input G = %s', True)
        if self.FT_cycle > 0:
            log.info('FT accuracy test cycle = %d', self.FT_cycle)
            log.info('exit if FT error is large = %s', self.FT_error_stop)
        log.info('')
        return

    def kernel(self):
        """Run one-shot space-time GW calculation.
        This function is of O(N^4) scaling. No sparsity is used.
        Minimax grids were taken from CP2K.
        Reference
        [1] 10.1021/acs.jctc.0c01282
        [2] 10.1021/acs.jpclett.7b02740
        [3] 10.1103/PhysRevB.94.165109
        [4] 10.1103/PhysRevB.101.205145
        [5] 10.1021/ct5001268
        """
        if self.Lpq is None:
            # initialize three-index density fitting matrix in AO space
            self.initialize_df(auxbasis=self.auxbasis)
            with temporary_env(self.with_df, verbose=0), temporary_env(self.mol, verbose=0):
                self.Lpq = self.ao2mo(np.eye(self.nmo))

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)

        return

    def initialize_grid(self):
        """Initialize minimax grid.
        """
        if self.E_min is None:
            self.E_min = (self.mo_energy[self.nocc] - self.mo_energy[self.nocc - 1]) * self.E_min_scale
        if self.E_max is None:
            self.E_max = (self.mo_energy[-1] - self.mo_energy[0]) * self.E_min_scale
        assert self.E_max > self.E_min
        self.t, self.t_wt, self.w, self.w_wt = get_minimax_grid(k=self.ngrid, E_min=self.E_min, E_max=self.E_max)
        self.wt_t_to_w_cos, self.wt_t_to_w_sin, self.wt_w_to_t_cos, self.wt_w_to_t_sin = get_ft_weight(
            E_min=self.E_min, E_max=self.E_max, t=self.t, w=self.w)
        return

    def energy_tot(self):
        """Get GW correlation and total energy using Galitskii-Migdal formula on the imaginary time.
        See equation.13 in doi.org/10.1103/PhysRevB.101.205145

        Returns
        -------
        e_tot : double
            GW total energy
        e_hf : double
            HF total energy
        e_c : double
            GW correlation energy
        """
        g0_t_pos, g0_t_neg = get_g0_t(self.nocc, self._scf.mo_energy, self.t, ef=self.ef)
        Ec_pos = einsum('tij,tji,t->', g0_t_pos, self.sigma_t_neg, self.t_wt)
        Ec_neg = einsum('tij,tji,t->', g0_t_neg, self.sigma_t_pos, self.t_wt)
        e_c = (Ec_pos + Ec_neg).real

        dm = self._scf.make_rdm1()
        with temporary_env(self.mol, verbose=0):
            # keep this condition for embedding calculations
            if (not isinstance(self._scf, dft.rks.RKS)) and isinstance(self._scf, scf.hf.RHF):
                rhf = self._scf
            else:
                rhf = self._scf.to_hf()
            e_hf = rhf.energy_elec(dm=dm)[0] + self._scf.energy_nuc()
        e_tot = e_hf + e_c

        logger.info(self, 'HF energy@GW density  = %.8f', e_hf)
        logger.info(self, 'GW correlation energy = %.8f', e_c)
        logger.info(self, 'GW total energy       = %.8f', e_tot)

        return e_tot, e_hf, e_c

    def make_rdm1(self, mode='linear', ao_repr=False):
        r"""Get GW density matrix from G(it=0).
        The integration for G is separated into two parts:
        the Hartree-Fock part density matrix is simply diagonal (2, 2 ... 2, 0, 0 ...),
        the correlation part is Gc(it=0) = \int Gc(iw) dw, where the correlation GW function is
        Gc(iw) = G(iw) - GHF(iw) and GHF(iw) = 1.0 / (iw - ef + H_HF).
        See equation 2-8 in 10.1103/PhysRevB.98.155143.

        Parameters
        ----------
        mode : str, optional
            mode for Dyson equation, by default 'linear'
        ao_repr : bool, optional
            return density matrix in AO, by default False

        Returns
        -------
        rdm1 : double 2d array
            one-particle density matrix
        """
        with temporary_env(self._scf, verbose=0):
            vxc = self._scf.get_veff() - self._scf.get_j()
            vxc = self.mo_coeff.T @ vxc @ self.mo_coeff

        vk = self.get_sigma_exchange(mo_coeff=self.mo_coeff)

        g0_w = get_g0(omega=1j * self.w, mo_energy=self._scf.mo_energy - self.ef, eta=0)
        # NOTE: get_g0 puts frequency as the last index
        g0_w = g0_w.transpose(2, 0, 1)
        g_w = np.zeros_like(g0_w)
        for iw in range(self.ngrid):
            if mode == 'linear':
                g_w[iw] = g0_w[iw] + g0_w[iw] @ (self.sigma_w[iw] + vk - vxc) @ g0_w[iw]
            elif mode == 'dyson':
                g_w[iw] = np.linalg.inv(np.linalg.inv(g0_w[iw]) - self.sigma_w[iw] - vk + vxc)

        # get Hartree-Fock part Green's function, not the interacting Green's function
        # keep this condition for embedding calculations
        if (not isinstance(self._scf, dft.rks.RKS)) and isinstance(self._scf, scf.hf.RHF):
            rhf = self._scf
        else:
            rhf = scf.RHF(self.mol)
        fock = self._scf.mo_coeff.T @ rhf.get_fock(dm=self._scf.make_rdm1()) @ self._scf.mo_coeff
        ghf_w = np.zeros_like(g_w)
        # equation 4
        for iw in range(self.ngrid):
            ghf_w[iw] = np.linalg.inv((self.ef + 1j * self.w[iw]) * np.eye(self.nmo) - fock)
        # equation 3
        gc_w = g_w - ghf_w
        # equation 7
        rdm1 = 2.0 / np.pi * einsum('wij,w->ij', gc_w, self.w_wt).real
        rdm1[: self.nocc, : self.nocc] += np.eye(self.nocc) * 2.0
        rdm1 = 0.5 * (rdm1 + rdm1.T)
        logger.info(self, 'GW particle number = %s', np.trace(rdm1))

        if ao_repr is True:
            rdm1 = self._scf.mo_coeff @ rdm1 @ self._scf.mo_coeff.T

        return rdm1

    def make_gf(self, omega, eta=0.0, mode='dyson'):
        """Get G0W0 Green's function by AC fitting.

        Parameters
        ----------
        omega : complex 1d array
            frequency on which to evaluate the Green's function
        eta : double, optional
            broadening parameter. Defaults to 0.
        mode : str, optional
            mode for Dyson equation, 'linear' or 'dyson', by default 'dyson'

        Returns
        -------
        gf : complex 3d array
            GW Green's function
        gf0 : complex 3d array
            non-interacting Green's function
        sigma : complex 3d array
            self-energy
        """
        mo_energy = np.asarray(self._scf.mo_energy)
        gf0 = get_g0(omega, mo_energy, eta)
        # NOTE: get_g0 puts frequency as the last index
        gf0 = gf0.transpose(2, 0, 1)

        if self.fullsigma is True:
            sigma = self.acobj.ac_eval(omega + 1j * eta - self.ef, axis=0)
            sigma_diff = np.array(sigma, copy=True)
            sigma_diff += self.vk[None] - self.vxc[None]
        else:
            sigma = np.zeros_like(gf0)
            sigma_diff = np.zeros_like(gf0)
            for iw in range(len(omega)):
                for i in range(len(mo_energy)):
                    sigma[iw, i, i] = self.acobj[i].ac_eval(omega + 1j * eta - self.ef, axis=0)
                    sigma_diff[iw, i, i] = sigma[iw, i, i] + self.vk[i, i] - self.vxc[i, i]

        if mode == 'linear':
            gf = gf0 + (gf0 @ sigma_diff @ gf0)
        elif mode == 'dyson':
            gf = np.linalg.inv(np.linalg.inv(gf0) - sigma_diff)

        return gf, gf0, sigma

    FT_test_bosonic = FT_test_bosonic
    FT_test_fermionic = FT_test_fermionic

    def shift_sigma(self, w, E_min_scale=1.0, E_max_scale=1.0):
        """NOTE: experimental function.
        Get sigma(iw) on new grids by sigma(iw new) = FT[sigma(it old)].
        Accuracy can be tuned by E_min_scale and E_max_scale.

        Parameters
        ----------
        w : double 1d array
            _description_
        E_min_scale : double, optional
            scale factor of E_min, by default 1.0
        E_max_scale : double, optional
            scale factor of E_max, by default 1.0

        Returns
        -------
        sigma_w : complex 3d array
            sigma(iw) on new grids
        """
        assert len(w) == self.ngrid
        wt_t_to_w_cos, wt_t_to_w_sin = get_ft_weight(
            E_min=self.E_min * E_min_scale, E_max=self.E_max * E_max_scale, t=self.t, w=w, return_error=False
        )[:2]
        sigma_t_cos = 0.5 * (self.sigma_t_pos + self.sigma_t_neg)
        sigma_t_sin = 0.5 * (self.sigma_t_neg - self.sigma_t_pos)
        if self.fullsigma is True:
            sigma_w_cos = einsum('pqt,tw->pqw', sigma_t_cos, np.cos(w[None, :] * self.t[:, None]) * wt_t_to_w_cos)
            sigma_w_sin = einsum('pqt,tw->pqw', sigma_t_sin, np.sin(w[None, :] * self.t[:, None]) * wt_t_to_w_sin)
        else:
            sigma_w_cos = einsum('pt,tw->pw', sigma_t_cos, np.cos(w[None, :] * self.t[:, None]) * wt_t_to_w_cos)
            sigma_w_sin = einsum('pt,tw->pw', sigma_t_sin, np.sin(w[None, :] * self.t[:, None]) * wt_t_to_w_sin)
        sigma_w = sigma_w_cos + 1j * sigma_w_sin

        return sigma_w


if __name__ == '__main__':
    from pyscf import gto, dft, scf

    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    gw = GWSpaceTime(mf)
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.42657405) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16495551) < 1e-5

    # Galitskii-Migdal total energy
    gw = GWSpaceTime(mf)
    gw.fullsigma = True
    gw.FT_cycle = 5
    gw.kernel()
    e_tot, e_hf, e_c = gw.energy_tot()
    assert abs(gw.mo_energy[4] - -0.42657405) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16495551) < 1e-5
    assert abs(e_c - -0.49425105) < 1.0e-7

    print('passed tests!')
