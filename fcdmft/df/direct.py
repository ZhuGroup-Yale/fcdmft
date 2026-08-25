import numpy
import scipy.linalg
from pyscf import __config__
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from fcdmft.df.incore import _eig_decompose
from fcdmft.df.addons import make_auxmol
from fcdmft.df.outcore import _guess_shell_ranges

MAX_MEMORY = getattr(__config__, 'df_outcore_max_memory', 2000)  # 2GB
LINEAR_DEP_THR = getattr(__config__, 'df_df_DF_lindep', 1e-12)

#
# for auxe1 (P|ij)
#

def cholesky_eri_loop(mol, auxbasis='weigend+etb', auxmol=None,
                 int3c='int3c2e', aosym='s2ij', int2c='int2c2e', comp=1,
                 max_memory=MAX_MEMORY, decompose_j2c='cd',
                 lindep=LINEAR_DEP_THR, verbose=0):
    '''
    Returns:
        2D array of (naux,nao*(nao+1)/2) in C-contiguous
    '''
    assert comp == 1
    t0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, verbose)
    if auxmol is None:
        auxmol = make_auxmol(mol, auxbasis)

    j2c = auxmol.intor(int2c, hermi=1)
    if decompose_j2c == 'eig':
        low = _eig_decompose(mol, j2c, lindep)
    else:
        try:
            low = scipy.linalg.cholesky(j2c, lower=True)
            decompose_j2c = 'cd'
        except scipy.linalg.LinAlgError:
            low = _eig_decompose(mol, j2c, lindep)
            decompose_j2c = 'eig'
    j2c = None
    naux, naoaux = low.shape
    log.debug('size of aux basis %d', naux)
    log.timer_debug1('2c2e', *t0)

    int3c = gto.moleintor.ascint3(mol._add_suffix(int3c))
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = gto.moleintor.make_loc(bas, int3c)
    nao = ao_loc[mol.nbas]

    if aosym == 's1':
        nao_pair = nao * nao
    else:
        nao_pair = nao * (nao+1) // 2

    cderi = numpy.empty((naux, nao_pair))

    max_words = max_memory*.98e6/8 - low.size - cderi.size
    # Divide by 3 because scipy.linalg.solve may create a temporary copy for
    # ints and return another copy for results
    buflen = min(max(int(max_words/naoaux/comp/3), 8), nao_pair)
    shranges = _guess_shell_ranges(mol, buflen, aosym)
    log.debug1('shranges = %s', shranges)

    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    bufs1 = numpy.empty((comp*max(x[2] for x in shranges), naoaux))
    bufs2 = None

    p1 = 0
    for istep, sh_range in enumerate(shranges):
        log.debug('int3c2e [%d/%d], AO [%d:%d], nrow = %d',
                  istep+1, len(shranges), *sh_range)
        bstart, bend, nrow = sh_range
        shls_slice = (bstart, bend, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
        ints = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                       aosym, ao_loc, cintopt, out=bufs1)

        if ints.ndim == 3 and ints.flags.f_contiguous:
            ints = lib.transpose(ints.T, axes=(0,2,1), out=bufs2).reshape(naoaux,-1)
            if bufs2 is None:
                bufs2 = numpy.empty_like(bufs1)
            bufs1, bufs2 = bufs2, bufs1
        else:
            ints = ints.reshape((-1,naoaux)).T

        p0, p1 = p1, p1 + nrow
        indexes = (bstart, bend, nrow, p0, p1)
        if decompose_j2c == 'cd':
            if ints.flags.c_contiguous:
                trsm, = scipy.linalg.get_blas_funcs(('trsm',), (low, ints))
                dat = trsm(1.0, low, ints.T, lower=True, trans_a = 1, side = 1, overwrite_b=True).T
            else:
                dat = scipy.linalg.solve_triangular(low, ints, lower=True,
                                                   overwrite_b=True, check_finite=False)
            if dat.flags.f_contiguous:
                dat = lib.transpose(dat.T, out=bufs2)
            yield dat, indexes
        else:
            if bufs2 is None:
                bufs2 = numpy.empty_like(bufs1)
            dat = numpy.ndarray((naux, ints.shape[1]), buffer=bufs2)
            dat = lib.dot(low, ints, c=dat)
            yield dat, indexes
        dat = ints = None

    log.timer('cholesky_eri', *t0)


del MAX_MEMORY
