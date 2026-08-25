# The outcore routine contains code licensed under the following license:
#
# Copyright (c) 2020 Nathan Wycoff

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
import math
import numpy as np
import h5py
import scipy.linalg as sla

from pyscf import lib


class incore:
    def __init__(self, eri, err_tol=1e-6, aosym='s1'):
        """Constructor for incore pivoted cholesky

        Parameters
        ----------
        eri : ERI matrix
            s1 or s4 symmetry. For s1, either 4d or 2d is accepted.
        err_tol : float, optional
            threshold to stop pivoted cholesky.
            The default is 1e-5.
        aosym : str, optional
            eri symmetry, by default 's1'
        """
        self.eri = eri
        self.err_tol = err_tol
        self.aosym = aosym
        if aosym == 's1':
            if eri.ndim == 4:
                self.norb = eri.shape[0]
            elif eri.ndim == 2:
                self.norb = int(np.rint(np.sqrt(eri.shape[0])))
            self.eri = eri.reshape(self.norb**2, -1)
        elif aosym == 's4':
            npair = self.eri.shape[0]
            norb = math.floor(math.sqrt(1 + 8 * npair) / 2)
            self.norb = norb
        else:
            raise NotImplementedError

    def kernel(self, overwrite_input=False):
        """Fast pivoted cholesky using LAPACK

        Parameters
        ----------
        overwrite_input : bool, optional
            overwrites the input ERI for efficiency, by default False

        Returns
        -------
        np.ndarray
            (naux, norb, norb) array containing CDERI
        """

        erimat = self.eri

        pivoted_cholesky = sla.lapack.get_lapack_funcs('pstrf', (erimat,))
        # documentation for ?pstrf:
        # https://www.netlib.org/lapack/explore-html/de/de0/group__pstrf.html
        copy = sla.get_blas_funcs('copy', (erimat,))
        cderi_perm, piv, rank, info = pivoted_cholesky(erimat.T, tol=self.err_tol, lower=0, overwrite_a=overwrite_input)
        cderi = np.zeros((rank, erimat.shape[0]), dtype=erimat.dtype)
        invpiv = np.argsort(piv)
        cderi_ravel = cderi.reshape(-1)

        # Now we must permute the columns of the cholesky factor
        # In addition, the lower triangle is filled with garbage,
        # so we shouldn't copy that.
        # Only the first `rank` columns are needed.
        for j in range(erimat.shape[0]):
            i = invpiv[j]
            # the invocation of BLAS ?copy is equivalent to
            # Lpq[:min(rank, i+1), j] = Lpq_perm[:min(rank, i+1), i]
            copy(cderi_perm[: min(rank, i + 1), i], cderi_ravel, offy=j, incy=cderi.shape[1], n=min(rank, i + 1))
        cderi_perm = None
        if self.aosym == 's1':
            return cderi
        else:  # s4
            return lib.unpack_tril(cderi)


class outcore:
    def __init__(self, erifile, aosym='s4', dataname='eri_mo', err_tol=1e-6, max_rank=10000):
        """Constructor for outcore pivoted cholesky

        Parameters
        ----------
        erifile : str
            path to the h5 file containing the ERI
        err_tol : float, optional
            threshold to stop pivoted cholesky.
            The default is 1e-5.
        aosym : str, optional
            eri symmetry, by default 's1'.
            for s1, array format is (norb*norb, norb*norb)
            for s4, array format is (norb*(norb+1)/2, norb*(norb+1)/2)
        """
        self.erifile = erifile
        self.err_tol = err_tol
        self.max_rank = max_rank
        self.norb = None
        self.aosym = aosym
        self.dataname = dataname
        self.success = False
        self.nonposdef = False
        self.err = None
        assert aosym in ['s4', 's1']

    def kernel(self):
        """run outcore pivoted cholesky.

        Returns
        -------
        ndarray
            CDERI, shape (naux, norb*norb)
        """
        cderi = self.outcore_pivoted_cholesky(self.max_rank, self.err_tol)
        if self.aosym == 's1':
            return cderi
        else:  # s4
            return lib.unpack_tril(cderi)

    def outcore_pivoted_cholesky(self, max_rank, err_tol):
        """Computes the pivoted Cholesky decomposition/approximation of positive semi-definite operator.

        Parameters
        ----------
        max_rank : int
            The maximum rank of the approximate decomposition; an integer.
        err_tol : float
           The maximum error tolerance, that is difference between the approximate decomposition and true matrix, allowed.

        Returns
        -------
        ndarray
            The upper triangular cd matrix
        """
        f = h5py.File(self.erifile, 'r')
        dat = f[self.dataname]
        assert isinstance(dat, h5py.Dataset)
        npair = dat.shape[0]
        if self.aosym == 's4':
            self.norb = int(np.sqrt(1 + 8 * npair) / 2)
        elif self.aosym == 's1':
            self.norb = int(np.rint(np.sqrt(dat.shape[0])))
        assert dat.shape == (npair, npair)

        def get_row(dataset, i):
            return dataset[i, :]

        def get_diag(dataset):
            diagonal = np.asarray([dataset[i, i] for i in range(npair)])
            return diagonal

        d = np.ascontiguousarray(get_diag(dat).real)
        N = npair
        pi = np.arange(N, dtype=int)
        # Fortran-order array is fast for column-wise operations
        R = np.zeros([max_rank, N], dtype=dat.dtype, order='F')
        err = np.max(np.abs(d))

        swap = sla.get_blas_funcs('swap', (R,))

        m = 0
        while (m < max_rank) and (err > err_tol):
            # find the pivot
            i = m + np.argmax(d[pi[m:]])

            # swap columns i and m in R; update the permutation
            pi[i], pi[m] = pi[m], pi[i]
            swap(R[:, m], R[:, i])

            R[m, m] = np.sqrt(d[pi[m]])
            Apim = get_row(dat, pi[m])

            # slow loop that works
            # for j in range(m + 1, N):
            #     if m > 0:
            #         ip = np.inner(R[:m, m], R[:m, j].conj())
            #     else:
            #         ip = 0
            #     R[m, j] = (Apim[pi[j]] - ip) / R[m, m]
            #     d[pi[j]] -= R[m, j] ** 2

            # this is vectorized code that achieves the same thing
            if m > 0:
                Rj = R[:m, m + 1 :]  # vectors that are indexed by j during loop
                ip_vec = Rj.T @ R[:m, m].conj()
            else:
                ip_vec = 0

            # subtract the inner products
            R[m, m + 1 :] = (Apim[pi[(m + 1) :]] - ip_vec) / R[m, m]

            # update diagonal
            d[pi[(m + 1) :]] -= (R[m, m + 1 :] * R[m, m + 1 :].conj()).real

            err = np.max(d[pi[(m + 1) :]])
            m += 1

        # Undo all the column swapping in R by applying the
        # inverse permutation of pi
        R = np.ascontiguousarray(R[:m, np.argsort(pi)])

        if err > err_tol:
            print('Max diagonal error is not below tolerance but decomposition is max_rank!')
            print(f'Error: {err}, tolerance: {err_tol}')
            self.success = False
            self.err = err

        else:
            self.success = True

        # Check for non-positive definite matrix.
        # This causes regular Cholesky to fail, but we keep going.
        # If A is positive semidefinite, very small negative diagonal
        # elements may appear due to roundoff error accumulation.
        dmin = np.min(d)
        idx = np.argmin(d)
        if dmin < -np.finfo(float).eps * N:
            warnings.warn(f'Negative diagonal element {dmin}, orbital {idx} in Cholesky decomposition')
            self.nonposdef = True
        f.close()

        return R


cholesky = incore
