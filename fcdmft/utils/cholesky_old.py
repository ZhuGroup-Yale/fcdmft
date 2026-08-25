import numpy as np
import math

from pyscf.lib.numpy_helper import unpack_tril

LINEAR_DEP_THR = 1e-12


def ij2index(i, j):
    if i > j:
        ij = int((i + 1) * i // 2 + j)
    else:
        ij = int((j + 1) * j // 2 + i)
    return ij


def mat_loop_2d(ni, nj, i0=0, istep=1, j0=0, jstep=1):
    """Loop over the full 2d matrix"""

    for i in range(i0, ni, istep):
        for j in range(j0, nj, jstep):
            yield i, j


def get_eri_diag(eri, sym, nao):
    out = np.empty([nao, nao], dtype=np.double)
    if sym == 1:
        for mu in range(nao):
            for nu in range(nao):
                out[mu, nu] = eri[mu, nu, mu, nu]
    elif sym == 4:
        for mu in range(nao):
            for nu in range(mu + 1):
                mn = ij2index(mu, nu)
                out[mu, nu] = eri[mn, mn]
    elif sym == 8:
        for mu in range(nao):
            for nu in range(mu + 1):
                mn = ij2index(mu, nu)
                mnmn = ij2index(mn, mn)
                out[mu, nu] = eri[mnmn]
    else:
        raise ValueError

    return out


class AOPair:
    def __init__(self, mu, nu):
        self.mu = mu
        self.nu = nu
        self.Bmask = 0
        self.Dmask = 0
        self.eri_diag = -999999.0
        self.eri_off_ioff = 0
        self.Lpq = None


class AOPairs:
    def __init__(self, nao, sym):
        self.nao = nao
        self.n_aopairs = nao * nao
        self.sym = sym
        self.eri_diag = None
        self.sorted_ind = None
        self.data = None
        return

    def init_aopairs(self):
        ni = self.nao
        nj = self.nao

        self.data = np.empty([ni, nj], dtype=object)
        for i, j in mat_loop_2d(ni, nj):
            self.data[i, j] = AOPair(i, j)

    def get_eri_diag(self, eri):
        self.eri_diag = get_eri_diag(eri, self.sym, self.nao)
        for i, j in self.ijloop():
            ao_pair = self.get_aopair(i, j)
            ao_pair.eri_diag = self.eri_diag[i, j]

    def print_eri_diag(self):
        for i, j in self.ijloop():
            ao_pair = self.get_aopair(i, j)
            print(i, j, ao_pair.eri_diag)

    def get_aopair(self, i, j):
        return self.data[i, j]

    def ijloop(self):
        for i, j in mat_loop_2d(self.nao, self.nao):
            yield i, j

    def reorder_aopairs(self):
        max_diag_values = np.zeros([self.n_aopairs])
        ind = 0
        for i, j in self.ijloop():
            max_diag_values[ind] = self.eri_diag[i, j]
            ind += 1
        self.sorted_ind = np.argsort(-max_diag_values)

    def sorted_ijloop(self):
        if self.sorted_ind is None:
            self.reorder_aopairs()

        nj = self.nao
        for ind in self.sorted_ind:
            i = ind // nj
            j = ind % nj
            yield i, j

    def sorted_aopairs(self):
        for i, j in self.sorted_ijloop():
            yield self.get_aopair(i, j)

    def make_eri_offdiag_slow(self, eri, p_aopair_ind, q_aopair_ind):
        nij = p_aopair_ind.shape[0]
        nkl = q_aopair_ind.shape[0]
        out = np.empty([nij * nkl], dtype=np.double)
        ind = 0
        for ij in range(nij):
            i = p_aopair_ind[ij, 0]
            j = p_aopair_ind[ij, 1]
            for kl in range(nkl):
                k = q_aopair_ind[kl, 0]
                l = q_aopair_ind[kl, 1]
                out[ind] = eri[i, j, k, l]
                ind += 1

        return out, nij, nkl

    def make_eri_ao_aopair_slow(self, eri, kl_ind):
        nao = self.nao
        nao_r = len(kl_ind)
        nao_l = nao * nao
        out = np.empty([nao_l, nao_r], dtype=np.double)
        ind = 0
        eri = eri.reshape(nao_l, nao, nao)
        for kl in kl_ind:
            k = kl[0]
            l = kl[1]
            out[:, ind] = eri[:, k, l]
            ind += 1

        return out, nao_l, nao_r

    def make_eri_offdiag(self, eri, sym, pair_ind_p, pair_ind_q):
        # Get i, j, k, l indices directly from pair indices arrays
        i, j = pair_ind_p[:, 0], pair_ind_p[:, 1]
        k, l = pair_ind_q[:, 0], pair_ind_q[:, 1]
        nij = pair_ind_p.shape[0]
        nkl = pair_ind_q.shape[0]
        if sym == 1:
            # Use NumPy broadcasting to index eri and reshape the result into the desired output format
            out = eri[i[:, None], j[:, None], k, l].reshape(-1)
        elif sym == 4:
            ij = np.zeros(shape=[nij], dtype=int)
            for n in range(nij):
                ij[n] = ij2index(pair_ind_p[n, 0], pair_ind_p[n, 1])
            out = eri[ij, :]
            kl = np.zeros(shape=[nkl], dtype=int)
            for n in range(nkl):
                kl[n] = ij2index(pair_ind_q[n, 0], pair_ind_q[n, 1])
            out = out[:, kl].reshape(-1)
        elif sym == 8:
            ij = np.zeros(shape=[nij], dtype=int)
            for n in range(nij):
                ij[n] = ij2index(pair_ind_p[n, 0], pair_ind_p[n, 1])
            kl = np.zeros(shape=[nkl], dtype=int)
            for n in range(nkl):
                kl[n] = ij2index(pair_ind_q[n, 0], pair_ind_q[n, 1])
            ijkl = np.zeros(shape=[nij * nkl], dtype=int)
            for n1 in range(nij):
                for n2 in range(nkl):
                    ijkl[n1 * nkl + n2] = ij2index(ij[n1], kl[n2])
        else:
            raise ValueError

        return out, nij, nkl

    def make_eri_ao_aopair(self, eri, sym, kl_ind):
        nao = self.nao
        nao_r = len(kl_ind)
        nao_l = nao * nao
        out = np.empty([nao_l, nao_r], dtype=np.double)
        k, l = kl_ind[:, 0], kl_ind[:, 1]
        if sym == 1:
            out = eri.reshape(nao_l, nao, nao)[:, k, l]  # Shape will be (nao_l, nao_r)
        elif sym == 4:
            nkl = kl_ind.shape[0]
            kl = np.zeros(shape=[nkl], dtype=int)
            for n in range(nkl):
                kl[n] = ij2index(kl_ind[n, 0], kl_ind[n, 1])
            for n in range(nkl):
                out[:, n] = unpack_tril(eri[:, kl[n]]).reshape(-1)
        elif sym == 8:
            raise NotImplementedError
        else:
            raise ValueError
        return out, nao_l, nao_r


class cholesky():
    def __init__(self, eri, tau=1e-8, sigma=1e-2, dimQ=10, sym=1, nao=None):
        self.eri = eri
        self.tau = tau
        self.sigma = sigma
        self.dimQ = dimQ
        self.sym = sym
        if sym == 1:
            self.nao = self.eri.shape[0]
        else:
            assert nao is not None
            self.nao = nao
        self.ao_pairs = AOPairs(self.nao, self.sym)
        self.ao_pairs.init_aopairs()
        return

    def kernel(self):
        self.step1()
        return self.step2()

    def step1(self):
        # This is algorithm step (1) in https://pubs.aip.org/aip/jcp/article/139/13/134105/192851/General-implementation-of-the-resolution-of-the
        self.ao_pairs.get_eri_diag(self.eri)

        it = 0
        while True:
            it += 1
            if it > 200:
                print("decomposition is likely to fail!!!!")
                exit()

            Dmax = 0.0
            D = []
            Q = []
            nq = 0
            ioff = 0
            self.ao_pairs.sorted_ind = None

            for ind, ao_pair in enumerate(self.ao_pairs.sorted_aopairs()):
                max_diag = ao_pair.eri_diag
                if max_diag < self.tau:
                    ao_pair.Dmask = 0
                    ao_pair.Lpq = None
                    continue

                if ind == 0:
                    Dmax = max_diag

                i = ao_pair.mu
                j = ao_pair.nu

                ao_pair.Dmask = 1
                D.append((i, j))

                ao_pair.eri_off_ioff = ioff
                ioff += 1

                if nq < self.dimQ:
                    if ao_pair.eri_diag > self.sigma * Dmax:
                        ao_pair.Dmask = 2
                        Q.append((i, j))
                        nq += 1

            if nq == 0:
                break
            p_aopair_ind = np.asarray(D)
            q_aopair_ind = np.asarray(Q)

            eri_offdiag, nao_ij, nao_kl = self.ao_pairs.make_eri_offdiag(self.eri, self.sym, p_aopair_ind, q_aopair_ind)
            eri_offdiag = eri_offdiag.reshape((nao_ij, nao_kl))

            tmp = []
            for q in Q:
                i = q[0]
                j = q[1]
                ao_pair = self.ao_pairs.get_aopair(i,j)
                Dq = -999999.0
                if ao_pair.Dmask == 2:
                    Dq = ao_pair.eri_diag
                tmp.append((i, j, 0, Dq))

            tmp1 = np.asarray(tmp, dtype=object)[:, -1]
            sorted_q = np.argsort(-tmp1)

            ao_pairs_p = [self.ao_pairs.get_aopair(p[0], p[1]) for p in D]
            p_todo = [ao_pair.Lpq is not None for ao_pair in ao_pairs_p]
            ao_pairs_p = [p for p, td in zip(ao_pairs_p, p_todo) if td]
            eri_off_ioffs = [p.eri_off_ioff for p in ao_pairs_p]

            ao_pairs_q = [self.ao_pairs.get_aopair(q[0], q[1]) for q in tmp]
            index_q = list(range(len(ao_pairs_q)))
            q_todo = [not q[-1] < -9999 and ao_pair_q.Lpq is not None for ao_pair_q, q in zip(ao_pairs_q, tmp)]
            index_LL = [iq for iq, td in zip(index_q, q_todo) if td]
            ao_pairs_q = [q for q, td in zip(ao_pairs_q, q_todo) if td]

            if len(index_LL) != 0:
                L_pair_q = np.concatenate([ao_pair_q.Lpq for ao_pair_q in ao_pairs_q], axis=0)
                L_pair_p = np.concatenate([ao_pair_p.Lpq for ao_pair_p in ao_pairs_p], axis=0)

                LL = np.zeros([L_pair_p.shape[0], nao_kl], dtype=np.double)
                LL[:, index_LL] = np.matmul(L_pair_p, L_pair_q.T)
                eri_offdiag[eri_off_ioffs, :] -= LL

            Lpq = np.empty([nao_ij, nq], dtype=np.double)
            iq = 0

            for ind in sorted_q:
                q = tmp[ind]
                if q[-1] < -9999:
                    break

                Mpq = eri_offdiag[:, ind]
                Mqq = -1.0
                ao_pair = self.ao_pairs.get_aopair(q[0], q[1])
                ioff = ao_pair.eri_off_ioff
                q_left = ioff + q[2]
                Mqq = eri_offdiag[q_left, ind]

                Ltmp = np.zeros([nao_ij], dtype=np.double)
                if iq > 0:
                    Ltmp += np.sum(Lpq[:, :iq] * Lpq[q_left, :iq], axis=1)

                Lpq[:, iq] = (Mpq - Ltmp) / math.sqrt(Mqq)

                ao_pair.Dmask = 1  # remove q from Q
                ao_pair.Bmask = 1
                iq += 1

            for p in D:
                ao_pair = self.ao_pairs.get_aopair(p[0], p[1])
                ioff = ao_pair.eri_off_ioff
                ao_pair.eri_diag -= np.sum(Lpq[ioff, :iq] ** 2)
                size = 1
                if ao_pair.Lpq is None:
                    ao_pair.Lpq = Lpq[ioff : ioff + size, :].copy()
                else:
                    ao_pair.Lpq = np.append(ao_pair.Lpq, Lpq[ioff : ioff + size, :], axis=1)

    def step2(self):
        aopair_ind = []
        for i, j in self.ao_pairs.ijloop():
            ao_pair = self.ao_pairs.get_aopair(i, j)
            if ao_pair.Bmask == 1:
                aopair_ind.append((i, j))

        aopair_ind = np.asarray(aopair_ind)

        eri_S, nao_ij, nao_kl = self.ao_pairs.make_eri_offdiag(self.eri, self.sym, aopair_ind, aopair_ind)
        eri_S = eri_S.reshape(nao_ij, nao_kl)
        eri_S_ao = eri_S

        try:
            L = np.linalg.cholesky(eri_S_ao)
            tag = 'cd'
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(eri_S_ao)
            idx = w > LINEAR_DEP_THR
            L = v[:, idx] / np.sqrt(w[idx])
            tag = 'eig'

        if tag == 'cd':
            L = np.linalg.inv(L).T

        eri, nao_ij, nao_kl = self.ao_pairs.make_eri_ao_aopair(self.eri, self.sym, aopair_ind)
        eri = eri.reshape(nao_ij, nao_kl)
        cderi = np.dot(eri, L).T

        return cderi
