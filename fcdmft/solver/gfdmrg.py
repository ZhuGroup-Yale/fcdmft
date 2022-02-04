
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2020 Huanchen Zhai <hczhai@caltech.edu>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#

"""
DDMRG++ for Green's Function.
using pyscf and block2.

Original version:
     Huanchen Zhai, Nov 5, 2020
Revised: support for mpi
     Huanchen Zhai, Tianyu Zhu Mar 29, 2021
"""

from block2 import SU2, SZ, Global, OpNamesSet, Threading, ThreadingTypes
from block2 import init_memory, release_memory, SiteIndex, ConvergenceTypes
from block2 import VectorUInt8, PointGroup, FCIDUMP, QCTypes, SeqTypes, OpNames, Random
from block2 import VectorUBond, VectorDouble, NoiseTypes, DecompositionTypes, EquationTypes
from block2 import OrbitalOrdering, VectorUInt16, ExpectationAlgorithmTypes
import time
import numpy as np

# Set spin-adapted or non-spin-adapted here
SpinLabel = SU2
#SpinLabel = SZ

if SpinLabel == SU2:
    from block2.su2 import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.su2 import MPSInfo, MPS, MovingEnvironment, DMRG, Linear, IdentityMPO
    from block2.su2 import CASCIMPSInfo, MRCIMPSInfo, ParallelRuleNPDMQC
    from block2.su2 import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect
    from block2.su2 import VectorOpElement, LocalMPO, VectorMPS, VectorMovingEnvironment
    from block2.su2 import MPICommunicator, DeterminantTRIE, UnfusedMPS, PDM2MPOQC, ParallelRulePDM2QC, ParallelRulePDM1QC
else:
    from block2.sz import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.sz import MPSInfo, MPS, MovingEnvironment, DMRG, Linear, IdentityMPO
    from block2.sz import CASCIMPSInfo, MRCIMPSInfo, ParallelRuleNPDMQC
    from block2.sz import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect
    from block2.sz import VectorOpElement, LocalMPO, VectorMPS, VectorMovingEnvironment
    from block2.sz import MPICommunicator, DeterminantTRIE, UnfusedMPS, PDM2MPOQC, ParallelRulePDM2QC, ParallelRulePDM1QC

MPI = MPICommunicator()


def _print(*args, **kwargs):
    if MPI.rank == 0:
        print(*args, **kwargs)


class GFDMRGError(Exception):
    pass


def orbital_reorder(h1e, g2e, method='gaopt'):
    """
    Find an optimal ordering of orbitals for DMRG.
    Ref: J. Chem. Phys. 142, 034102 (2015)

    Args:
        method :
            'gaopt' - genetic algorithm, take several seconds
            'fiedler' - very fast, may be slightly worse than 'gaopt'

    Return a index array "midx":
        reordered_orb_sym = original_orb_sym[midx]
    """
    from pyscf import ao2mo
    if not isinstance(h1e, tuple):
        n_sites = h1e.shape[0]
        hmat = np.zeros((n_sites, n_sites))
        xmat = np.zeros((n_sites, n_sites))
        hmat[:] = np.abs(h1e[:])
        g2e = ao2mo.restore(1, g2e, n_sites)
        for i in range(0, n_sites):
            for j in range(0, n_sites):
                xmat[i, j] = abs(g2e[i, j, j, i])
    else:
        assert SpinLabel == SZ
        assert isinstance(h1e, tuple) and len(h1e) == 2
        assert isinstance(g2e, tuple) and len(g2e) == 3
        n_sites = h1e[0].shape[0]
        hmat = np.zeros((n_sites, n_sites))
        xmat = np.zeros((n_sites, n_sites))
        hmat[:] = 0.5 * np.abs(h1e[0][:]) + 0.5 * np.abs(h1e[1][:])
        g2eaa = ao2mo.restore(1, g2e[0], n_sites)
        g2ebb = ao2mo.restore(1, g2e[1], n_sites)
        g2eab = ao2mo.restore(1, g2e[2], n_sites)
        for i in range(0, n_sites):
            for j in range(0, n_sites):
                xmat[i, j] = 0.25 * abs(g2eaa[i, j, j, i]) \
                    + 0.25 * abs(g2ebb[i, j, j, i]) \
                    + 0.5 * abs(g2eab[i, j, j, i])
    kmat = VectorDouble((np.array(hmat) * 1E-7 + np.array(xmat)).flatten())
    if method == 'gaopt':
        n_tasks = 32
        opts = dict(
            n_generations=10000, n_configs=n_sites * 2,
            n_elite=8, clone_rate=0.1, mutate_rate=0.1
        )
        midx, mf = None, None
        for _ in range(0, n_tasks):
            idx = OrbitalOrdering.ga_opt(n_sites, kmat, **opts)
            f = OrbitalOrdering.evaluate(n_sites, kmat, idx)
            idx = np.array(idx)
            if mf is None or f < mf:
                midx, mf = idx, f
    elif method == 'fiedler':
        idx = OrbitalOrdering.fiedler(n_sites, kmat)
        midx = np.array(idx)
    else:
        midx = np.array(range(n_sites))
    return midx


def orbital_reorder_subspace(h1e, g2e, ncore, nvirt, method='gaopt'):
    """Find an optimal ordering of orbitals for DMRG-MRCI.
    Core and virtual orbitals will be kept at the beginning and the end."""
    midx = orbital_reorder(h1e, g2e, method=method)
    nact = h1e.shape[-1] - ncore - nvirt
    midx = np.concatenate((midx[midx < ncore],
        midx[(midx >= ncore) & (midx < nact + ncore)],
        midx[midx >= nact + ncore]), axis=0)
    return midx


class GFDMRG:
    """
    DDMRG++ for Green's Function for molecules.
    """

    def __init__(self, scratch='./nodex', memory=1 * 1E9, omp_threads=8, verbose=2,
                 print_statistics=True, mpi=None, dctr=True, dmain_ratio=0.4):
        """
        Memory is in bytes.
        verbose = 0 (quiet), 2 (per sweep), 3 (per iteration)
        """

        Random.rand_seed(0)
        isize = min(int(memory * 0.1), 200000000)
        init_memory(isize=isize, dsize=int(memory - isize), save_dir=scratch, dmain_ratio=dmain_ratio)
        Global.threading = Threading(
            ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, omp_threads, omp_threads, 1)
        Global.threading.seq_type = SeqTypes.Tasked
        Global.frame.load_buffering = False
        Global.frame.save_buffering = False
        Global.frame.use_main_stack = False
        Global.frame.minimal_disk_usage = True
        self.fcidump = None
        self.hamil = None
        self.verbose = verbose
        self.scratch = scratch
        self.mpo_orig = None
        self.print_statistics = print_statistics
        self.mpi = mpi
        self.dctr = dctr
        self.idx = None # reorder
        self.ridx = None # inv reorder
        self.dyn_corr_method = None
        self.restart_gs = False

        if self.verbose >= 2:
            _print(Global.frame)
            _print(Global.threading)

        if mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelRuleQC, ParallelRuleNPDMQC, ParallelRuleSiteQC
                from block2.su2 import ParallelRuleSiteQC, ParallelRuleIdentity
            else:
                from block2.sz import ParallelRuleQC, ParallelRuleNPDMQC
                from block2.sz import ParallelRuleSiteQC, ParallelRuleIdentity
            self.prule = ParallelRuleQC(mpi)
            self.pdm1rule = ParallelRulePDM1QC(mpi)
            self.pdm2rule = ParallelRulePDM2QC(mpi)
            self.npdmrule = ParallelRuleNPDMQC(mpi)
            self.siterule = ParallelRuleSiteQC(mpi)
            self.identrule = ParallelRuleIdentity(mpi)
        else:
            self.prule = None
            self.pdm1rule = None
            self.pdm2rule = None
            self.npdmrule = None
            self.siterule = None
            self.identrule = None

    def init_hamiltonian_fcidump(self, pg, filename, idx=None):
        """Read integrals from FCIDUMP file."""
        assert self.fcidump is None
        self.fcidump = FCIDUMP()
        self.fcidump.read(filename)
        if idx is not None:
            self.fcidump.reorder(VectorUInt16(idx))
            self.idx = idx
            self.ridx = np.argsort(idx)
        self.orb_sym = VectorUInt8(
            map(PointGroup.swap_d2h, self.fcidump.orb_sym))

        vacuum = SpinLabel(0)
        self.target = SpinLabel(self.fcidump.n_elec, self.fcidump.twos,
                                PointGroup.swap_d2h(self.fcidump.isym))
        self.n_sites = self.fcidump.n_sites

        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)
        assert pg in ["d2h", "c1"]

    def init_hamiltonian(self, pg, n_sites, n_elec, twos, isym, orb_sym,
                         e_core, h1e, g2e, tol=1E-13, idx=None,
                         save_fcidump=None):
        """Initialize integrals using h1e, g2e, etc."""
        assert self.fcidump is None
        self.fcidump = FCIDUMP()
        if not isinstance(h1e, tuple):
            mh1e = np.zeros((n_sites * (n_sites + 1) // 2))
            k = 0
            for i in range(0, n_sites):
                for j in range(0, i + 1):
                    assert abs(h1e[i, j] - h1e[j, i]) < tol
                    mh1e[k] = h1e[i, j]
                    k += 1
            mg2e = g2e.flatten()
            mh1e[np.abs(mh1e) < tol] = 0.0
            mg2e[np.abs(mg2e) < tol] = 0.0
            self.fcidump.initialize_su2(
                n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
        else:
            assert SpinLabel == SZ
            assert isinstance(h1e, tuple) and len(h1e) == 2
            assert isinstance(g2e, tuple) and len(g2e) == 3
            mh1e_a = np.zeros((n_sites * (n_sites + 1) // 2))
            mh1e_b = np.zeros((n_sites * (n_sites + 1) // 2))
            mh1e = (mh1e_a, mh1e_b)
            for xmh1e, xh1e in zip(mh1e, h1e):
                k = 0
                for i in range(0, n_sites):
                    for j in range(0, i + 1):
                        assert abs(xh1e[i, j] - xh1e[j, i]) < tol
                        xmh1e[k] = xh1e[i, j]
                        k += 1
                xmh1e[np.abs(xmh1e) < tol] = 0.0
            mg2e = tuple(xg2e.flatten() for xg2e in g2e)
            for xmg2e in mg2e:
                xmg2e[np.abs(xmg2e) < tol] = 0.0
            self.fcidump.initialize_sz(
                n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
        self.fcidump.orb_sym = VectorUInt8(orb_sym)
        if idx is not None:
            self.fcidump.reorder(VectorUInt16(idx))
            self.idx = idx
            self.ridx = np.argsort(idx)
        self.orb_sym = VectorUInt8(
            map(PointGroup.swap_d2h, self.fcidump.orb_sym))

        vacuum = SpinLabel(0)
        self.target = SpinLabel(n_elec, twos, PointGroup.swap_d2h(isym))
        self.n_sites = n_sites

        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)

        if save_fcidump is not None:
            self.fcidump.orb_sym = VectorUInt8(orb_sym)
            self.fcidump.write(save_fcidump)
        assert pg in ["d2h", "c1"]

    @staticmethod
    def fmt_size(i, suffix='B'):
        if i < 1000:
            return "%d %s" % (i, suffix)
        else:
            a = 1024
            for pf in "KMGTPEZY":
                p = 2
                for k in [10, 100, 1000]:
                    if i < k * a:
                        return "%%.%df %%s%%s" % p % (i / a, pf, suffix)
                    p -= 1
                a *= 1024
        return "??? " + suffix

    def dmrg(self, bond_dims, noises, n_steps=30, conv_tol=1E-7, cutoff=1E-14, occs=None, bias=1.0):
        """Ground-State DMRG."""

        if self.verbose >= 2:
            _print('>>> START GS-DMRG <<<')
        t = time.perf_counter()

        if self.mpi is not None:
            self.mpi.barrier()

        if not self.restart_gs:
            if self.dyn_corr_method is None or self.dyn_corr_method.startswith("gfmrci"):
                mps_info = MPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target, self.hamil.basis)
            elif self.dyn_corr_method == "casci":
                mps_info = CASCIMPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target, self.hamil.basis,
                                self.ncore, self.n_sites - self.ncore - self.nvirt, self.nvirt)
            elif self.dyn_corr_method.startswith("mrci"):
                ci_order = len(self.dyn_corr_method) - 4
                assert "sdtqph"[:ci_order] == self.dyn_corr_method[4:]
                mps_info = MRCIMPSInfo(self.n_sites, self.ncore, self.nvirt, ci_order,
                                self.hamil.vacuum, self.target, self.hamil.basis)
            else:
                raise RuntimeError("Unsupported dyn corr method %s" % self.dyn_corr_method)
            mps_info.tag = 'KET'
            if occs is None:
                if self.verbose >= 2:
                    _print("Using FCI INIT MPS")
                mps_info.set_bond_dimension(bond_dims[0])
            else:
                if self.verbose >= 2:
                    _print("Using occupation number INIT MPS")
                if self.idx is not None:
                    occs = self.fcidump.reorder(VectorDouble(occs), VectorUInt16(self.idx))
                mps_info.set_bond_dimension_using_occ(
                    bond_dims[0], VectorDouble(occs), bias=bias)
            mps = MPS(self.n_sites, 0, 2)
            mps.initialize(mps_info)
            mps.random_canonicalize()

            mps.save_mutable()
            mps.deallocate()
            mps_info.save_mutable()
            mps_info.deallocate_mutable()
        else:
            _print("Loading saved MPS")
            mps_info = MPSInfo(0)
            mps_info.load_data(self.scratch + "/GS_MPS_INFO")
            mps = MPS(mps_info)
            mps.load_data()

        # MPO
        tx = time.perf_counter()
        mpo = MPOQC(self.hamil, QCTypes.Conventional)
        mpo = SimplifiedMPO(mpo, RuleQC(), True, True,
                            OpNamesSet((OpNames.R, OpNames.RD)))
        self.mpo_orig = mpo

        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO
            mpo = ParallelMPO(mpo, self.prule)

        if self.verbose >= 3:
            _print('MPO time = ', time.perf_counter() - tx)

        if self.print_statistics:
            _print('GS MPO BOND DIMS = ', ''.join(
                ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
            max_d = max(bond_dims)
            mps_info2 = MPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target, self.hamil.basis)
            mps_info2.set_bond_dimension(max_d)
            _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
            _print("GS EST MAX MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            _print("GS EST PEAK MEM = ", GFDMRG.fmt_size(
                mem2), " SCRATCH = ", GFDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

        # DMRG
        me = MovingEnvironment(mpo, mps, mps, "DMRG")
        if self.dctr:
            me.delayed_contraction = OpNamesSet.normal_ops()
            me.cached_contraction = True
        tx = time.perf_counter()
        me.init_environments(self.verbose >= 4)
        if self.verbose >= 3:
            _print('DMRG INIT time = ', time.perf_counter() - tx)
        dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
        dmrg.davidson_soft_max_iter = 4000
        dmrg.noise_type = NoiseTypes.ReducedPerturbative
        dmrg.decomp_type = DecompositionTypes.SVD
        dmrg.iprint = max(self.verbose - 1, 0)
        dmrg.cutoff = cutoff
        dmrg.solve(n_steps, mps.center == 0, conv_tol)

        self.gs_energy = dmrg.energies[-1][0]
        self.bond_dim = bond_dims[-1]

        mps.save_data()
        mps_info.save_data(self.scratch + "/GS_MPS_INFO")
        mps_info.deallocate()

        if self.print_statistics:
            dmain, dseco, imain, iseco = Global.frame.peak_used_memory
            _print("GS PEAK MEM USAGE:",
                   "DMEM = ", GFDMRG.fmt_size(dmain + dseco),
                   "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
                   "IMEM = ", GFDMRG.fmt_size(imain + iseco),
                   "(%.0f%%)" % (imain * 100 / (imain + iseco)))

        if self.verbose >= 1:
            _print("=== GS Energy = %20.15f" % self.gs_energy)

        if self.verbose >= 2:
            _print('>>> COMPLETE GS-DMRG | Time = %.2f <<<' %
                   (time.perf_counter() - t))

    # one-particle density matrix
    # return value:
    #     pdm[0, :, :] -> <AD_{i,alpha} A_{j,alpha}>
    #     pdm[1, :, :] -> < AD_{i,beta}  A_{j,beta}>
    def get_one_pdm(self, algo_type='auto', site_type=1):
        if self.verbose >= 2:
            _print('>>> START one-pdm .. algo = %s site = %d <<<' % (algo_type, site_type))
        t = time.perf_counter()

        if self.mpi is not None:
            self.mpi.barrier()

        mps_info = MPSInfo(0)
        mps_info.load_data(self.scratch + "/GS_MPS_INFO")
        mps = MPS(mps_info)
        mps.load_data()

        mps = mps.deep_copy('PDM-TMP')
        mps_info = mps.info
        assert mps.dot == 2
        if site_type != 2:
            mps.dot = 1
            if mps.center == mps.n_sites - 2:
                mps.center = mps.n_sites - 1
                mps.canonical_form = mps.canonical_form[:-1] + 'S'
            elif mps.center == 0:
                mps.canonical_form = 'K' + mps.canonical_form[1:]
            else:
                assert False

        if self.mpi is not None:
            self.mpi.barrier()

        mps.info.load_mutable()
        max_bdim = max([x.n_states_total for x in mps.info.left_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim
        max_bdim = max([x.n_states_total for x in mps.info.right_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim

        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO
        
        if algo_type == "auto":
            expt_algo_type = ExpectationAlgorithmTypes.Automatic
        elif algo_type == "fast":
            expt_algo_type = ExpectationAlgorithmTypes.Fast
        else:
            expt_algo_type = ExpectationAlgorithmTypes.Normal

        # 1PDM MPO
        pmpo = PDM1MPOQC(self.hamil)
        pmpo = SimplifiedMPO(pmpo, RuleQC())

        if self.mpi is not None:
            if expt_algo_type == ExpectationAlgorithmTypes.Normal:
                pmpo = ParallelMPO(pmpo, self.npdmrule)
            else:
                pmpo = ParallelMPO(pmpo, self.pdm1rule)

        # 1PDM
        pme = MovingEnvironment(pmpo, mps, mps, "1PDM")
        if expt_algo_type == ExpectationAlgorithmTypes.Normal:
            pme.delayed_contraction = OpNamesSet.normal_ops()
        pme.init_environments(False)
        expect = Expect(pme, mps.info.bond_dim, mps.info.bond_dim)
        expect.algo_type = expt_algo_type
        if site_type == 0:
            expect.zero_dot_algo = True
        expect.iprint = max(self.verbose - 1, 0)
        expect.solve(True, mps.center == 0)
        if SpinLabel == SU2:
            dmr = expect.get_1pdm_spatial(self.n_sites)
            dm = np.array(dmr).copy()
        else:
            dmr = expect.get_1pdm(self.n_sites)
            dm = np.array(dmr).copy()
            dm = dm.reshape((self.n_sites, 2,
                             self.n_sites, 2))
            dm = np.transpose(dm, (0, 2, 1, 3))

        if self.ridx is not None:
            dm[:, :] = dm[self.ridx, :][:, self.ridx]

        mps.save_data()
        mps_info.deallocate()
        dmr.deallocate()
        pmpo.deallocate()

        if self.verbose >= 2:
            _print('>>> COMPLETE one-pdm | Time = %.2f <<<' %
                   (time.perf_counter() - t))

        if SpinLabel == SU2:
            return np.concatenate([dm[None, :, :], dm[None, :, :]], axis=0) / 2
        else:
            return np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)

    # two-particle density matrix
    # return value:
    #     pdm[0, i, j, k, l] -> <AD_{i,alpha} AD_{j,alpha} A_{k,alpha} A_{l,alpha}>
    #     pdm[1, :, :, :, :] -> <AD_{i,alpha} AD_{j,beta} A_{k,beta} A_{l,alpha}>
    #     pdm[2, :, :, :, :] -> <AD_{i,beta} AD_{j,beta} A_{k,beta} A_{l,beta}>
    def get_two_pdm(self, algo_type='auto', site_type=1):
        if self.verbose >= 2:
            _print('>>> START two-pdm .. algo = %s site = %d <<<' % (algo_type, site_type))
        t = time.perf_counter()

        if self.mpi is not None:
            self.mpi.barrier()

        mps_info = MPSInfo(0)
        mps_info.load_data(self.scratch + "/GS_MPS_INFO")
        mps = MPS(mps_info)
        mps.load_data()

        mps = mps.deep_copy('PDM-TMP')
        mps_info = mps.info
        assert mps.dot == 2
        if site_type != 2:
            mps.dot = 1
            if mps.center == mps.n_sites - 2:
                mps.center = mps.n_sites - 1
                mps.canonical_form = mps.canonical_form[:-1] + 'S'
            elif mps.center == 0:
                mps.canonical_form = 'K' + mps.canonical_form[1:]
            else:
                assert False

        if self.mpi is not None:
            self.mpi.barrier()

        mps.info.load_mutable()
        max_bdim = max([x.n_states_total for x in mps.info.left_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim
        max_bdim = max([x.n_states_total for x in mps.info.right_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim

        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO
        
        if algo_type == "auto":
            expt_algo_type = ExpectationAlgorithmTypes.Automatic
        elif algo_type == "fast":
            expt_algo_type = ExpectationAlgorithmTypes.Fast
        else:
            expt_algo_type = ExpectationAlgorithmTypes.Normal

        # 2PDM MPO
        pmpo = PDM2MPOQC(self.hamil)
        pmpo = SimplifiedMPO(pmpo, RuleQC())

        if self.mpi is not None:
            if expt_algo_type == ExpectationAlgorithmTypes.Normal:
                pmpo = ParallelMPO(pmpo, self.npdmrule)
            else:
                pmpo = ParallelMPO(pmpo, self.pdm2rule)

        # 2PDM
        pme = MovingEnvironment(pmpo, mps, mps, "2PDM")
        if expt_algo_type == ExpectationAlgorithmTypes.Normal:
            pme.delayed_contraction = OpNamesSet.normal_ops()
        pme.init_environments(False)
        expect = Expect(pme, mps.info.bond_dim, mps.info.bond_dim)
        expect.algo_type = expt_algo_type
        if site_type == 0:
            expect.zero_dot_algo = True
        expect.iprint = max(self.verbose - 1, 0)
        expect.solve(True, mps.center == 0)

        dmr = expect.get_2pdm(self.n_sites)
        dm = np.array(dmr, copy=True)
        dm = dm.reshape((self.n_sites, 2, self.n_sites, 2, self.n_sites, 2, self.n_sites, 2))
        dm = np.transpose(dm, (0, 2, 4, 6, 1, 3, 5, 7))
        dm = np.concatenate([dm[None, :, :, :, :, 0, 0, 0, 0], dm[None, :, :, :, :, 0, 1, 1, 0],
                                dm[None, :, :, :, :, 1, 1, 1, 1]], axis=0)

        if self.ridx is not None:
            dm[:, :, :, :, :] = dm[:, self.ridx, :, :, :][:, :, self.ridx,
                :, :][:, :, :, self.ridx, :][:, :, :, :, self.ridx]

        mps.save_data()
        mps_info.deallocate()
        pmpo.deallocate()

        if self.verbose >= 2:
            _print('>>> COMPLETE two-pdm | Time = %.2f <<<' %
                   (time.perf_counter() - t))

        return dm

    def save_gs_mps(self, save_dir='./gs_mps'):
        import shutil
        import pickle
        import os
        if self.mpi.rank == 0:
            pickle.dump(self.gs_energy, open(
                self.scratch + '/GS_ENERGY', 'wb'))
            for k in os.listdir(self.scratch):
                if '.KET.' in k or k == 'GS_MPS_INFO' or k == 'GS_ENERGY':
                    shutil.copy(self.scratch + "/" + k, save_dir + "/" + k)
        self.mpi.barrier()

    def load_gs_mps(self, load_dir='./gs_mps'):
        import shutil
        import pickle
        import os
        if self.mpi.rank == 0:
            for k in os.listdir(load_dir):
                shutil.copy(load_dir + "/" + k, self.scratch + "/" + k)
        self.mpi.barrier()
        self.gs_energy = pickle.load(open(self.scratch + '/GS_ENERGY', 'rb'))

    def greens_function(self, bond_dims, noises, gmres_tol, conv_tol, n_steps,
                        cps_bond_dims, cps_noises, cps_conv_tol, cps_n_steps, idxs,
                        eta, freqs, addition, extra_freqs=None, extra_eta=None,
                        cutoff=1E-14, diag_only=False,
                        n_off_diag_cg=0, alpha=True, occs=None, bias=1.0, mo_coeff=None):
        """Green's function."""
        ops = [None] * len(idxs)
        rkets = [None] * len(idxs)
        rmpos = [None] * len(idxs)

        if self.mpi is not None:
            self.mpi.barrier()

        if self.mpo_orig is None:
            mpo = MPOQC(self.hamil, QCTypes.Conventional)
            mpo = SimplifiedMPO(mpo, RuleQC(), True, True,
                                OpNamesSet((OpNames.R, OpNames.RD)))
            self.mpo_orig = mpo

        mps_info = MPSInfo(0)
        mps_info.load_data(self.scratch + "/GS_MPS_INFO")
        mps = MPS(mps_info)
        mps.load_data()

        mps.info.load_mutable()
        max_bdim = max([x.n_states_total for x in mps.info.left_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim
        max_bdim = max([x.n_states_total for x in mps.info.right_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim

        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO

        if addition:
            mpo = -1.0 * self.mpo_orig
            mpo.const_e += self.gs_energy
        else:
            mpo = 1.0 * self.mpo_orig
            mpo.const_e -= self.gs_energy

        if self.mpi is not None:
            mpo = ParallelMPO(mpo, self.prule)

        if self.print_statistics:
            _print('GF MPO BOND DIMS = ', ''.join(
                ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
            max_d = max(bond_dims)
            mps_info2 = MPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target, self.hamil.basis)
            mps_info2.set_bond_dimension(max_d)
            _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
            _print("GF EST MAX MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            _print("GF EST PEAK MEM = ", GFDMRG.fmt_size(
                mem2), " SCRATCH = ", GFDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

        impo = SimplifiedMPO(IdentityMPO(self.hamil),
                             NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

        if self.mpi is not None:
            impo = ParallelMPO(impo, self.identrule)

        def align_mps_center(ket, ref):
            if self.mpi is not None:
                self.mpi.barrier()
            cf = ket.canonical_form
            if ref.center == 0:
                ket.center += 1
                ket.canonical_form = ket.canonical_form[:-1] + 'S'
                while ket.center != 0:
                    ket.move_left(mpo.tf.opf.cg, self.prule)
            else:
                ket.canonical_form = 'K' + ket.canonical_form[1:]
                while ket.center != ket.n_sites - 1:
                    ket.move_right(mpo.tf.opf.cg, self.prule)
                ket.center -= 1
            if self.verbose >= 2:
                _print('CF = %s --> %s' % (cf, ket.canonical_form))
                if self.mpi is not None:
                    self.mpi.barrier()
                ket.info.load_mutable()
                _print("CF left  bond dim = ", ["%6d" % x.n_states_total for x in ket.info.left_dims])
                _print("CF right bond dim = ", ["%6d" % x.n_states_total for x in ket.info.right_dims])
                _print("CF final = ", ket.info.bond_dim)
                ket.info.deallocate_mutable()

        if mo_coeff is None:
            if self.ridx is not None:
                gidxs = self.ridx[np.array(idxs)]
            else:
                gidxs = idxs
        else:
            if self.idx is not None:
                mo_coeff = mo_coeff[:, self.idx]
            gidxs = list(range(self.n_sites))
            ops = [None] * self.n_sites
            print('idxs = ', idxs, 'gidxs = ', gidxs)

        for ii, idx in enumerate(gidxs):
            if SpinLabel == SZ:
                if addition:
                    ops[ii] = OpElement(OpNames.C, SiteIndex(
                        (idx, ), (0 if alpha else 1, )), SZ(1, 1 if alpha else -1, self.orb_sym[idx]))
                else:
                    ops[ii] = OpElement(OpNames.D, SiteIndex(
                        (idx, ), (0 if alpha else 1, )), SZ(-1, -1 if alpha else 1, self.orb_sym[idx]))
            else:
                if addition:
                    ops[ii] = OpElement(OpNames.C, SiteIndex(
                        (idx, ), ()), SU2(1, 1, self.orb_sym[idx]))
                else:
                    ops[ii] = OpElement(OpNames.D, SiteIndex(
                        (idx, ), ()), SU2(-1, 1, self.orb_sym[idx]))

        for ii, idx in enumerate(idxs):
            if self.mpi is not None:
                self.mpi.barrier()
            if self.verbose >= 2:
                _print('>>> START Compression Site = %4d <<<' % idx)
            t = time.perf_counter()

            if self.dyn_corr_method is None:
                rket_info = MPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target + ops[ii].q_label, self.hamil.basis)
            elif self.dyn_corr_method == "casci":
                rket_info = CASCIMPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target + ops[ii].q_label, self.hamil.basis,
                                self.ncore, self.n_sites - self.ncore - self.nvirt, self.nvirt)
            elif self.dyn_corr_method.startswith("mrci"):
                ci_order = len(self.dyn_corr_method) - 4
                assert "sdtqph"[:ci_order] == self.dyn_corr_method[4:]
                rket_info = MRCIMPSInfo(self.n_sites, self.ncore, self.nvirt, ci_order,
                                self.hamil.vacuum, self.target + ops[ii].q_label, self.hamil.basis)
            elif self.dyn_corr_method.startswith("gfmrci"):
                ci_order = len(self.dyn_corr_method) - 6
                assert "sdtqph"[:ci_order] == self.dyn_corr_method[6:]
                rket_info = MRCIMPSInfo(self.n_sites, self.ncore, self.nvirt, ci_order,
                                self.hamil.vacuum, self.target + ops[ii].q_label, self.hamil.basis)
            else:
                raise RuntimeError("Unsupported dyn corr method %s" % self.dyn_corr_method)
            rket_info.tag = 'DKET%d' % idx
            rket_info.set_bond_dimension(mps.info.bond_dim)
            if occs is None:
                if self.verbose >= 2:
                    _print("Using FCI INIT MPS")
                rket_info.set_bond_dimension(mps.info.bond_dim)
            else:
                if self.verbose >= 2:
                    _print("Using occupation number INIT MPS")
                rket_info.set_bond_dimension_using_occ(
                    mps.info.bond_dim, VectorDouble(occs), bias=bias)
            rkets[ii] = MPS(self.n_sites, mps.center, 2)
            rkets[ii].initialize(rket_info)
            rkets[ii].random_canonicalize()
            rkets[ii].info.bond_dim = cps_bond_dims[-1]

            rkets[ii].save_mutable()
            rkets[ii].deallocate()
            rket_info.save_mutable()
            rket_info.deallocate_mutable()
            if self.mpi is None or self.mpi.rank == 0:
                rket_info.save_data(self.scratch + "/DK%d_MPS_INFO" % ii)

            if mo_coeff is None:
                # the mpo and gf are in the same basis
                # the mpo is SiteMPO
                rmpos[ii] = SimplifiedMPO(
                    SiteMPO(self.hamil, ops[ii]), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
            else:
                # the mpo is in mo basis and gf is in ao basis
                # the mpo is sum of SiteMPO (LocalMPO)
                ao_ops = VectorOpElement([None] * self.n_sites)
                for ix in range(self.n_sites):
                    ao_ops[ix] = ops[ix] * mo_coeff[idx, ix]
                rmpos[ii] = SimplifiedMPO(
                    LocalMPO(self.hamil, ao_ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

            if self.mpi is not None:
                rmpos[ii] = ParallelMPO(rmpos[ii], self.siterule)

            if len(cps_noises) == 1 and cps_noises[0] == 0:
                pme = None
            else:
                pme = MovingEnvironment(mpo, rkets[ii], rkets[ii], "PERT")
                pme.init_environments(False)
            rme = MovingEnvironment(rmpos[ii], rkets[ii], mps, "RHS")
            rme.init_environments(False)
            if self.dctr:
                if pme is not None:
                    pme.delayed_contraction = OpNamesSet.normal_ops()
                rme.delayed_contraction = OpNamesSet.normal_ops()

            cps = Linear(pme, rme, VectorUBond(cps_bond_dims),
                         VectorUBond([mps.info.bond_dim]), VectorDouble(cps_noises))
            cps.noise_type = NoiseTypes.ReducedPerturbative
            cps.decomp_type = DecompositionTypes.SVD
            if pme is not None:
                cps.eq_type = EquationTypes.PerturbativeCompression
            cps.iprint = max(self.verbose - 1, 0)
            cps.cutoff = cutoff
            cps.solve(cps_n_steps, mps.center == 0, cps_conv_tol)

            if self.verbose >= 2:
                _print('>>> COMPLETE Compression Site = %4d | Time = %.2f <<<' %
                       (idx, time.perf_counter() - t))

        gf_mat = np.zeros((len(idxs), len(idxs), len(freqs)), dtype=complex)
        if extra_freqs is None:
            extra_freqs = [[]] * len(freqs)
        else:
            assert len(extra_freqs) == len(freqs)
        ex_gf_mats = [np.zeros((len(idxs), len(idxs), len(ex_fs)), dtype=complex)
            for ex_fs in extra_freqs]

        def debug_info(info):
            mrank = 0 if self.mpi is None else self.mpi.rank
            msize = 1 if self.mpi is None else self.mpi.size
            import psutil, os
            mem = psutil.Process(os.getpid()).memory_info().rss
            for ir in range(msize):
                if self.mpi is not None:
                    self.mpi.barrier()
                if ir == mrank:
                    print('mpi rank = %d mem = %s .. %s' % (ir, GFDMRG.fmt_size(mem), info), flush=True)
                if self.mpi is not None:
                    self.mpi.barrier()

        for ii, idx in enumerate(idxs):

            debug_info("DEBUG A %d" % ii)
            if rkets[ii].center != mps.center:
                debug_info("DEBUG B %d" % ii)
                align_mps_center(rkets[ii], mps)
            debug_info("DEBUG C %d" % ii)
            lme = MovingEnvironment(mpo, rkets[ii], rkets[ii], "LHS")
            lme.init_environments(True)
            rme = MovingEnvironment(rmpos[ii], rkets[ii], mps, "RHS")
            rme.init_environments(True)
            if self.dctr:
                lme.delayed_contraction = OpNamesSet.normal_ops()
                rme.delayed_contraction = OpNamesSet.normal_ops()
            debug_info("DEBUG D %d" % ii)

            linear = Linear(lme, rme, VectorUBond(bond_dims),
                            VectorUBond(cps_bond_dims[-1:]), VectorDouble(noises))
            linear.gf_eta = eta
            linear.gf_extra_eta = 0 if extra_eta is None else extra_eta
            linear.linear_conv_thrds = VectorDouble([gmres_tol] * n_steps)
            linear.noise_type = NoiseTypes.ReducedPerturbative
            linear.decomp_type = DecompositionTypes.SVD
            # NOTE: Not raising error even if CG is not converged
            max_cg_iter = 1000
            linear.linear_soft_max_iter = max_cg_iter
            linear.linear_max_iter = max_cg_iter + 100
            linear.target_bra_bond_dim = cps_bond_dims[-1]
            linear.target_ket_bond_dim = cps_bond_dims[-1]
            linear.eq_type = EquationTypes.GreensFunction
            linear.linear_use_precondition = False
            linear.linear_solver_params = (50, -1)
            linear.iprint = max(self.verbose - 1, 0)
            linear.cutoff = cutoff

            debug_info("DEBUG E %d" % ii)

            for iw, (w, xw) in enumerate(zip(freqs, extra_freqs)):

                if self.verbose >= 2:
                    _print('>>>   START  GF OMEGA = %10.5f Site = %4d %4d <<<' %
                           (w, idx, idx))
                t = time.perf_counter()

                linear.tme = None
                linear.ext_mpss = VectorMPS([])
                linear.ext_tmes = VectorMovingEnvironment([])
                linear.noises[0] = noises[0]
                linear.linear_soft_max_iter = max_cg_iter
                linear.noises = VectorDouble(noises)
                linear.bra_bond_dims = VectorUBond(bond_dims)
                linear.gf_omega = w
                linear.gf_extra_omegas_at_site = -1
                linear.conv_type = ConvergenceTypes.MiddleSite
                # consecutive sweeps for determining conv
                linear.conv_required_sweeps = 2
                linear.solve(n_steps, mps.center == 0, conv_tol)
                min_site = np.argmin(np.array(linear.sweep_targets)[:, 1])
                mid_site = len(np.array(linear.sweep_targets)[:, 1]) // 2
                if mps.center == 0:
                    min_site = mps.n_sites - 2 - min_site
                    mid_site = mps.n_sites - 2 - mid_site
                if self.dyn_corr_method is not None:
                    mid_site = self.ncore + (mps.n_sites - self.ncore - self.nvirt) // 2
                _print("GF.IMAG MIN SITE = %4d" % min_site)
                _print("GF.IMAG MID SITE = %4d" % mid_site)

                if n_off_diag_cg == -2:
                    assert not diag_only
                    linear.noises = VectorDouble(noises[-1:])
                    linear.bra_bond_dims = VectorUBond(bond_dims[-1:])
                    if len(xw) != 0:
                        linear.gf_extra_omegas_at_site = mid_site
                        linear.gf_extra_omegas = VectorDouble(xw)
                    else:
                        linear.gf_extra_omegas_at_site = -1
                    
                    ext_mpss = []
                    ext_mes = []
                    for jj, idx2 in enumerate(idxs):

                        if jj > ii and rkets[jj].info.target == rkets[ii].info.target:

                            if rkets[jj].center != rkets[ii].center:
                                align_mps_center(rkets[jj], rkets[ii])

                            if self.verbose >= 2:
                                if self.mpi is not None:
                                    self.mpi.barrier()
                                rkets[jj].info.load_mutable()
                                _print("EXT left  bond dim = ", ["%6d" % x.n_states_total for x in rkets[jj].info.left_dims])
                                _print("EXT right bond dim = ", ["%6d" % x.n_states_total for x in rkets[jj].info.right_dims])
                                rkets[jj].info.deallocate_mutable()

                            ext_mpss.append(rkets[jj])
                            xme = MovingEnvironment(
                                impo, rkets[jj], rkets[ii], "XGF-%d" % jj)
                            xme.init_environments(False)
                            if self.dctr:
                                xme.delayed_contraction = OpNamesSet.normal_ops()
                            ext_mes.append(xme)
                    
                    linear.ext_mpss = VectorMPS(ext_mpss)
                    linear.ext_tmes = VectorMovingEnvironment(ext_mes)
                    linear.ext_target_at_site = mid_site
                    
                    linear.solve(1, mps.center == 0, 0)

                    # take middle site GF
                    if mps.center == 0:
                        rgf, igf = np.array(linear.sweep_targets)[::-1][mid_site, :]
                    else:
                        rgf, igf = np.array(linear.sweep_targets)[mid_site, :]
                    gf_mat[ii, ii, iw] = rgf + 1j * igf

                    if self.verbose >= 1:
                        _print("=== %3s     GF (%4d%4d | OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                            ("ADD" if addition else "REM", idx, idx, w, rgf, igf, time.perf_counter() - t))
                    
                    for ixw in range(len(xw)):
                        xrgf, xigf = np.array(linear.gf_extra_targets[ixw])
                        ex_gf_mats[iw][ii, ii, ixw] = xrgf + 1j * xigf
                        if self.verbose >= 1:
                            _print("=== %3s EXT GF (%4d%4d | OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                                ("ADD" if addition else "REM", idx, idx, xw[ixw], xrgf, xigf, 0))
                    
                    if self.verbose >= 2:
                        _print('>>> COMPLETE GF OMEGA = %10.5f Site = %4d %4d | Time = %.2f <<<' %
                            (w, idx, idx, time.perf_counter() - t))

                    jidx = 0
                    for jj, idx2 in enumerate(idxs):

                        if jj > ii and rkets[jj].info.target == rkets[ii].info.target:

                            if self.verbose >= 2:
                                _print('>>>   START  GF OMEGA = %10.5f Site = %4d %4d <<<' % (
                                    w, idx2, idx))
                            
                            rgf, igf = np.array(linear.ext_targets[jidx])
                            gf_mat[jj, ii, iw] = rgf + 1j * igf
                            gf_mat[ii, jj, iw] = rgf + 1j * igf

                            if self.verbose >= 1:
                                _print("=== %3s     GF (%4d%4d | OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                                    ("ADD" if addition else "REM", idx2, idx, w, rgf, igf, 0))

                            for ixw in range(len(xw)):
                                xrgf, xigf = np.array(linear.gf_extra_ext_targets[jidx][ixw])
                                ex_gf_mats[iw][jj, ii, ixw] = xrgf + 1j * xigf
                                ex_gf_mats[iw][ii, jj, ixw] = xrgf + 1j * xigf
                                if self.verbose >= 1:
                                    _print("=== %3s EXT GF (%4d%4d | OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                                        ("ADD" if addition else "REM", idx2, idx, xw[ixw], xrgf, xigf, 0))

                            if self.verbose >= 2:
                                _print('>>> COMPLETE GF OMEGA = %10.5f Site = %4d %4d | Time = %.2f <<<' %
                                    (w, idx2, idx, 0))

                            jidx += 1
                    
                    dmain, dseco, imain, iseco = Global.frame.peak_used_memory

                    continue

                if len(xw) != 0:
                    linear.noises = VectorDouble(noises[-1:])
                    linear.bra_bond_dims = VectorUBond(bond_dims[-1:])
                    linear.gf_extra_omegas_at_site = mid_site
                    linear.gf_extra_omegas = VectorDouble(xw)
                    linear.solve(1, mps.center == 0, 0)

                # take middle site GF
                if mps.center == 0:
                    rgf, igf = np.array(linear.sweep_targets)[::-1][mid_site, :]
                else:
                    rgf, igf = np.array(linear.sweep_targets)[mid_site, :]
                gf_mat[ii, ii, iw] = rgf + 1j * igf

                dmain, dseco, imain, iseco = Global.frame.peak_used_memory

                if self.verbose >= 1:
                    _print("=== %3s     GF (%4d%4d | OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                           ("ADD" if addition else "REM", idx, idx, w, rgf, igf, time.perf_counter() - t))

                for ixw in range(len(xw)):
                    xrgf, xigf = np.array(linear.gf_extra_targets[ixw])
                    ex_gf_mats[iw][ii, ii, ixw] = xrgf + 1j * xigf
                    if self.verbose >= 1:
                        _print("=== %3s EXT GF (%4d%4d | OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                            ("ADD" if addition else "REM", idx, idx, xw[ixw], xrgf, xigf, 0))
                    
                if self.verbose >= 2:
                    _print('>>> COMPLETE GF OMEGA = %10.5f Site = %4d %4d | Time = %.2f <<<' %
                           (w, idx, idx, time.perf_counter() - t))

                if diag_only:
                    continue

                for jj, idx2 in enumerate(idxs):

                    if jj > ii and rkets[jj].info.target == rkets[ii].info.target:

                        if rkets[jj].center != rkets[ii].center:
                            align_mps_center(rkets[jj], rkets[ii])

                        if self.verbose >= 2:
                            _print('>>>   START  GF OMEGA = %10.5f Site = %4d %4d <<<' % (
                                w, idx2, idx))
                        t = time.perf_counter()

                        tme = MovingEnvironment(
                            impo, rkets[jj], rkets[ii], "GF")
                        tme.init_environments(False)
                        if self.dctr:
                            tme.delayed_contraction = OpNamesSet.normal_ops()
                        linear.noises = VectorDouble(noises[-1:])
                        linear.bra_bond_dims = VectorUBond(bond_dims[-1:])
                        linear.target_bra_bond_dim = cps_bond_dims[-1]
                        linear.target_ket_bond_dim = cps_bond_dims[-1]
                        linear.gf_extra_omegas_at_site = -1
                        linear.tme = tme
                        if n_off_diag_cg == 0:
                            if len(xw) != 0:
                                linear.gf_extra_omegas_at_site = mps.center
                                linear.gf_extra_omegas = VectorDouble(xw)
                            linear.solve(1, mps.center != 0, 0)
                            rgf, igf = linear.targets[-1]
                        else:
                            if len(xw) != 0:
                                linear.gf_extra_omegas_at_site = mid_site
                                linear.gf_extra_omegas = VectorDouble(xw)
                            linear.linear_soft_max_iter = n_off_diag_cg if n_off_diag_cg != -1 else max_cg_iter
                            linear.solve(1, mps.center == 0, 0)
                            if mps.center == 0:
                                rgf, igf = np.array(linear.sweep_targets)[::-1][mid_site]
                            else:
                                rgf, igf = np.array(linear.sweep_targets)[mid_site]
                        gf_mat[jj, ii, iw] = rgf + 1j * igf
                        gf_mat[ii, jj, iw] = rgf + 1j * igf

                        if self.verbose >= 1:
                            _print("=== %3s     GF (%4d%4d | OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                                   ("ADD" if addition else "REM", idx2, idx, w, rgf, igf, time.perf_counter() - t))

                        for ixw in range(len(xw)):
                            xrgf, xigf = np.array(linear.gf_extra_targets[ixw])
                            ex_gf_mats[iw][jj, ii, ixw] = xrgf + 1j * xigf
                            ex_gf_mats[iw][ii, jj, ixw] = xrgf + 1j * xigf
                            if self.verbose >= 1:
                                _print("=== %3s EXT GF (%4d%4d | OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                                    ("ADD" if addition else "REM", idx2, idx, xw[ixw], xrgf, xigf, 0))

                        if self.verbose >= 2:
                            _print('>>> COMPLETE GF OMEGA = %10.5f Site = %4d %4d | Time = %.2f <<<' %
                                   (w, idx2, idx, time.perf_counter() - t))
        mps.save_data()
        mps_info.deallocate()

        if self.print_statistics:
            _print("GF PEAK MEM USAGE:",
                   "DMEM = ", GFDMRG.fmt_size(dmain + dseco),
                   "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
                   "IMEM = ", GFDMRG.fmt_size(imain + iseco),
                   "(%.0f%%)" % (imain * 100 / (imain + iseco)))

        return self.assemble_gf_mat(gf_mat, ex_gf_mats, freqs, extra_freqs)

    def assemble_gf_mat(self, main_gf_mat, ex_gf_mats, freqs, extra_freqs):
        all_freqs = np.array(sorted(list(set(list(freqs) +
            ([x for xx in extra_freqs for x in xx] if extra_freqs is not None else [])))))
        gf_mat = np.zeros((main_gf_mat.shape[0], main_gf_mat.shape[1], len(all_freqs)), dtype=complex)
        for ix, (freq, ex_freqs) in enumerate(zip(freqs, extra_freqs)):
            gf_mat[:, :, all_freqs == freq] = main_gf_mat[:, :, ix:ix + 1]
            for ixx, ex_freq in enumerate(ex_freqs):
                gf_mat[:, :, all_freqs == ex_freq] = ex_gf_mats[ix][:, :, ixx:ixx + 1]
        return gf_mat
    
    def sample_det(self, sample_cutoff):

        mps_info = MPSInfo(0)
        mps_info.load_data(self.scratch + "/GS_MPS_INFO")
        mps = MPS(mps_info)
        mps.load_data()

        tx = time.perf_counter()
        dtrie = DeterminantTRIE(mps.n_sites, True)
        dtrie.evaluate(UnfusedMPS(mps), sample_cutoff)
        _print("dtrie finished", time.perf_counter() - tx)
        if self.mpi is None or self.mpi.rank == 0:
            dname = "CSF" if SpinLabel == SU2 else "DET"
            _print("Number of %s = %10d (cutoff = %9.5g)" %
                   (dname, len(dtrie), sample_cutoff))
            ddstr = "0+-2" if SpinLabel == SU2 else "0ab2"
            dvals = np.array(dtrie.vals)
            gidx = np.argsort(np.abs(dvals))[::-1][:500]
            _print("Sum of weights of sampled %s = %20.15f\n" %
                   (dname, (dvals ** 2).sum()))
            for ii, idx in enumerate(gidx):
                if self.ridx is not None:
                    det = ''.join([ddstr[x] for x in np.array(dtrie[idx])[self.ridx]])
                else:
                    det = ''.join([ddstr[x] for x in np.array(dtrie[idx])])
                val = dvals[idx]
                _print(dname, "%10d" % ii, det, " = %20.15f" % val)
            if len(dvals) > 500:
                _print(" ... and more ... ")
            dets = np.zeros((len(dtrie), mps.n_sites), dtype=np.uint8)
            for i in range(len(dtrie)):
                if self.ridx is not None:
                    dets[i] = np.array(dtrie[i])[self.ridx]
                else:
                    dets[i] = np.array(dtrie[i])
            np.save(self.scratch + "/sample-vals.npy", dvals)
            np.save(self.scratch + "/sample-dets.npy", dets)

    def __del__(self):
        if self.hamil is not None:
            self.hamil.deallocate()
        if self.fcidump is not None:
            self.fcidump.deallocate()
        if self.mpo_orig is not None:
            self.mpo_orig.deallocate()
        release_memory()


def dmrg_mo_gf(mf, freqs, delta, ao_orbs=None, mo_orbs=None, gmres_tol=1E-7, add_rem='+-',
               extra_freqs=None, extra_delta=None, n_threads=8, memory=1E9, verbose=1, ignore_ecore=True,
               gs_bond_dims=[500], gs_noises=[1E-5, 1E-5, 1E-6, 1E-7, 0], gs_tol=1E-10, gs_n_steps=30,
               cps_bond_dims=[500], cps_noises=[0], cps_tol=1E-10, cps_n_steps=30,
               gf_bond_dims=[750], gf_noises=[1E-5, 0], gf_tol=1E-8, gf_n_steps=20, scratch='./tmp',
               mo_basis=True, load_dir=None, save_dir=None, pdm_return=True, reorder_method=None,
               lowdin=False, diag_only=False, alpha=True, occs=None, bias=1.0, cutoff=1E-14,
               n_off_diag_cg=0, mpi=None, sample_det=None, do_2pdm=False,
               dyn_corr_method=None, ncore=0, nvirt=0, algo_type='auto', site_type=1,
               dmain_ratio=0.4):
    '''
    Calculate the DMRG GF matrix in the MO or AO basis.

    Args:
        mf : scf object
        freqs : np.ndarray of frequencies (real)
        extra_freqs : list of np.ndarray of frequencies (real) or None
            the extra freqs to be calculated in the same basis for each freq in freqs
        delta : broadening (real)
        extra_delta : if not None, use this broadening for extra_freqs
        ao_orbs : list of indices of atomic orbtials (if not None, gf will be in ao basis)
        mo_orbs : list of indices of molecular orbtials (if not None, gf will be in mo basis)
            one of ao_orbs or mo_orbs must be None
            if both ao_orbs and mo_orbs are None, gf will be in mo basis
        gmres_tol : conjugate gradient (min res) conv tol (if too low will be extemely time-consuming)
        add_rem : '+' (addition) or '-' (removal) or '+-' (both)
        n_threads : number of threads (need parallel MKL library)
        memory : stack memory in bytes (default is 1 GB)
        verbose : 0 (quiet) 1 (per omega) 2 (per sweep) 3 (per orbital) 4 (per cg iteration)
        ignore_ecore : if True, set ecore to zero (should not affect GF)
        gs_bond_dims : np.ndarray of integers. Ground-State DMRG MPS bond dims for each sweep
        gs_noises : np.ndarray of float64. Ground-State DMRG noises for each sweep
        gs_tol : float64. Ground-State DMRG energy convergence.
        gs_n_steps : int. Ground-State DMRG max number of sweeps.
        cps_bond_dims : np.ndarray of integers. Compression MPS bond dims for each sweep
        cps_noises : np.ndarray of float64. Compression noises for each sweep
        cps_tol : float64. Compression energy convergence.
        cps_n_steps : int. Compression max number of sweeps.
        gf_bond_dims : np.ndarray of integers. Green's function MPS bond dims for each sweep
        gf_noises : np.ndarray of float64. Green's function noises for each sweep
        gf_tol : float64. Green's function Im GF (i, i) convergence.
        gf_n_steps : int. Green's function max number of sweeps.
        scratch : scratch folder for temporary files.
        lowdin : if True, will use lowdin orbitals instead of molecular orbitals
        diag_only : if True, only calculate diagonal GF elements.
        alpha : bool. alpha spin or beta spin (not used if SU2)
        occs : list(double) or None
            if occs = None, use FCI init MPS (default)
            if occs = list(double), use occ init MPS
            either
                (A) len(occs) == n_sites (RHF or UHF, 0 <= occ <= 2)
             or (B) len(occs) == n_sites * 2 (UHF, 0 <= occ <= 1)
                    order: 0a, 0b, 1a, 1b, ...
        bias : float64, effective when occs is not None
            bias = 0.0: HF occ (e.g. occs => 2 2 2 ... 0 0 0)
            bias = 1.0: no bias (e.g. occs => 1.7 1.8 ... 0.1 0.05)
            bias = +inf: fully random (e.g. occs => 1 1 1 ... 1 1 1)

            increase bias if you want an init MPS with larger bond dim

            0 <= occ <= 2
            biased occ = 1 + (occ - 1) ** bias    (if occ > 1)
                        = 1 - (1 - occ) ** bias    (if occ < 1)
        cutoff : float64, lowest kept density matrix eigen value (default 1E-14)
            use smaller cutoff (e.g. 1E-20) when you need super accurate ground state energy
            noise smaller than cutoff will have no effect.
        load_dir : if not None, skip ground-state calculation and load previous results
        save_dir : if not None, save results to a separate dir after ground-state calculation
            Note: if both load_dir and save_dir are not None, will restart ground-state calculation
                and save results to a separate dir
        mpi : if not None, MPI is used
        mo_basis: if False, will use Hamiltonian in AO basis
        pdm_return: if False, not return 1-pdm 
        reorder_method: None or 'fielder' or 'gaopt'
        n_off_diag_cg: limit number of cg for off-diagonal GF matrix elements
            if zero, no sweep is performed for off-diagonal GF matrix elements
        sample_det: if not None, the cutoff of det/csf sampling
        do_2pdm: whether to calcualte 2pdm
        dyn_corr_method: str or None; if not None, treat part of the orbitals as core and virtual
            can be "casci", "mrcis", "mrcisd", "mrcisdt", or "mrcisdtq".
            "gfmrci*" will only apply mrci in compression and gf; but gs will be dmrg.
        ncore: int, number of core orbitals. Core orbitals must have smallest indices.
        nvirt: int, number of virtual orbitals. Virtual orbitals must have biggest indices.
        algo_type: str, algorithm type for one pdm / two pdm
            can be auto (default), fast or normal. auto is the same as fast
        site_type: 0 or 1 (default) or 2, site type of sweep algorithm for one pdm / two pdm
            The smaller, the better memory and computational complexity
        dmain_ratio: the percentage of the main stack memory in block2

    Returns:
        gfmat : np.ndarray of dims (len(mo_orbs), len(mo_orbs), len(freqs)) (complex)
            GF matrix in the MO basis (for selected mo_orbs).
    '''
    from pyscf import lib, lo, symm, ao2mo
    assert ao_orbs is None or mo_orbs is None

    if mpi is not None:
        mpi = MPI

    mol = mf.mol

    pg = mol.symmetry.lower()
    if pg == 'd2h':
        fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
    elif pg == 'c1':
        fcidump_sym = ["A"]
    else:
        raise GFDMRGError("Point group %d not supported yet!" % pg)

    if lowdin:
        mo_coeff = lo.orth.lowdin(mol.intor('cint1e_ovlp_sph'))
    else:
        mo_coeff = mf.mo_coeff

    is_uhf = isinstance(mo_coeff, tuple) or mo_coeff.ndim == 3

    if not is_uhf:

        if mo_basis:
            n_mo = mo_coeff.shape[1]
            orb_sym = [1] * n_mo
            h1e = mo_coeff.T @ mf.get_hcore() @ mo_coeff
            g2e = ao2mo.restore(8, ao2mo.kernel(mf._eri, mo_coeff), n_mo)
            ecore = mol.energy_nuc()
            if ignore_ecore:
                ecore = 0
            na = nb = mol.nelectron // 2
        else:
            n_mo = mo_coeff.shape[1]
            orb_sym = [1] * n_mo
            h1e = mf.get_hcore()
            g2e = ao2mo.restore(8, mf._eri, n_mo)
            ecore = mol.energy_nuc()
            if ignore_ecore:
                ecore = 0
            na = nb = mol.nelectron // 2

    else:

        if mo_basis:
            mo_coeff_a, mo_coeff_b = mo_coeff[0], mo_coeff[1]
            n_mo = mo_coeff_b.shape[1]
            orb_sym = [1] * n_mo
            h1ea = mo_coeff_a.T @ mf.get_hcore() @ mo_coeff_a
            h1eb = mo_coeff_b.T @ mf.get_hcore() @ mo_coeff_b
            g2eaa = ao2mo.restore(8, ao2mo.kernel(mf._eri, mo_coeff_a), n_mo)
            g2ebb = ao2mo.restore(8, ao2mo.kernel(mf._eri, mo_coeff_b), n_mo)
            g2eab = ao2mo.kernel(
                mf._eri, [mo_coeff_a, mo_coeff_a, mo_coeff_b, mo_coeff_b])
            h1e = (h1ea, h1eb)
            g2e = (g2eaa, g2ebb, g2eab)
            ecore = mol.energy_nuc()
            if ignore_ecore:
                ecore = 0
            na, nb = mol.nelectron

        else:
            mo_coeff_a, mo_coeff_b = mo_coeff[0], mo_coeff[1]
            n_mo = mo_coeff_b.shape[1]
            orb_sym = [1] * n_mo
            h1e = mf.get_hcore()
            h1e = (h1e[0], h1e[1])
            g2eaa = ao2mo.restore(8, mf._eri[0], n_mo)
            g2ebb = ao2mo.restore(8, mf._eri[1], n_mo)
            g2eab = ao2mo.restore(4, mf._eri[2], n_mo)
            g2e = (g2eaa, g2ebb, g2eab)
            ecore = mol.energy_nuc()
            if ignore_ecore:
                ecore = 0
            na, nb = mol.nelectron

    dmrg = GFDMRG(scratch=scratch, memory=memory,
                  verbose=verbose, omp_threads=n_threads, mpi=mpi,
                  dmain_ratio=dmain_ratio)

    dmrg.dyn_corr_method = dyn_corr_method
    dmrg.ncore = ncore
    dmrg.nvirt = nvirt

    if load_dir is None:
        if dmrg.dyn_corr_method is None:
            re_idx = orbital_reorder(h1e, g2e, method=reorder_method)
        else:
            re_idx = orbital_reorder_subspace(h1e, g2e, ncore, nvirt, method=reorder_method)
        if mpi is not None:
            from mpi4py import MPI as MPIPY
            comm = MPIPY.COMM_WORLD
            re_idx = comm.bcast(re_idx, root=0)

        save_fd = save_dir + "/GS_FCIDUMP" if save_dir is not None else None
        dmrg.init_hamiltonian(pg, n_sites=n_mo, n_elec=na + nb, twos=na - nb, isym=1,
                              orb_sym=orb_sym, e_core=ecore, h1e=h1e, g2e=g2e, idx=re_idx,
                              save_fcidump=save_fd)
        dmrg.dmrg(bond_dims=gs_bond_dims, noises=gs_noises,
                  n_steps=gs_n_steps, conv_tol=gs_tol, occs=occs, bias=bias, cutoff=cutoff)
        if save_dir is not None:
            _print('saving ground state ...')
            dmrg.save_gs_mps(save_dir)
            if mpi.rank == 0:
                np.save(save_dir + "/reorder.npy", re_idx)
    else:
        dmrg.init_hamiltonian_fcidump(pg, load_dir + "/GS_FCIDUMP")
        re_idx = np.load(load_dir + "/reorder.npy")
        dmrg.idx = re_idx
        dmrg.ridx = np.argsort(re_idx)
        _print('loading ground state ...')
        dmrg.load_gs_mps(load_dir)

        if save_dir is not None:
            dmrg.restart_gs = True
            dmrg.dmrg(bond_dims=gs_bond_dims, noises=gs_noises,
                n_steps=gs_n_steps, conv_tol=gs_tol, occs=occs,
                bias=bias, cutoff=cutoff)
            _print('saving ground state ...')
            dmrg.save_gs_mps(save_dir)
            if mpi.rank == 0:
                np.save(save_dir + "/reorder.npy", re_idx)

    _print('reorder method = %r reorder = %r' % (reorder_method, re_idx))

    if sample_det is not None:
        dmrg.sample_det(sample_det)

    if mo_orbs is None and ao_orbs is None:
        mo_orbs = range(n_mo)
        mo_orbs = np.argsort(re_idx)[np.array(mo_orbs)]

    pdm = dmrg.get_one_pdm(algo_type=algo_type, site_type=site_type)
    if do_2pdm:
        pdm2 = dmrg.get_two_pdm(algo_type=algo_type, site_type=site_type)

    idxs = mo_orbs if ao_orbs is None else ao_orbs
    all_freqs = np.array(sorted(list(set(list(freqs) +
        ([x for xx in extra_freqs for x in xx] if extra_freqs is not None else [])))))
    gf = np.zeros((len(idxs), len(idxs), len(all_freqs)), dtype=complex)
    if is_uhf:
        gf_beta = np.zeros((len(idxs), len(idxs), len(all_freqs)), dtype=complex)

    for addit in [x == '+' for x in add_rem]:
        # only calculate alpha spin
        gf += dmrg.greens_function(gf_bond_dims, gf_noises, gmres_tol, gf_tol, gf_n_steps,
                                       cps_bond_dims, cps_noises, cps_tol, cps_n_steps,
                                       idxs=mo_orbs if ao_orbs is None else ao_orbs,
                                       mo_coeff=None if ao_orbs is None or not mo_basis else (mo_coeff[0] if is_uhf else mo_coeff),
                                       eta=delta, freqs=freqs, extra_freqs=extra_freqs, extra_eta=extra_delta,
                                       addition=addit, diag_only=diag_only,  alpha=alpha, n_off_diag_cg=n_off_diag_cg)

    if is_uhf:
        for addit in [x == '+' for x in add_rem]:
            # calculate beta spin
            gf_beta += dmrg.greens_function(gf_bond_dims, gf_noises, gmres_tol, gf_tol, gf_n_steps,
                                                cps_bond_dims, cps_noises, cps_tol, cps_n_steps,
                                                idxs=mo_orbs if ao_orbs is None else ao_orbs,
                                                mo_coeff=None if ao_orbs is None or not mo_basis else (mo_coeff[1] if is_uhf else mo_coeff),
                                                eta=delta, freqs=freqs, extra_freqs=extra_freqs, extra_eta=extra_delta,
                                                addition=addit, diag_only=diag_only, alpha=False, n_off_diag_cg=n_off_diag_cg)

    if is_uhf:
        gf = np.asarray((gf, gf_beta))

    del dmrg

    if pdm_return:
        if do_2pdm:
            return pdm, pdm2, gf
        else:
            return pdm, gf
    else:
        return gf

def dmrg_mo_pdm(mf, ao_orbs=None, mo_orbs=None, n_threads=8, memory=1E9, verbose=1, ignore_ecore=True,
               gs_bond_dims=[500], gs_noises=[1E-5, 1E-5, 1E-6, 1E-7, 0], gs_tol=1E-10, gs_n_steps=30,
               scratch='./tmp', mo_basis=True, load_dir=None, save_dir=None, reorder_method=None,
               lowdin=False, diag_only=False, occs=None, bias=1.0, cutoff=1E-14, mpi=None,
               dyn_corr_method=None, ncore=0, nvirt=0, do_2pdm=False, algo_type='auto', site_type=1,
               dmain_ratio=0.4):
    '''
    Calculate the DMRG density matrix in the MO or AO basis.

    Args:
        Same as in dmrg_mo_gf

    Returns:
        pdm : np.ndarray of dims (len(mo_orbs), len(mo_orbs)) (real)
            density matrix in the MO or AO basis (for selected mo_orbs).
    '''
    from pyscf import lib, lo, symm, ao2mo
    assert ao_orbs is None or mo_orbs is None

    if mpi is not None:
        mpi = MPI

    mol = mf.mol

    pg = mol.symmetry.lower()
    if pg == 'd2h':
        fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
    elif pg == 'c1':
        fcidump_sym = ["A"]
    else:
        raise GFDMRGError("Point group %d not supported yet!" % pg)

    if lowdin:
        mo_coeff = lo.orth.lowdin(mol.intor('cint1e_ovlp_sph'))
    else:
        mo_coeff = mf.mo_coeff

    is_uhf = isinstance(mo_coeff, tuple) or mo_coeff.ndim == 3

    if not is_uhf:

        if mo_basis:
            n_mo = mo_coeff.shape[1]
            orb_sym = [1] * n_mo
            h1e = mo_coeff.T @ mf.get_hcore() @ mo_coeff
            g2e = ao2mo.restore(8, ao2mo.kernel(mf._eri, mo_coeff), n_mo)
            ecore = mol.energy_nuc()
            if ignore_ecore:
                ecore = 0
            na = nb = mol.nelectron // 2
        else:
            n_mo = mo_coeff.shape[1]
            orb_sym = [1] * n_mo
            h1e = mf.get_hcore()
            g2e = ao2mo.restore(8, mf._eri, n_mo)
            ecore = mol.energy_nuc()
            if ignore_ecore:
                ecore = 0
            na = nb = mol.nelectron // 2

    else:

        if mo_basis:
            mo_coeff_a, mo_coeff_b = mo_coeff[0], mo_coeff[1]
            n_mo = mo_coeff_b.shape[1]
            orb_sym = [1] * n_mo
            h1ea = mo_coeff_a.T @ mf.get_hcore() @ mo_coeff_a
            h1eb = mo_coeff_b.T @ mf.get_hcore() @ mo_coeff_b
            g2eaa = ao2mo.restore(8, ao2mo.kernel(mf._eri, mo_coeff_a), n_mo)
            g2ebb = ao2mo.restore(8, ao2mo.kernel(mf._eri, mo_coeff_b), n_mo)
            g2eab = ao2mo.kernel(
                mf._eri, [mo_coeff_a, mo_coeff_a, mo_coeff_b, mo_coeff_b])
            h1e = (h1ea, h1eb)
            g2e = (g2eaa, g2ebb, g2eab)
            ecore = mol.energy_nuc()
            if ignore_ecore:
                ecore = 0
            na, nb = mol.nelectron

        else:
            mo_coeff_a, mo_coeff_b = mo_coeff[0], mo_coeff[1]
            n_mo = mo_coeff_b.shape[1]
            orb_sym = [1] * n_mo
            h1e = mf.get_hcore()
            h1e = (h1e[0], h1e[1])
            g2eaa = ao2mo.restore(8, mf._eri[0], n_mo)
            g2ebb = ao2mo.restore(8, mf._eri[1], n_mo)
            g2eab = ao2mo.restore(4, mf._eri[2], n_mo)
            g2e = (g2eaa, g2ebb, g2eab)
            ecore = mol.energy_nuc()
            if ignore_ecore:
                ecore = 0
            na, nb = mol.nelectron

    dmrg = GFDMRG(scratch=scratch, memory=memory,
                  verbose=verbose, omp_threads=n_threads, mpi=mpi,
                  dmain_ratio=dmain_ratio)

    dmrg.dyn_corr_method = dyn_corr_method
    dmrg.ncore = ncore
    dmrg.nvirt = nvirt

    if load_dir is None:
        if dmrg.dyn_corr_method is None:
            re_idx = orbital_reorder(h1e, g2e, method=reorder_method)
        else:
            re_idx = orbital_reorder_subspace(h1e, g2e, ncore, nvirt, method=reorder_method)
        if mpi is not None:
            from mpi4py import MPI as MPIPY
            comm = MPIPY.COMM_WORLD
            re_idx = comm.bcast(re_idx, root=0)

        save_fd = save_dir + "/GS_FCIDUMP" if save_dir is not None else None
        dmrg.init_hamiltonian(pg, n_sites=n_mo, n_elec=na + nb, twos=na - nb, isym=1,
                              orb_sym=orb_sym, e_core=ecore, h1e=h1e, g2e=g2e, idx=re_idx,
                              save_fcidump=save_fd)
        dmrg.dmrg(bond_dims=gs_bond_dims, noises=gs_noises,
                  n_steps=gs_n_steps, conv_tol=gs_tol, occs=occs, bias=bias, cutoff=cutoff)
        if save_dir is not None:
            _print('saving ground state ...')
            dmrg.save_gs_mps(save_dir)
            if mpi.rank == 0:
                np.save(save_dir + "/reorder.npy", re_idx)
    else:
        dmrg.init_hamiltonian_fcidump(pg, load_dir + "/GS_FCIDUMP")
        re_idx = np.load(load_dir + "/reorder.npy")
        dmrg.idx = re_idx
        dmrg.ridx = np.argsort(re_idx)
        _print('loading ground state ...')
        dmrg.load_gs_mps(load_dir)

        if save_dir is not None:
            dmrg.restart_gs = True
            dmrg.dmrg(bond_dims=gs_bond_dims, noises=gs_noises,
                n_steps=gs_n_steps, conv_tol=gs_tol, occs=occs,
                bias=bias, cutoff=cutoff)
            _print('saving ground state ...')
            dmrg.save_gs_mps(save_dir)
            if mpi.rank == 0:
                np.save(save_dir + "/reorder.npy", re_idx)

    _print('reorder method = %r reorder = %r' % (reorder_method, re_idx))

    if mo_orbs is None and ao_orbs is None:
        mo_orbs = range(n_mo)
        mo_orbs = np.argsort(re_idx)[np.array(mo_orbs)]

    pdm = dmrg.get_one_pdm(algo_type=algo_type, site_type=site_type)
    if do_2pdm:
        pdm2 = dmrg.get_two_pdm(algo_type=algo_type, site_type=site_type)

    del dmrg

    if do_2pdm:
        return pdm, pdm2
    else:
        return pdm


if __name__ == "__main__":

    # parameters
    n_threads = 14
    hf_type = "RHF"  # RHF or UHF
    mpg = 'c1'  # point group: d2h or c1
    scratch = './tmp'
    save_dir = './gs_mps'
    load_dir = None
    lowdin = False
    do_ccsd = True

    import os
    if MPI.rank == 0:
        if not os.path.isdir(scratch):
            os.mkdir(scratch)
        if save_dir is not None:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
    MPI.barrier()
    os.environ['TMPDIR'] = scratch

    from pyscf import gto, scf

    # H chain
    N = 10
    BOHR = 0.52917721092  # Angstroms
    R = 1.8 * BOHR
    mol = gto.M(atom=[['H', (0, 0, i * R)] for i in range(N)],
                basis='sto6g', verbose=0, symmetry=mpg)
    pg = mol.symmetry.lower()

    if hf_type == "RHF":
        mf = scf.RHF(mol)
        ener = mf.kernel()
    elif hf_type == "UHF":
        assert SpinLabel == SZ
        mf = scf.UHF(mol)
        ener = mf.kernel()

    _print("SCF Energy = %20.15f" % ener)
    _print(("NON-" if SpinLabel == SZ else "") + "SPIN-ADAPTED")

    if do_ccsd:
        from pyscf import cc
        mcc = cc.CCSD(mf)
        mcc.kernel()
        dmmo = mcc.make_rdm1()
        if hf_type == "RHF":
            # RHF: 0 <= occ <= 2, len = n_mo
            occs = np.diag(dmmo)
        else:
            # UHF: 0 <= occ <= 1, order: 0a, 0b, 1a, 1b, ..., len = n_mo * 2
            occs = np.array(
                [i for j in zip(np.diag(dmmo[0]), np.diag(dmmo[1])) for i in j])
        _print('OCCS = ', occs)
    else:
        occs = None

    if MPI is not None:
        from mpi4py import MPI as MPIPY
        comm = MPIPY.COMM_WORLD
        occs = comm.bcast(occs, root=0)
        ener = comm.bcast(ener, root=0)
        mf.mo_coeff = comm.bcast(mf.mo_coeff, root=0)

    if lowdin:

        eta = 0.005
        freqs = np.arange(-0.8, -0.2, 0.01)
        mo_orbs = [4]
        t = time.perf_counter()
        pdm, gfmat = dmrg_mo_gf(mf, freqs=freqs, delta=eta, mo_orbs=mo_orbs, scratch=scratch, add_rem='-',
                                gf_bond_dims=[150], gf_noises=[1E-3, 5E-4], gf_tol=1E-4,
                                gmres_tol=1E-8, lowdin=True, ignore_ecore=False, n_threads=n_threads)

        _print(gfmat)  # alpha only

        # alpha + beta
        pdos = (-2 / np.pi) * gfmat.imag.trace(axis1=0, axis2=1)
        _print("PDOS = ", pdos)
        _print("TIME = ", time.perf_counter() - t)

        import matplotlib.pyplot as plt

        plt.plot(freqs, pdos, 'o-', markersize=2)
        plt.xlabel('Frequency $\\omega$ (a.u.)')
        plt.ylabel('LDOS (a.u.)')
        plt.savefig('gf-figure.png', dpi=600)

    else:

        eta = 0.005
        freqs = [-0.2]
        t = time.perf_counter()
        pdm, gfmat = dmrg_mo_gf(mf, freqs=freqs, delta=eta, mo_orbs=None, scratch=scratch, add_rem='+-',
                                gs_bond_dims=[500], gs_noises=[1E-7, 1E-8, 1E-10, 0], gs_tol=1E-14, gs_n_steps=30,
                                cps_bond_dims=[500], cps_noises=[0], cps_tol=1E-14, cps_n_steps=30,
                                gf_bond_dims=[500], gf_noises=[1E-7, 1E-8, 1E-10, 0], gf_tol=1E-8,
                                gmres_tol=1E-20, lowdin=False, ignore_ecore=False, alpha=False, verbose=3,
                                n_threads=n_threads, occs=None, bias=1.0, save_dir=save_dir, load_dir=load_dir,
                                n_off_diag_cg=0, mpi=True)
        xgfmat = np.einsum('ip,pqr,jq->ijr', mf.mo_coeff, gfmat, mf.mo_coeff)
        _print("MO to AO method = ", xgfmat)

        pdm, gfmat = dmrg_mo_gf(mf, freqs=freqs, delta=eta, mo_orbs=None, scratch=scratch, add_rem='+-',
                                gs_bond_dims=[500], gs_noises=[1E-7, 1E-8, 1E-10, 0], gs_tol=1E-14, gs_n_steps=30,
                                cps_bond_dims=[500], cps_noises=[0], cps_tol=1E-14, cps_n_steps=30,
                                gf_bond_dims=[500], gf_noises=[1E-7, 1E-8, 1E-10, 0], gf_tol=1E-8,
                                gmres_tol=1E-20, lowdin=False, ignore_ecore=False, alpha=False, verbose=3,
                                ao_orbs=range(N),
                                n_threads=n_threads, occs=None, bias=1.0, save_dir=save_dir, load_dir=load_dir)
        _print("AO IN MO method = ", gfmat)

        _print('diff = ', gfmat - xgfmat)
        _print(np.linalg.norm(gfmat - xgfmat))
