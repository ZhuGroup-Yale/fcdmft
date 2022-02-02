fcdmft
======

Ab initio full cell DMFT and GW+DMFT package based on PySCF

Authors: Tianyu Zhu (tianyu.zhu@yale.edu), Huanchen Zhai, Zhihao Cui, Linqing Peng, Garnet Chan

Installation
------------

* Prerequisites
    - PySCF 1.7 or higher, and all dependencies 
	- libdmet (by Zhi-Hao Cui, https://github.com/gkclab/libdmet_preview)
	- block2 (optional, by Huanchen Zhai, https://github.com/block-hczhai/block2-preview)
	- CheMPS2 (optional)

* You need to set environment variable `PYTHONPATH` to export fcdmft to Python. 
  E.g. if fcdmft is installed in `/opt`, your `PYTHONPATH` should be

        export PYTHONPATH=/opt/fcdmft:$PYTHONPATH

Features
--------

* Full cell G0W0+DMFT and HF+DMFT (mixed MPI and OpenMP parallelization)

* Hamiltonian-based impurity solvers
	* Coupled-cluster Green's function
	* Quantum chemistry dynamical DMRG (from block2)
	* DMRG-MRCI Green's function (from block2)
	* FCI/ED Green's function (from CheMPS2, for test only)

* Molecular and periodic G0W0

* Molecular and periodic RPA

* CAS-CI treatment of the impurity problem

QuickStart
----------

You can find Python scripts for running DMFT calculations in `/fcdmft/examples`.
For example, in `/fcdmft/examples/Si`, the steps to run a full cell GW+DMFT
calculations are:

1. Perform DFT and G0W0 calculations by running `si_gw.py` 
(Note: For large systems, G0W0 should be performed separately using multiple nodes, 
i.e. MPI, see `/fcdmft/examples/NiO`);

2. Derive impurity Hamiltonian and GW double counting term by running `si_set_ham.py`;

3. Perform GW+DMFT calculation by running `run_dmft.py` (serial or MPI/OpenMP). 
All DMFT parameters should be set in `run_dmft.py`.

References
----------

Please cite the following papers in publications utilizing the fcdmft package:

* T. Zhu and G. K.-L. Chan, Phys. Rev. X 11, 021006 (2021)

* T. Zhu, Z.-H. Cui, and G. K.-L. Chan, J. Chem. Theory Comput. 16, 141-153 (2020)

* T. Zhu, C. A. Jimenez-Hoyos, J. McClain, T. C. Berkelbach, and G. K.-L. Chan, Phys. Rev. B 100, 115154 (2019)

Cite the following paper if GW code is used:

* T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 17, 727-741 (2021)

Cite the following paper if libdmet package is used:

* Z.-H. Cui, T. Zhu, and G. K.-L. Chan, J. Chem. Theory Comput. 16, 119-129 (2020)
