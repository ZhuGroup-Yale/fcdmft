This folder contains scripts for running GW+DMFT for SrMoO3, a moderately correlated metal.
Band interpolation is used for getting large k-mesh GW bands, which is used for computing 
DMFT lattie GF and hybridization function.

Steps
-----

1. Run `srmoo3_lda.py` (serial run) at 5x5x5 k-mesh to get LDA results and GDF integrals.
2. Run `srmoo3_gw.py` (parallel run, normally 2-3 nodes, each node 4 MPI processes), to get GW results at 5x5x5 k-mesh.
3. Run `srmoo3_set_ham.py` (serial run) to get local basis and impurity Hamiltonian for DMFT.
4. Run `srmoo3_interpolate.py` (serial run) to get LDA bands at 12x12x12 k-mesh and interpolate GW results to 12x12x12 k-mesh. Steps 3 and 4 can be run simultaneously.
5. Run `run_dmft.py` to perform GW+DMFT calculation using local basis, impurity Hamiltonian, and interpolated GW bands from Steps 3 and 4.
