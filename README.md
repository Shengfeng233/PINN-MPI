# PINN-MPI
This repository presents a trial for Spatiotemporal parallel physics-informed neural networks. By incorprating RANS equations and adopting overlapping domain decomposition strategy, we managed to efficiently solve the inverse problem of fluid mechanics.

The sub-domains are divided along x,y,and t directions. For each sub-domain, a seperate independent sub neural network is allocated. The sub neural networks are trained simultaneously using MPI (mpi4py). The implementation is aimed for 2D unsteady flow thus the domain decomposition is splited in three directions. Adjustment can be easily made in the function "commmunicate" to implement 1D and 2D domain decomposition, while for 4D (3D unsteady flow), more complicated design will be needed.

The spatiotemporal parallel physics-informed neural networks are called STPINNs, in this repository, both the implementation of STPINN-NS and STPINN-RANS are presented.

The data are uploaded in the CFD_data through git_lfs.

Annotations are written in both English and Chinese.

Details of the algorithm and visualization can be found in <https://doi.org/10.1063/5.0155087>.

# Dependencies
torch(cpu), mpi4py, numpy, pyDOE, scipy, pandas, matplotlib, imageio
