# PINN-MPI
This repository presents a trial for Spatiotemporal parallel physics-informed neural networks. By incorprating RANS equations and adopting overlapping domain decomposition strategy, we managed to efficiently solve the inverse problem of fluid mechanics.

The sub-domains are divided along x,y,and t directions. For each sub-domain, a seperate independent sub neural network is allocated. The sub neural networks are trained simultaneously using MPI (mpi4py).

The spatiotemporal parallel physics-informed neural networks are called STPINNs, in this repository, both the implementation of STPINN-NS and STPINN-RANS are presented.

The data are uploaded in the CFD_data through git_lfs.

Annotations are written in both English and Chinese.

# Dependencies
torch(cpu), mpi4py, numpy, pyDOE, scipy, pandas, matplotlib, imageio
