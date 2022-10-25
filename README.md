# PINN-MPI
Working for two months, after hunderds times of attempting, the idea of distinctively accelarating the training of PINNs through domain decomposition (implemented through MPI) eventually failed though this is so far the most elegant code we have written.

The reason why PINN-MPI failed mainly lies in the unknown distribution of fluid flow. In some area, there might be a large gradient; in some area, the fluid interacts heavily with the wall, leading to significantly different difficulty in solving these sub domains, and the pain in the ass problem is that you can not know this distribution beforehand.

Besides, to accurately solve the sub domains, the complexity of the network in the sub domain might not be simpler than the original "one" network, making the domain decomposition less necessary.

The extreme learning of the PINNs, which attempts to solve a very tiny subdomain with a very tiny neural network has also failed due to inability of solving  the tiny sub domain near wall.



Howerver, there are some conclusions we can get from this trial: 1.Breaking a large domain into several relatively large domains(less than 10) might help. 2.Domain decomposition along the time axis seems to always help. 

We left this unsuccessful code here to warn those who are interested in domain decomposition to be cautious, and wait to see if there are some more tricks we can adopt to solve this problem.
