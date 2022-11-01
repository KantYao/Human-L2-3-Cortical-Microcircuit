# Human L2/3 Cortical Circuit Model --- Yao-et-al.-2022
==============================================================================
Author: Heng Kang yao

This is the readme for the model associated with the paper:

Yao, H. K. et al. Reduced inhibition in depression impairs
stimulus processing in human cortical microcircuits. Cell Rep.
38, (2022).


Network Simulations:
Simulation code associated with the default L2/3 circuit used throughout the manuscript is in the /L23Net directory.

To run simulations, install all of the necessary python modules (see lfpy_env.yml), compile the mod files within the mod folder, and submit the simulations in parallel. To run the depression condition, set MDD to True.

To run the circuit in parallel, use the following command:

mpiexec -n 400 python circuit.py 1234

The 400 corresponds to the number of CPU and 1234 correspond to the seed.