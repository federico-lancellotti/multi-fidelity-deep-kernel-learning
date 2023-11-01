# Multi-Fidelity Deep Kernel Learning

### First step
Generate the dataset, a simulated video of a pendulum.

### Second step
Investigate the Intrinsic Dimensionality of the dynamical system, using the Levina-Bickel algoritm.

## To-Do
1. check the ID algorithm on different datasets
2. adapt the code to different sizes of the images (or keep the size constant but decrease the quality in some other ways)
3. add regularizer for the likelihood (?)

## Issues
1. the training does not work on Apple Silicon. multitask_gaussian_likelihood seems to be using the SparseMPS backend, currently not implemented