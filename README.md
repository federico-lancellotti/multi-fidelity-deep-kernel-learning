# Multi-Fidelity Deep Kernel Learning
This project proposes a Stochastic Variational Deep Kernel Learning method for the data-driven discovery of a low-dimensional representation of a dynamical system, from high-dimensional data at different levels of fidelity.
The framework is composed of multiple instances of a DKL model, each of them associated with a particular level of fidelity. Each instance includes an autoencoder, that compresses the high dimensional data (a video of the dynamical system) into a low-dimensional latent state space, and a latent dynamical model, that predicts the system evolution over time. 
The latent representation of the system and its dynamics is used to estimate the intrinsic dimensionality of the system and as a correction term for the higher levels of fidelity.

![Model scheme](/Reports/model.png)

## Getting Started
These instructions will help you run the code on you local machine. Please be aware that the usage of a GPU is strongly advised, especially for the training stage. Currently, only CUDA is supported for GPU-based training.
A config.yaml file is available to set the main parameters.

### Requirements
The code has been tested on Python 3.9.6 and 3.10.13.
The required libraries are collected in the requirements.txt file:
```
pip install -r requirements.txt
```

## Generate the dataset
The dataset can be generated by running the `generate_dataset.py` file. 

Please, be sure to select the desired environment and the desired number of episodes. The following are currently available:
- Gym framework: 
  - Pendulum (200 frames per episode);
  - Acrobot (500 frames per episode);
  - MountainCarContinuous (400 frames per episode).
- PDE framework:
  - reaction-diffusion;
  - diffusion-advection.

For the Gym datasets, the frames can be optionally cropped and/or an occlusion can be generated for specific levels of fidelity. It is sufficient to specify the portion of the cropping/occlusion and its position as a dictionary in the crop/occlusion parameter list.
An example, where for the second and third levels only 60% of the frame is selected, from the fourth quadrant:
```
crop: [{'portion': 1, 'pos': 4}, {'portion': 0.6, 'pos': 4}, {'portion': 1, 'pos': 4}, {'portion': 1, 'pos': 4}]
```
More info on this can be found in the documentation of the GenerateDataset class.

Two pickle files for each level of fidelity are eventually produced, one for training and the other for testing purposes, inside the `Data` folder.

## Train the model
The model can be initialized and trained by running the `main.py` file. Here you can also set `use_gpu = True` if you want to use your CUDA compatible GPU.

If you prefer to customize your training script, you can follow the follwing instructions.
To initialize the model, you firstly need to instantiate a BuildModel object.
You can 
- add a level to the model with the method `add_level(level, latent_dim)`, by specifying the desired level of fidelity and the desired dimension of its latent space;
- train the level with the method `train_level(level, model_n)`, by specifying the desired level of fidelity and passing the sub-model returned when adding the respective level;
- evaluate the trained level with the method `get_latent_representations(model_n, train_loader_n)`, by passing the trained sub-model and its data loader. By evaluating the model, the latent representations of the pair of frames (t-1,t) and (t,t+1) and of the dynamics of the system will be returned.

You can estimate the intrinsic dimension of the system with the function `estimate_ID(z_n, z_next_n, z_fwd_n)`, by passing the previously computed latent representations of the dynamical system.
You can then use the estimated ID as the `latent_dim` of the next level of fidelity.

The weights of the trained model will be saved in the folder `Results/NameOfEnv/%Y-%m-%d_%H-%M-%S`.
By default, an `ID.txt` file is produced and saved inside the same folder. It contains the estimate of the intrinsic dimension of the system computed until the second to last fidelity level, namely the dimension of the latent space of the level of highest fidelity.

## Test the model
The trained model can be tested by running the `test.py` file. 

Please, make sure you are correctly specifying the `results_folder, weights_filename, ID` parameters inside the `config.yaml` file. 

By default, the `test.py` file produces:
- (if Gym env) the plots of the true state variables, saved during the generation of the dataset;
- the plots of the latent variables of the level of highest fidelity;
- a tuple of four figures, for each tested time instant; each tuple includes: the reconstructed frame, its absolute error, the one-step forward predicted frame, its absolute error.
The plots can be found inside the folder `Results/NameOfEnv/%Y-%m-%d_%H-%M-%S/plots`.

If you prefer to customize your training script, you can follow the follwing instructions.
To test the model, you should firstly instantiate a BuildModel object, specifying the `test=True` argument. 
You can then:
- add a level to the model with the method `add_level(level, latent_dim)`, by specifying the desired level of fidelity and the used dimension of its latent space;
- test the level with the method `test_level(level, model_n)`, by specifying the desired level of fidelity and passing the sub-model returned when adding the respective level;
- evaluate the trained level with the method `eval_level(model_n, train_loader_n)`, by passing the trained sub-model and its data loader. By evaluating the model, the latent representations of the pair of frames (t-1,t) and (t,t+1) and of the dynamics of the system will be returned.

## `config.yaml` parameters
### General
- `seed`, seed.

### Dataset generation
- `env_name`, environment name (available: Pendulum, Acrobot, MountainCarContinuous, reaction-diffusion, diffusion-advection);
  
#### Gym
- `num_episodes`, number of training episodes; dictionary with an int for each fidelity level;
- `num_episodes_test`, number of testing episodes; dictionary with an int for each fidelity level;
- `crop`, whether to crop the frames; list of dictionaries (one for each fidelity level), with keys "portion" and "pos";
- `occlusion`, whether to mask the frames; list of dictionaries (one for each fidelity level), with keys "portion" and "pos".

#### PDE
- `d`, diffusion coefficients; dictionary with a double for each fidelity level;
- `mu`, mu training coefficients; dictionary with a list of doubles for each fidelity level;
- `mu_test`, mu testing coefficients; list of double;
- `T`: training time horizon; dictionary with an int for each fidelity level;
- `T_test`: testing time horizon; int;
- `dt`: step of time discretization, double;
- `L`: spacial domain size; int;
- `n`: grid size, should be the same as obs_dim_1 and obs_dim_2; dictionary with an in for each fidelity level.

### Model hyperparameters
- `batch_size`: dimension of the training batch; int;
- `max_epoch`: number of training epochs; int;
- `rho`: $\rho$ coefficient, to weight the LF correction; double;
- `training`: whether to activate the training; bool;
- `lr`: (general) learning rate; double;
- `lr_gp`: learning rate for the gaussian process; double;
- `lr_gp_lik`: learning rate for the likelihood; double;
- `lr_gp_var`: learning rate for the variational inference; double;
- `reg_coef`: regularization coefficient; double;
- `k1`: coefficient of the reconstruction term in the loss; double
- `k2`: coefficient of the forward K-L term in the loss; double;
- `grid_size`: size of the grid; int;
- `latent_dim`: dimension of the latent space of first sub-model (lowest fidelity); int;
- `obs_dim_1`: first dimension of the observation; dictionary with an int for each fidelity level;
- `obs_dim_2`: second dimension of the observation; dictionary with an int for each fidelity level;
- `obs_dim_3`: third dimension of the observation; int (typically `=6`, since two RGB frames are concatenated along the color channel);
- `h_dim`: dimension of the hidden layer; int;
- `jitter`: jitter to cure numerical instabilities; double;
- `log_interval`: interval for logging the trained weights; int;

### Datasets and weights
- `training_dataset`: path in the `Results` folder where to find the training dataset files; dictionary with a string for each fidelity level;
- `testing_dataset`: path in the `Results` folder where to find the testing dataset files; dictionary with a string for each fidelity level;
- `results_folder`: name of the folder where to look for the .pth weights files; string;
- `weights_filename`: name of the weights file (without the .pth); list of strings, one for each fidelity level;
- `ID`: estimated ID value; int.


## References
- https://github.com/nicob15/DeepKernelLearningOfDynamicalModels
- https://github.com/BoyuanChen/neural-state-variables