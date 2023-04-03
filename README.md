# DynamicsAE
![](https://github.com/tiwarylab/DynamicsAE/blob/main/VAE_z_movie.gif?raw=true "VAE") ![](https://github.com/tiwarylab/DynamicsAE/blob/main/DynAE_z_movie.gif?raw=true "Dynamics-AE")

DynamicsAE: A deep learning-based framework to uniquely identify an uncorrelated, isometric and meaningful latent representation.

Please read and cite this manuscript when using DynamicsAE: https://arxiv.org/abs/2209.00905. Here is an implementation of DynamicsAE in Pytorch.

## Data Preparation
Our implementation now only supports the npy files as the input, and also saves all the results into npy files for further anlyses. Users can refer to the data files in the ```examples``` subdirectory.


## Usage

To train and test model:

```
python main.py	-config	# Input the configuration file 
```

Here, a configuration file in INI format is supported, which allows a more flexible control of the training process. A sample configuration file is shown in the ```examples``` subdirectory. 

#### Example

Train and test SPIB on the three-well analytical potential:
```
python main.py -config examples/sample_config.ini
```