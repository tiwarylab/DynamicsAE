# DynamicsAE
Input|VAE: Imposing a Gaussian|Dynamics-AE: Imposing overdamped Langevin dynamics
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/tiwarylab/DynamicsAE/blob/main/files/three_state_transformed.jpg?raw=true" width="300" height="300"/>|<img src="https://github.com/tiwarylab/DynamicsAE/blob/main/files/3state_VAE_z_movie.gif?raw=true" width="300" height="300"/>|<img src="https://github.com/tiwarylab/DynamicsAE/blob/main/files/3state_DynAE_z_movie.gif?raw=true" width="300" height="300"/>
<img src="https://github.com/tiwarylab/DynamicsAE/blob/main/files/dSprites_x_pos_y_pos.gif?raw=true" width="300" height="300"/>|<img src="https://github.com/tiwarylab/DynamicsAE/blob/main/files/dSprites_2_VAE_z_movie.gif?raw=true" width="300" height="300"/>|<img src="https://github.com/tiwarylab/DynamicsAE/blob/main/files/dSprites_2_DynAE_z_movie.gif?raw=true" width="300" height="300"/>
<img src="https://github.com/tiwarylab/DynamicsAE/blob/main/files/dSprites_scale_x_pos_y_pos.gif?raw=true" width="300" height="300"/>|<img src="https://github.com/tiwarylab/DynamicsAE/blob/main/files/dSprites_3_VAE_z_scale_movie1.gif?raw=true" width="300" height="300"/>|<img src="https://github.com/tiwarylab/DynamicsAE/blob/main/files/dSprites_3_DynAE_z_scale_movie.gif?raw=true" width="300" height="300"/>

DynamicsAE: A deep learning-based framework to uniquely identify an uncorrelated, isometric and meaningful latent representation.

Please read and cite this manuscript when using DynamicsAE: https://arxiv.org/abs/2209.00905. Here is an implementation of DynamicsAE in Pytorch. A demonstrative colab notebook can be found [here](https://github.com/tiwarylab/DynamicsAE/blob/main/DynAE_Demo.ipynb).

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