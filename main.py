"""
DynamicsAE: A deep learning-based framework to uniquely identify an uncorrelated,
isometric and meaningful latent representation. Code maintained by Dedi.

Read and cite the following when using this method:
https://arxiv.org/abs/2209.00905
"""
import numpy as np
import torch
import os
import sys
import configparser
import json
import random

from models.VAE import VAE
from models.DynamicsAE import DynamicsAE
from models.SDEVAE import SDEVAE
from models.SWAE import SWAE
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")

# For reproducibility
# torch.use_deterministic_algorithms(True)

def main():
    # Settings
    # ------------------------------------------------------------------------------
    # By default, we save all the results in subdirectories of the following path.
    base_path = "Results"

    # If there is a configuration file, import the configuration file
    # Otherwise, an error will be reported
    if '-config' in sys.argv:
        config = configparser.ConfigParser(allow_no_value=True)

        config.read(sys.argv[sys.argv.index('-config') + 1])

        # Model parameters
        
        # By default, we use all the all the data to train and test our model
        t0 = 0
        # By default, set dt = 0 like normal auto-encoder
        dt = 0
        
        # Dimension of RC or bottleneck
        RC_dim_list = json.loads(config.get("Model Parameters","d"))
        
        # Encoder type (fc_encoder or conv_encoder)
        if config.get("Model Parameters","encoder_type")=='conv_encoder':
            encoder_type = 'conv_encoder'
        else:
            encoder_type = 'fc_encoder'

        # Decoder type (fc_decoder or deconv_decoder)
        if config.get("Model Parameters", "decoder_type") == 'deconv_decoder':
            decoder_type = 'deconv_decoder'
        else:
            decoder_type = 'fc_decoder'

        # Number of nodes in each hidden layer of the encoder
        neuron_num1_list = json.loads(config.get("Model Parameters", "neuron_num1"))
        # Number of nodes in each hidden layer of the encoder
        neuron_num2_list = json.loads(config.get("Model Parameters", "neuron_num2"))

        # Model (VAE or DynamicsAE)
        if config.get("Model Parameters", "model") == 'DynamicsAE':
            model_type = 'DynamicsAE'
            model_path = os.path.join(base_path, "DynAE")
        elif config.get("Model Parameters", "model") == 'SDEVAE':
            model_type = 'SDEVAE'
            model_path = os.path.join(base_path, "SDEVAE")
        elif config.get("Model Parameters", "model") == 'SWAE':
            model_type = 'SWAE'
            model_path = os.path.join(base_path, "SWAE")
        else:
            model_type = 'VAE'
            model_path = os.path.join(base_path, "VAE")

        if model_type == 'DynamicsAE':
            # Number of projections to approximate sliced wasserstein distance
            projection_num = int(config.get("Model Parameters", "projection_num"))

            # Minimal distances between cluster centers
            min_dist = float(config.get("Model Parameters", "min_dist"))

            # Bias factor for resampling
            bias_factor = float(config.get("Model Parameters", "bias_factor"))

            # Prior dynamics (only support Langevin for now)
            if config.get("Model Parameters", "prior_dynamics") == 'Hamiltonian':
                prior_dynamics = 'Hamiltonian'
            else:
                prior_dynamics = 'Langevin'

        if model_type == 'SWAE':
            # Number of projections to approximate sliced wasserstein distance
            projection_num = int(config.get("Model Parameters", "projection_num"))

        if model_type == 'SDEVAE':
            # Number of projections to approximate sliced wasserstein distance
            nu = float(config.get("Model Parameters", "nu"))

        # Training parameters
        batch_size = int(config.get("Training Parameters","batch_size"))

        # Number of epochs with the change of the state population smaller than the threshold after which this iteration of training finishes
        max_epochs = int(config.get("Training Parameters","max_epochs"))

        # By default, we save the model every 1 epoch
        log_interval = int(config.get("Training Parameters","log_interval"))
        
        # Period of learning rate decay
        lr_scheduler_step_size = int(config.get("Training Parameters","lr_scheduler_step_size"))

        # Multiplicative factor of learning rate decay. Default: 1 (No learning rate decay)
        lr_scheduler_gamma = float(config.get("Training Parameters","lr_scheduler_gamma"))

        # learning rate of Adam optimizer for IB
        learning_rate_list = json.loads(config.get("Training Parameters","learning_rate"))
        
        # Hyper-parameter beta
        beta_list = json.loads(config.get("Training Parameters","beta"))

        # Import data

        # Path to the input trajectory data
        input_data_path = config.get("Data","input_data")
        if input_data_path[0]=='[' and input_data_path[-1]==']':
            input_data_path = input_data_path.replace('[','').replace(']','')
            input_data_path = input_data_path.split(',')

            # Load the data
            input_data_list = [np.load(file_path) for file_path in input_data_path]
            
            input_data_list = [torch.from_numpy(input_data).float().to(default_device)\
                              for input_data in input_data_list]
        
        # Path to the target data
        target_data_path = config.get("Data","target_data")
        if target_data_path[0]=='[' and target_data_path[-1]==']':
            target_data_path = target_data_path.replace('[','').replace(']','')
            target_data_path = target_data_path.split(',')
            
            target_data_list = [torch.from_numpy(np.load(file_path)).float().to(default_device) for file_path in target_data_path]
            
        
        assert len(input_data_list)==len(target_data_list)

        # Other controls

        # Random seed
        seed_list = json.loads(config.get("Other Controls","seed"))
        
        # Whether to save trajectory results
        if config.get("Other Controls","SaveTrajResults") == 'True':
            SaveTrajResults = True
        else:
            SaveTrajResults = False

        # Whether to save training progress
        if config.get("Other Controls","SaveTrainingProgress") == 'True':
            SaveTrainingProgress = True
        else:
            SaveTrainingProgress = False

    else:
        print("Please input the config file!")
        return
    
    
    # Train and Test our model
    # ------------------------------------------------------------------------------

    final_result_path = model_path + '_result.dat'
    os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
    if os.path.exists(final_result_path):
        print("Final Result", file=open(final_result_path, 'a'))  # append if already exists
    else:
        print("Final Result", file=open(final_result_path, 'w'))
    
    det = 1

    for RC_dim in RC_dim_list:
        for neuron_num1 in neuron_num1_list:
            for neuron_num2 in neuron_num2_list:
                for beta in beta_list:
                    for learning_rate in learning_rate_list:
                        for seed in seed_list:
                            np.random.seed(seed)
                            torch.manual_seed(seed)
                            random.seed(seed)

                            data_init_list = []
                            for i in range(len(input_data_list)):
                                data_init_list += [utils.data_init(t0, dt, input_data_list[i], target_data_list[i])]

                            data_shape = data_init_list[0][0]

                            train_past_data0 = torch.cat(
                                [data_init_list[i][1] for i in range(len(input_data_list))], dim=0)[::det]
                            train_past_data1 = torch.cat(
                                [data_init_list[i][2] for i in range(len(input_data_list))], dim=0)[::det]
                            train_target_data0 = torch.cat(
                                [data_init_list[i][3] for i in range(len(input_data_list))], dim=0)[::det]
                            train_target_data1 = torch.cat(
                                [data_init_list[i][4] for i in range(len(input_data_list))], dim=0)[::det]

                            test_past_data0 = torch.cat([data_init_list[i][5] for i in range(len(input_data_list))],
                                                        dim=0)[::det]
                            test_past_data1 = torch.cat([data_init_list[i][6] for i in range(len(input_data_list))],
                                                        dim=0)[::det]
                            test_target_data0 = torch.cat(
                                [data_init_list[i][7] for i in range(len(input_data_list))], dim=0)[::det]
                            test_target_data1 = torch.cat(
                                [data_init_list[i][8] for i in range(len(input_data_list))], dim=0)[::det]

                            output_shape = train_target_data0.shape[1:]

                            if model_type == 'DynamicsAE':
                                prior_learning_rate = learning_rate
                                output_path = model_path + "_d=%d_dmin=%.1f_b=%f_lr=%f" \
                                              % (RC_dim, min_dist, beta, learning_rate)

                                AE_model = DynamicsAE(encoder_type, decoder_type, prior_dynamics, RC_dim, output_shape,
                                                      data_shape, projection_num, min_dist, bias_factor, device, neuron_num1, neuron_num2)

                                AE_model.to(device)

                                AE_model.train()

                                train_result = AE_model.train_model(beta, input_data_list, train_past_data0,
                                                                    train_past_data1, train_target_data0,
                                                                    train_target_data1,
                                                                    test_past_data0, test_past_data1, test_target_data0,
                                                                    test_target_data1,
                                                                    learning_rate, prior_learning_rate,
                                                                    lr_scheduler_step_size, lr_scheduler_gamma,
                                                                    batch_size, max_epochs,
                                                                    output_path, log_interval,
                                                                    SaveTrainingProgress, seed)

                                if train_result:
                                    return

                                AE_model.eval()
                                AE_model.output_final_result(train_past_data0, train_past_data1, train_target_data0, train_target_data1,
                                                          test_past_data0, test_past_data1, test_target_data0, test_target_data1,
                                                          batch_size, output_path, final_result_path, beta,
                                                          learning_rate, seed)
                                
                            elif model_type == 'SDEVAE':
                                output_path = model_path + "_d=%d_b=%f_lr=%f" \
                                              % (RC_dim, beta, learning_rate)

                                AE_model = SDEVAE(encoder_type, decoder_type, RC_dim, output_shape, data_shape, nu, device, neuron_num1, neuron_num2)

                                AE_model.to(device)

                                AE_model.train()

                                train_result = AE_model.train_model(beta, input_data_list, train_past_data0,
                                                                    train_past_data1, train_target_data0,
                                                                    train_target_data1,
                                                                    test_past_data0, test_past_data1, test_target_data0,
                                                                    test_target_data1, learning_rate, lr_scheduler_step_size, lr_scheduler_gamma,
                                                                    batch_size, max_epochs,
                                                                    output_path, log_interval,
                                                                    SaveTrainingProgress, seed)

                                if train_result:
                                    return

                                AE_model.eval()
                                AE_model.output_final_result(train_past_data0, train_past_data1, train_target_data0, train_target_data1,
                                                          test_past_data0, test_past_data1, test_target_data0, test_target_data1,
                                                          batch_size, output_path, final_result_path, beta,
                                                          learning_rate, seed)
                                
                            elif model_type == 'SWAE':
                                output_path = model_path + "_d=%d_b=%f_lr=%f" \
                                              % (RC_dim, beta, learning_rate)

                                AE_model = SWAE(encoder_type, decoder_type, RC_dim, output_shape, data_shape, projection_num, device, neuron_num1, neuron_num2)

                                AE_model.to(device)

                                train_result = AE_model.train_model(beta, input_data_list, train_past_data0,
                                                                    train_target_data0,
                                                                    test_past_data0, test_target_data0, learning_rate,
                                                                    lr_scheduler_step_size, lr_scheduler_gamma,
                                                                    max_epochs, batch_size, output_path, log_interval,
                                                                    SaveTrainingProgress, seed)

                                if train_result:
                                    return

                                AE_model.eval()
                                AE_model.output_final_result(train_past_data0, train_target_data0, test_past_data0, test_target_data0,
                                                             batch_size, output_path, final_result_path, beta, learning_rate, seed)

                            else:
                                output_path = model_path + "_d=%d_b=%f_lr=%f" \
                                              % (RC_dim, beta, learning_rate)

                                AE_model = VAE(encoder_type, decoder_type, RC_dim, output_shape, data_shape,
                                              device, neuron_num1, neuron_num2)

                                AE_model.to(device)

                                train_result = AE_model.train_model(beta, input_data_list, train_past_data0,
                                                                    train_target_data0,
                                                                    test_past_data0, test_target_data0, learning_rate,
                                                                    lr_scheduler_step_size, lr_scheduler_gamma,
                                                                    max_epochs, batch_size, output_path, log_interval,
                                                                    SaveTrainingProgress, seed)

                                if train_result:
                                    return

                                AE_model.eval()
                                AE_model.output_final_result(train_past_data0, train_target_data0, test_past_data0, test_target_data0,
                                                             batch_size, output_path, final_result_path, beta, learning_rate, seed)


                            for i in range(len(input_data_list)):
                                AE_model.save_traj_results(input_data_list[i], batch_size, output_path, SaveTrajResults, i, seed)

if __name__ == '__main__':
    
    main()
