"""
DynamicsAE: A deep learning-based framework to uniquely identify an uncorrelated,
isometric and meaningful latent representation. Code maintained by Dedi.

Read and cite the following when using this method:
https://arxiv.org/abs/2209.00905
"""
import torch
from torch import nn
import numpy as np
import os
import time
from models import architectures
import utils
        
# --------------------
# Model
# --------------------

class VAE(nn.Module):

    def __init__(self, encoder_type, decoder_type, z_dim, output_shape, data_shape, device, neuron_num1=16,
                 neuron_num2=16):

        super(VAE, self).__init__()
        if encoder_type == 'fc_encoder':
            self.encoder_type = 'fc_encoder'
            self.model_encoder = architectures.fc_encoder(2*z_dim, data_shape, device, neuron_num1)
        elif encoder_type == 'conv_encoder':
            self.encoder_type = 'conv_encoder'
            self.model_encoder = architectures.conv_encoder(2*z_dim, data_shape, device, neuron_num1)
        else:
            raise NotImplementedError

        if decoder_type == 'fc_decoder':
            self.decoder_type = 'fc_decoder'
            self.model_decoder = architectures.fc_decoder(z_dim, output_shape, device, neuron_num2)
        elif decoder_type == 'deconv_decoder':
            self.decoder_type = 'deconv_decoder'
            self.model_decoder = architectures.deconv_decoder(z_dim, output_shape, device, neuron_num2)
        else:
            raise NotImplementedError

        self.z_dim = z_dim
        self.output_shape = output_shape

        self.neuron_num1 = neuron_num1
        self.neuron_num2 = neuron_num2

        self.data_shape = data_shape

        self.eps = 1e-10
        self.device = device

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu

    def encode(self, inputs):
        h = self.model_encoder.encoder_input_layer(inputs)
        enc = self.model_encoder.encoder(h)
        if self.encoder_type == 'conv_encoder':
            enc = torch.flatten(enc, start_dim=1)

        output = self.model_encoder.encoder_output(enc)

        z_mean = output[:, :self.z_dim]
        z_logvar = output[:, self.z_dim:]

        return z_mean, z_logvar

    def decode(self, z):
        h = self.model_decoder.decoder_input_layer(z)
        if self.decoder_type == 'deconv_decoder':
            h = h.reshape(-1, 64, 4, 4)
        dec = self.model_decoder.decoder(h)

        outputs = self.model_decoder.decoder_output(dec)

        return outputs

    def forward(self, data):

        z_mean, z_logvar = self.encode(data)

        z_sample = self.reparameterize(z_mean, z_logvar)

        outputs = self.decode(z_sample)

        return outputs, z_sample, z_mean, z_logvar

    def calculate_loss(self, data_inputs, target_data, beta=1.0):

        # pass through VAE
        varibles_list0 = self.forward(data_inputs)

        outputs = varibles_list0[0]

        z_mean = varibles_list0[2]
        z_logvar = varibles_list0[3]

        KL_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim=1), dim=0)

        # MSE Reconstruction loss is used
        reconstruction_error = torch.sum(torch.square(target_data - outputs).flatten(start_dim=1), dim=1).mean()

        # loss = reconstruction_error + beta*entropy
        loss = reconstruction_error + beta * KL_loss

        return loss, reconstruction_error.detach(), KL_loss.detach()

    def train_model(self, beta, input_data_list, train_past_data, train_target_data, test_past_data, test_target_data,
              learning_rate, lr_scheduler_step_size, lr_scheduler_gamma,
              max_epochs, batch_size, output_path, log_interval, SaveTrainingProgress, index):
        self.train()

        step = 0
        start = time.time()
        log_path = output_path + '_train.log'
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        model_path = output_path + "cpt" + str(index) + "/VAE"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        epoch = 0

        # generate the optimizer and scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size,
                                                    gamma=lr_scheduler_gamma)

        while epoch < max_epochs:

            # move to device
            train_permutation = torch.randperm(len(train_past_data)).to(self.device)
            test_permutation = torch.randperm(len(test_past_data)).to(self.device)

            for i in range(0, len(train_past_data), batch_size):
                step += 1

                if i + batch_size > len(train_past_data):
                    break

                train_indices = train_permutation[i:i + batch_size]

                batch_inputs, batch_outputs = utils.sample_minibatch(train_past_data, train_target_data, train_indices, self.device)

                loss, reconstruction_error, kl_loss = self.calculate_loss(batch_inputs, batch_outputs, beta)

                # Stop if NaN is obtained
                if (torch.isnan(loss).any()):
                    print(loss)
                    return True

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 500 == 0:
                    train_time = time.time() - start

                    print(
                        "Iteration %i:\tTime %f s\nLoss (train) %f\tKL loss (train): %f\n"
                        "Reconstruction loss (train) %f" % (
                            step, train_time, loss, kl_loss, reconstruction_error))
                    print(
                        "Iteration %i:\tTime %f s\nLoss (train) %f\tKL loss (train): %f\n"
                        "Reconstruction loss (train) %f" % (
                            step, train_time, loss, kl_loss, reconstruction_error), file=open(log_path, 'a'))
                    j = i % len(test_permutation)

                    test_indices = test_permutation[j:j + batch_size]

                    batch_inputs, batch_outputs = utils.sample_minibatch(test_past_data, test_target_data, test_indices, self.device)

                    loss, reconstruction_error, kl_loss = self.calculate_loss(batch_inputs, batch_outputs, beta)

                    print(
                        "Loss (test) %f\tKL loss (test): %f\n"
                        "Reconstruction loss (test) %f" % (
                            loss, kl_loss, reconstruction_error))
                    print(
                        "Loss (test) %f\tKL loss (test): %f\n"
                        "Reconstruction loss (test) %f" % (
                            loss, kl_loss, reconstruction_error), file=open(log_path, 'a'))

            epoch += 1

            if SaveTrainingProgress:
                if epoch % log_interval == 0:
                    self.eval()
                    for i in range(len(input_data_list)):
                        self.save_traj_results(input_data_list[i], batch_size, output_path + '_epoch%d' % epoch,
                                               False, i, index)
                    self.train()

                    torch.save({'epoch': epoch,
                                'state_dict': self.state_dict()},
                               model_path + '_%d_cpt.pt' % epoch)

                    torch.save({'optimizer': optimizer.state_dict()},
                               model_path + '_%d_optim_cpt.pt' % epoch)

            scheduler.step()
            if scheduler.gamma < 1:
                print("Update lr to %f" % (optimizer.param_groups[0]['lr']))
                print("Update lr to %f" % (optimizer.param_groups[0]['lr']), file=open(log_path, 'a'))

            print("\nEpoch: %d\n" % (epoch))
            print("\nEpoch: %d\n" % (epoch), file=open(log_path, 'a'))

        # output the saving path
        total_training_time = time.time() - start
        print("Total training time: %f" % total_training_time)
        print("Total training time: %f" % total_training_time, file=open(log_path, 'a'))
        # save model
        torch.save({'step': step,
                    'state_dict': self.state_dict()},
                   model_path + '_%d_cpt.pt' % step)
        torch.save({'optimizer': optimizer.state_dict()},
                   model_path + '_%d_optim_cpt.pt' % step)

        torch.save({'step': step,
                    'state_dict': self.state_dict()},
                   model_path + '_final_cpt.pt')
        torch.save({'optimizer': optimizer.state_dict()},
                   model_path + '_final_optim_cpt.pt')

        return False

    def output_final_result(self, train_past_data, train_target_data,
                            test_past_data, test_target_data, batch_size, output_path,
                            path, beta, learning_rate, index=0):

        final_result_path = output_path + '_final_result' + str(index) + '.npy'
        os.makedirs(os.path.dirname(final_result_path), exist_ok=True)

        final_result = []
        # output the result

        loss, reconstruction_error, kl_loss = [0 for i in range(3)]

        for i in range(0, len(train_past_data), batch_size):
            train_indices = range(i, min(i + batch_size, len(train_past_data)))

            batch_inputs, batch_outputs = utils.sample_minibatch(train_past_data, train_target_data, train_indices, self.device)
            loss1, reconstruction_error1, kl_loss1 = self.calculate_loss(batch_inputs, batch_outputs, beta)

            with torch.no_grad():
                loss += loss1 * len(batch_inputs)
                reconstruction_error += reconstruction_error1 * len(batch_inputs)
                kl_loss += kl_loss1 * len(batch_inputs)

        # output the result
        loss /= len(train_past_data)
        reconstruction_error /= len(train_past_data)
        kl_loss /= len(train_past_data)

        final_result += [loss.data.cpu().numpy(), reconstruction_error.cpu().data.numpy(), kl_loss.cpu().data.numpy()]
        print(
            "Final: %d\nLoss (train) %f\tKL loss (train): %f\n"
            "Reconstruction loss (train) %f" % (
                index, loss, kl_loss, reconstruction_error))
        print(
            "Final: %d\nLoss (train) %f\tKL loss (train): %f\n"
            "Reconstruction loss (train) %f" % (
                index, loss, kl_loss, reconstruction_error),
            file=open(path, 'a'))

        loss, reconstruction_error, kl_loss = [0 for i in range(3)]

        for i in range(0, len(test_past_data), batch_size):
            test_indices = range(i, min(i + batch_size, len(test_past_data)))

            batch_inputs, batch_outputs = utils.sample_minibatch(test_past_data, test_target_data, test_indices, self.device)

            loss1, reconstruction_error1, kl_loss1 = self.calculate_loss(batch_inputs, batch_outputs, beta)

            with torch.no_grad():
                loss += loss1 * len(batch_inputs)
                reconstruction_error += reconstruction_error1 * len(batch_inputs)
                kl_loss += kl_loss1 * len(batch_inputs)

        with torch.no_grad():
            # output the result
            loss /= len(test_past_data)
            reconstruction_error /= len(test_past_data)
            kl_loss /= len(test_past_data)

            final_result += [loss.cpu().data.numpy(), reconstruction_error.cpu().data.numpy(),
                             kl_loss.cpu().data.numpy()]
            print(
                "Loss (test) %f\tKL loss (train): %f\n"
                "Reconstruction loss (test) %f"
                % (loss, kl_loss, reconstruction_error))
            print(
                "Loss (test) %f\tKL loss (train): %f\n"
                "Reconstruction loss (test) %f"
                % (loss, kl_loss, reconstruction_error), file=open(path, 'a'))

            print("Beta: %f\t Learning_rate: %f" % (
                beta, learning_rate))
            print("Beta: %f\t Learning_rate: %f" % (
                beta, learning_rate),
                  file=open(path, 'a'))

            final_result = np.array(final_result)
            np.save(final_result_path, final_result)

    @torch.no_grad()
    def save_traj_results(self, inputs, batch_size, path, SaveTrajResults, traj_index=0, index=1):

        all_prediction = []
        all_z_sample = []
        all_z_mean = []

        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size].to(self.device)

            # pass through VAE
            prediction, z_sample, z_mean, z_logvar = self.forward(batch_inputs)

            all_prediction += [prediction.cpu()]
            all_z_sample += [z_sample.cpu()]
            all_z_mean += [z_mean.cpu()]

        all_prediction = torch.cat(all_prediction, dim=0)
        all_z_sample = torch.cat(all_z_sample, dim=0)
        all_z_mean = torch.cat(all_z_mean, dim=0)

        mean_representation_path = path + '_traj%d_mean_representation' % (traj_index) + str(index) + '.npy'

        os.makedirs(os.path.dirname(mean_representation_path), exist_ok=True)

        np.save(mean_representation_path, all_z_mean.cpu().data.numpy())

        if SaveTrajResults:
            prediction_path = path + '_traj%d_data_prediction' % (traj_index) + str(index) + '.npy'
            representation_path = path + '_traj%d_representation' % (traj_index) + str(index) + '.npy'

            np.save(prediction_path, all_prediction.cpu().data.numpy())
            np.save(representation_path, all_z_sample.cpu().data.numpy())