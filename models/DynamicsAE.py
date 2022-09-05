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
from itertools import chain
import utils
from models import architectures
        
# --------------------
# Model
# --------------------

class DynamicsAE(nn.Module):

    def __init__(self, encoder_type, decoder_type, prior_dynamics, z_dim, output_shape, data_shape, projection_num, min_dist, bias_factor, device, neuron_num1=16,
                 neuron_num2=16):

        super(DynamicsAE, self).__init__()
        if encoder_type == 'fc_encoder':
            self.encoder_type = 'fc_encoder'
            self.model_encoder = architectures.fc_encoder(z_dim, data_shape, device, neuron_num1)
        elif encoder_type == 'conv_encoder':
            self.encoder_type = 'conv_encoder'
            self.model_encoder = architectures.conv_encoder(z_dim, data_shape, device, neuron_num1)
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

        if prior_dynamics == 'Langevin':
            self.prior_dynamics = 'Langevin'
            # by default, we will aim to learn a latent representation with constant diffusion field
            self.ConstantDiffusionPrior = True

            self.model_prior = architectures.Langevin_prior(z_dim, data_shape, device, neuron_num=32)
        else:
            raise NotImplementedError

        self.z_dim = z_dim
        self.output_shape = output_shape
        self.projection_num = projection_num

        self.min_dist = min_dist

        self.bias_factor = bias_factor

        self.neuron_num1 = neuron_num1
        self.neuron_num2 = neuron_num2

        self.data_shape = data_shape

        self.eps = 1e-10
        self.device = device

    def encode(self, inputs):
        h = self.model_encoder.encoder_input_layer(inputs)
        enc = self.model_encoder.encoder(h)
        if self.encoder_type == 'conv_encoder':
            enc = torch.flatten(enc, start_dim=1)

        z = self.model_encoder.encoder_output(enc)

        return z

    def decode(self, z):
        h = self.model_decoder.decoder_input_layer(z)
        if self.decoder_type == 'deconv_decoder':
            h = h.reshape(-1, 64, 4, 4)
        dec = self.model_decoder.decoder(h)

        outputs = self.model_decoder.decoder_output(dec)

        return outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu

    def forward(self, data):
        z = self.encode(data)

        outputs = self.decode(z)

        return outputs, z

    def generate_evolved_samples(self, z0, T):

        if self.prior_dynamics == "Langevin":
            # evolve the original samples using T-step Langevin flow
            samples = z0.clone()
            for i in range(T):
                    samples = self.Langevin_flow(samples)
        else:
            raise NotImplementedError

        return samples

    def Langevin_flow(self, z0):

        z0_detach = z0.detach()
        z0_detach.requires_grad = True

        force = self.model_prior.prior_force(z0_detach)

        if self.ConstantDiffusionPrior:
            z_logdiff = torch.zeros_like(self.model_prior.prior_logdiff(z0_detach))
            z_diff = z_logdiff.exp()

            samples = self.reparameterize(z0 + z_diff * force, z_logdiff + np.log(2))

        else:

            z_logdiff = self.model_prior.prior_logdiff(z0_detach)
            z_diff = z_logdiff.exp()
            # calculate the diffusion gradient "Spurious Forces"
            # here instead of directly calculating the diffusion gradient,
            # log diffusion gradient is calculated to stabilize the algorithm
            log_diff_grad = []
            for i in range(self.z_dim):
                logdiff = z_logdiff[:, i]
                log_diff_grad += [torch.autograd.grad(logdiff.sum(), z0_detach, retain_graph=True)[0][:, i]]

            diff_grad = torch.stack(log_diff_grad, dim=-1) * z_diff

            samples = self.reparameterize(z0 + z_diff * force + diff_grad, z_logdiff + np.log(2))

        return samples

    def prior_loss(self, z0, z1):
        if self.prior_dynamics == "Langevin":
            force = self.model_prior.prior_force(z0)
            z_logdiff = self.model_prior.prior_logdiff(z0)

            z_diff = torch.exp(z_logdiff)

            # calculate the diffusion gradient "Spurious Forces"
            # here instead of directly calculating the diffusion gradient,
            # log diffusion gradient is calculated to stabilize the algorithm
            log_diff_grad = []
            for i in range(self.z_dim):
                logdiff = z_logdiff[:, i]
                log_diff_grad += [torch.autograd.grad(logdiff.sum(), z0, retain_graph=True)[0][:, i]]

            diff_grad = torch.stack(log_diff_grad, dim=-1) * z_diff

            log_r_z1_z0 = -0.5 * torch.sum(z_logdiff + (torch.pow(z1 - z0 - z_diff * force - diff_grad, 2)) / (2 * z_diff),
                                           dim=1)

            prior_loss = -log_r_z1_z0
        else:
            raise NotImplementedError

        return prior_loss

    def calculate_representation_loss(self, data_inputs0, data_inputs1, target_data0, target_data1, beta=1.0):

        # pass through VAE
        outputs0, z_mean0 = self.forward(data_inputs0)

        outputs1, z_mean1 = self.forward(data_inputs1)

        encoded_moves = z_mean1 - z_mean0

        # draw samples from prior distribution
        prior_samples1 = self.Langevin_flow(z_mean0).detach()

        prior_moves = prior_samples1 - z_mean0.detach()

        regularization_loss = utils.sliced_wasserstein_distance(encoded_moves, prior_moves,
                                                             self.projection_num, p=2, device=self.device)

        # MSE Reconstruction loss is used
        reconstruction_error = torch.sum(torch.square(target_data0 - outputs0).flatten(start_dim=1),
                                         dim=1).mean() + \
                               torch.sum(torch.square(target_data1 - outputs1).flatten(start_dim=1),
                                         dim=1).mean()

        # loss = reconstruction_error + beta*entropy
        loss = reconstruction_error + beta * regularization_loss

        return loss, reconstruction_error.detach(), regularization_loss.detach()

    def calculate_prior_loss(self, data_inputs0, data_inputs1):
        # pass through VAE
        _, z_mean0 = self.forward(data_inputs0)
        _, z_mean1 = self.forward(data_inputs1)

        # prior loss
        # detach the encoder from the prior loss
        z_mean0_detach = z_mean0.detach()
        z_mean0_detach.requires_grad = True
        z_mean1_detach = z_mean1.detach()

        prior_loss = self.prior_loss(z_mean0_detach, z_mean1_detach)

        prior_loss = prior_loss.mean()

        return prior_loss

    @torch.no_grad()
    def get_cluster_centers(self, train_input_data, test_input_data, batch_size=128, save_centers=False, log_path=None):
        # This function generates the cluster centers from regular space clustering

        if log_path!=None:
            start_time = time.time()

        # obtain the latent representation and its corresponding diffusion matrix
        train_all_z = []
        for i in range(0, len(train_input_data), batch_size):
            batch_inputs = train_input_data[i:i + batch_size].to(self.device)

            # pass through VAE
            z = self.encode(batch_inputs)

            train_all_z += [z.cpu()]

        train_all_z = torch.cat(train_all_z, dim=0)

        test_all_z = []
        for i in range(0, len(test_input_data), batch_size):
            batch_inputs = test_input_data[i:i + batch_size].to(self.device)

            # pass through VAE
            z = self.encode(batch_inputs)
            test_all_z += [z.cpu()]

        test_all_z = torch.cat(test_all_z, dim=0)

        # dicretize the latent space into bins
        cluster_centers = utils.RegSpaceClustering(train_all_z, self.min_dist)

        # obtain the cluster labels
        train_distance_matrix = torch.sqrt((torch.square(train_all_z.unsqueeze(1) - cluster_centers.unsqueeze(0))).sum(dim=-1))
        train_cluster_labels = torch.argmin(train_distance_matrix, dim=1)

        test_distance_matrix = torch.sqrt((torch.square(test_all_z.unsqueeze(1) - cluster_centers.unsqueeze(0))).sum(dim=-1))
        test_cluster_labels = torch.argmin(test_distance_matrix, dim=1)

        if log_path!=None:
            elapsed_time = time.time() - start_time
            print('Finished after ' + str(elapsed_time) + 's')
            print('%i cluster centers detected' % len(cluster_centers) + '\n')

            print('Finished after ' + str(elapsed_time) + 's', file=open(log_path, 'a'))
            print('%i cluster centers detected' % len(cluster_centers) + '\n', file=open(log_path, 'a'))

        if save_centers:
            return train_cluster_labels, test_cluster_labels, cluster_centers

        else:
            return train_cluster_labels, test_cluster_labels

    def resampling(self, train_past_data0, test_past_data0, batch_size, save_centers, output_path, log_path, index):
        '''
        Uniformly discretizing the latent space and resampling the dataset based on a well-tempered distribution.
            Args:
                train_past_data0: ndarray containing (n,d)-shaped float data for training
                test_past_data0: ndarray containing (n,d)-shaped float data for test
                save_centers: bool, whether to save cluster centers
                output_path: str
                log_path: str
                index: int, random seed number

            Returns:
                train_indices: the resampled indices of training dataset
                test_indices the resampled indices of test dataset
        '''

        # discretize the latent space into bins using regular clustering
        output_variables = self.get_cluster_centers(train_past_data0, test_past_data0, batch_size, save_centers, log_path)

        train_cluster_labels, test_cluster_labels = output_variables[0], output_variables[1]

        # output z cluster centers
        if save_centers:
            cluster_centers = output_variables[2]
            z_cluster_center_path = output_path + '_z_cluster_centers' + str(index) + '.npy'
            np.save(z_cluster_center_path, cluster_centers.cpu().data.numpy())

        num_cluster = int(torch.max(train_cluster_labels).cpu().numpy()) + 1

        # draw samples based on the bias factor (>=1)
        # n_k' ~ n_k^(1/bias_factor)
        cluster_weights = []
        total_weights = 0
        train_cluster_indices = []
        test_cluster_indices = []

        total_effective_samples = 0

        for k in range(num_cluster):
            train_cluster_indices += [torch.nonzero(train_cluster_labels == k, as_tuple=True)[0]]
            test_cluster_indices += [torch.nonzero(test_cluster_labels == k, as_tuple=True)[0]]

            if len(train_cluster_indices[k]) > batch_size:
                total_effective_samples += len(train_cluster_indices[k])

            weight = np.power(len(train_cluster_indices[k]), 1/self.bias_factor)
            cluster_weights += [weight]
            total_weights += weight

        if total_effective_samples < train_past_data0.shape[0]*0.8:
            print(1.0*total_effective_samples/train_past_data0.shape[0])
            print("Too few samples in each bin! Please increase dmin!")
            raise ValueError

        # create better dataset by resampling from each bin
        train_dataset_indices = []
        test_dataset_indices = []

        for k in range(num_cluster):
            train_dataset_size = int(train_past_data0.shape[0] * cluster_weights[k] / total_weights / batch_size + 1) * batch_size
            test_dataset_size = int(test_past_data0.shape[0] * cluster_weights[k] / total_weights / batch_size + 1) * batch_size

            if len(train_cluster_indices[k]) > train_dataset_size:
                train_dataset_indices += [train_cluster_indices[k][
                                              torch.randperm(len(train_cluster_indices[k]))[
                                              :train_dataset_size].to(
                                                  self.device)]]
            elif len(train_cluster_indices[k]) > batch_size:
                size = train_cluster_indices[k].shape[0] // batch_size * batch_size
                for i in range((train_dataset_size) // train_cluster_indices[k].shape[0]):
                    train_dataset_indices += [
                        train_cluster_indices[k][
                            torch.randperm(len(train_cluster_indices[k]))[:size].to(self.device)]]

            if len(test_cluster_indices[k]) > test_dataset_size:
                test_dataset_indices += [test_cluster_indices[k][
                                             torch.randperm(len(test_cluster_indices[k]))[
                                             :test_dataset_size].to(
                                                 self.device)]]
            elif len(test_cluster_indices[k]) > batch_size:
                size = test_cluster_indices[k].shape[0] // batch_size * batch_size
                for i in range(test_dataset_size // test_cluster_indices[k].shape[0]):
                    test_dataset_indices += [
                        test_cluster_indices[k][torch.randperm(len(test_cluster_indices[k]))[:size].to(self.device)]]

        train_dataset_indices = torch.cat(train_dataset_indices, dim=0).reshape((-1, batch_size))
        test_dataset_indices = torch.cat(test_dataset_indices, dim=0).reshape((-1, batch_size))

        train_indices = train_dataset_indices[
            torch.randperm((train_dataset_indices).shape[0]).to(self.device)].flatten()
        test_indices = test_dataset_indices[
            torch.randperm((test_dataset_indices).shape[0]).to(self.device)].flatten()

        return train_indices, test_indices

    def train_model(self, beta, input_data_list, train_past_data0, train_past_data1, train_target_data0, train_target_data1,
              test_past_data0, test_past_data1, test_target_data0, test_target_data1,
              learning_rate, prior_learning_rate, lr_scheduler_step_size, lr_scheduler_gamma,
              batch_size, max_epochs, output_path, log_interval,
              SaveTrainingProgress, index):
        self.train()

        step = 0
        start = time.time()

        log_path = output_path + '_train.log'

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        model_path = output_path + "cpt" + str(index) + "/DynAE"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        epoch = 0

        # generate the optimizer and scheduler
        optimizer = torch.optim.Adam(chain(self.model_encoder.parameters(), self.model_decoder.parameters()), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size,
                                                    gamma=lr_scheduler_gamma)

        prior_optimizer = torch.optim.Adam(self.model_prior.parameters(), lr=prior_learning_rate)

        while epoch < max_epochs:
            if epoch > 0:
                train_permutation, test_permutation = self.resampling(train_past_data0, test_past_data0, batch_size, False, output_path, log_path, index)
            else:
                train_permutation = torch.randperm((train_past_data0).shape[0]).to(self.device)
                test_permutation = torch.randperm((test_past_data0).shape[0]).to(self.device)

            for i in range(0, len(train_permutation), batch_size):
                step += 1

                if i + batch_size > len(train_permutation):
                    break

                train_indices = train_permutation[i:i + batch_size]

                batch_inputs0, batch_inputs1, batch_outputs0, batch_outputs1 = utils.sample_pairwise_minibatch(
                    train_past_data0, train_past_data1, train_target_data0,
                    train_target_data1, train_indices, self.device)

                loss, reconstruction_error, kl_loss = self.calculate_representation_loss(batch_inputs0, batch_inputs1,
                                                                                         batch_outputs0, batch_outputs1,
                                                                                         beta)

                # Stop if NaN is obtained
                if (torch.isnan(loss).any()):
                    print(loss)
                    return True

                optimizer.zero_grad()
                prior_optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_indices = train_permutation[i:i + batch_size]

                batch_inputs0, batch_inputs1, batch_outputs0, batch_outputs1 = utils.sample_pairwise_minibatch(
                    train_past_data0, train_past_data1, train_target_data0,
                    train_target_data1, train_indices, self.device)

                prior_loss = self.calculate_prior_loss(batch_inputs0, batch_inputs1)

                # Stop if NaN is obtained
                if (torch.isnan(prior_loss).any()):
                    print(prior_loss)
                    return True

                optimizer.zero_grad()
                prior_optimizer.zero_grad()
                prior_loss.backward()
                prior_optimizer.step()

                if step % 500 == 0:

                    train_time = time.time() - start

                    print(
                        "Iteration %i:\tTime %f s\nLoss (train) %f\tRegularization loss (train): %f\n"
                        "Reconstruction loss (train) %f\t Prior loss (train) %f" % (
                            step, train_time, loss, kl_loss, reconstruction_error, prior_loss))
                    print(
                        "Iteration %i:\tTime %f s\nLoss (train) %f\tRegularization loss (train): %f\n"
                        "Reconstruction loss (train) %f\t Prior loss (train) %f" % (
                            step, train_time, loss, kl_loss, reconstruction_error, prior_loss),
                        file=open(log_path, 'a'))
                    j = i % len(test_permutation)
                    if j + batch_size > len(test_permutation):
                        j -= batch_size

                    test_indices = test_permutation[j:j + batch_size]

                    batch_inputs0, batch_inputs1, batch_outputs0, batch_outputs1 = utils.sample_pairwise_minibatch(
                        test_past_data0, test_past_data1, test_target_data0,
                        test_target_data1, test_indices, self.device)

                    loss, reconstruction_error, kl_loss = self.calculate_representation_loss(batch_inputs0,
                                                                                             batch_inputs1,
                                                                                             batch_outputs0,
                                                                                             batch_outputs1,
                                                                                             beta)

                    prior_loss = self.calculate_prior_loss(batch_inputs0, batch_inputs1)

                    print(
                        "Loss (test) %f\tRegularization loss (test): %f\n"
                        "Reconstruction loss (test) %f\t Prior loss (test) %f" % (
                            loss, kl_loss, reconstruction_error, prior_loss))
                    print(
                        "Loss (test) %f\tRegularization loss (test): %f\n"
                        "Reconstruction loss (test) %f\t Prior loss (test) %f" % (
                            loss, kl_loss, reconstruction_error, prior_loss), file=open(log_path, 'a'))

            epoch += 1

            if SaveTrainingProgress:
                if epoch % log_interval == 0:
                    self.eval()
                    for i in range(len(input_data_list)):
                        self.save_traj_results(input_data_list[i], batch_size, output_path + '_epoch%d' % epoch, False, i, index)
                    self.train()

                    torch.save({'epoch': epoch,
                                'state_dict': self.state_dict()},
                               model_path + '_%d_cpt.pt' % epoch)

            scheduler.step()
            if scheduler.gamma < 1:
                print("Update lr to %f" % (optimizer.param_groups[0]['lr']))
                print("Update lr to %f" % (optimizer.param_groups[0]['lr']), file=open(log_path, 'a'))

            print("Epoch: %d\n" % (epoch))
            print("Epoch: %d\n" % (epoch), file=open(log_path, 'a'))

        # output the saving path
        total_training_time = time.time() - start
        print("Total training time: %f" % total_training_time)
        print("Total training time: %f" % total_training_time, file=open(log_path, 'a'))
        # save model
        torch.save({'epoch': epoch,
                    'state_dict': self.state_dict()},
                   model_path + '_final_cpt.pt')

        return False

    def output_final_result(self, train_past_data0, train_past_data1, train_target_data0, train_target_data1, \
                            test_past_data0, test_past_data1, test_target_data0, test_target_data1, \
                            batch_size, output_path, log_path, beta,
                            learning_rate, index=0):

        final_result_path = output_path + '_final_result' + str(index) + '.npy'
        os.makedirs(os.path.dirname(final_result_path), exist_ok=True)

        train_dataset_indices, test_dataset_indices = self.resampling(train_past_data0, test_past_data0, batch_size, True, output_path, log_path, index)

        # output the result
        final_result = []
        loss, reconstruction_error, kl_loss, prior_loss = [0 for i in range(4)]

        for i in range(0, len(train_dataset_indices), batch_size):
            if i + batch_size > len(train_dataset_indices):
                break
            train_indices = train_dataset_indices[i:min(i + batch_size, len(train_past_data0))]

            batch_inputs0, batch_inputs1, batch_outputs0, batch_outputs1 = utils.sample_pairwise_minibatch(
                train_past_data0, train_past_data1, train_target_data0,
                train_target_data1, train_indices, self.device)

            loss1, reconstruction_error1, kl_loss1 = self.calculate_representation_loss(batch_inputs0, batch_inputs1,
                                                                                     batch_outputs0, batch_outputs1,
                                                                                     beta)
            prior_loss1 = self.calculate_prior_loss(batch_inputs0, batch_inputs1)

            with torch.no_grad():
                loss += loss1 * len(batch_inputs0)
                reconstruction_error += reconstruction_error1 * len(batch_inputs0)
                kl_loss += kl_loss1 * len(batch_inputs0)
                prior_loss += prior_loss1 * len(batch_inputs0)

        # output the result
        loss /= len(train_dataset_indices)
        reconstruction_error /= len(train_dataset_indices)
        kl_loss /= len(train_dataset_indices)
        prior_loss /= len(train_dataset_indices)

        final_result += [loss.data.cpu().numpy(), reconstruction_error.cpu().data.numpy(), kl_loss.cpu().data.numpy(),
                         prior_loss.cpu().data.numpy()]
        print(
            "Final: %d\nLoss (train) %f\tRegularization loss (train): %f\n"
            "Reconstruction loss (train) %f\t Prior loss (train) %f" % (
                index, loss, kl_loss, reconstruction_error, prior_loss))
        print(
            "Final: %d\nLoss (train) %f\tRegularization loss (train): %f\n"
            "Reconstruction loss (train) %f\t Prior loss (train) %f" % (
                index, loss, kl_loss, reconstruction_error, prior_loss),
            file=open(log_path, 'a'))

        loss, reconstruction_error, kl_loss, prior_loss = [0 for i in range(4)]

        for i in range(0, len(test_dataset_indices), batch_size):
            if i + batch_size > len(test_dataset_indices):
                break

            test_indices = test_dataset_indices[i:min(i + batch_size, len(test_past_data0))]

            batch_inputs0, batch_inputs1, batch_outputs0, batch_outputs1 = utils.sample_pairwise_minibatch(
                test_past_data0, test_past_data1, test_target_data0,
                test_target_data1, test_indices, self.device)

            loss1, reconstruction_error1, kl_loss1 = self.calculate_representation_loss(batch_inputs0,
                                                                                     batch_inputs1,
                                                                                     batch_outputs0,
                                                                                     batch_outputs1,
                                                                                     beta)

            prior_loss1 = self.calculate_prior_loss(batch_inputs0, batch_inputs1)

            with torch.no_grad():
                loss += loss1 * len(batch_inputs0)
                reconstruction_error += reconstruction_error1 * len(batch_inputs0)
                kl_loss += kl_loss1 * len(batch_inputs0)
                prior_loss += prior_loss1 * len(batch_inputs0)

        with torch.no_grad():
            # output the result
            loss /= len(test_dataset_indices)
            reconstruction_error /= len(test_dataset_indices)
            kl_loss /= len(test_dataset_indices)
            prior_loss /= len(test_dataset_indices)

            final_result += [loss.cpu().data.numpy(), reconstruction_error.cpu().data.numpy(),
                             kl_loss.cpu().data.numpy(), prior_loss.cpu().data.numpy()]
            print(
                "Loss (test) %f\tRegularization loss (train): %f\n"
                "Reconstruction loss (test) %f\t Prior loss (test) %f"
                % (loss, kl_loss, reconstruction_error, prior_loss))
            print(
                "Loss (test) %f\tRegularization loss (train): %f\n"
                "Reconstruction loss (test) %f\t Prior loss (test) %f"
                % (loss, kl_loss, reconstruction_error, prior_loss), file=open(log_path, 'a'))

            print("Beta: %f\t Learning_rate: %f" % (
                beta, learning_rate))
            print("Beta: %f\t Learning_rate: %f" % (
                beta, learning_rate),
                  file=open(log_path, 'a'))

            final_result = np.array(final_result)
            np.save(final_result_path, final_result)
        
    @torch.no_grad()
    def save_traj_results(self, inputs, batch_size, path, SaveTrajResults, traj_index=0, index=1):

        all_prediction = []
        all_z = []

        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size].to(self.device)

            # pass through VAE
            z = self.encode(batch_inputs)
            prediction = self.decode(z)

            all_prediction += [prediction.cpu()]
            all_z += [z.cpu()]

        all_prediction = torch.cat(all_prediction, dim=0)
        all_z = torch.cat(all_z, dim=0)

        representation_path = path + '_traj%d_representation' % (traj_index) + str(index) + '.npy'

        os.makedirs(os.path.dirname(representation_path), exist_ok=True)

        np.save(representation_path, all_z.cpu().data.numpy())

        if SaveTrajResults:
            prediction_path = path + '_traj%d_data_prediction' % (traj_index) + str(index) + '.npy'
            np.save(prediction_path, all_prediction.cpu().data.numpy())



