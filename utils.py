"""
DynamicsAE: A deep learning-based framework to uniquely identify an uncorrelated,
isometric and meaningful latent representation. Code maintained by Dedi.

Read and cite the following when using this method:
https://arxiv.org/abs/2209.00905
"""
import torch
import numpy as np

def data_init(t0, dt, traj_data, traj_target):
    # This function generates the datasets for training

    assert len(traj_data) == len(traj_target)

    # skip the first t0 data
    past_data0 = traj_data[t0:(len(traj_data) - dt - 1)]
    past_data1 = traj_data[(t0 + 1):(len(traj_data) - dt)]

    target0 = traj_target[(t0 + dt):(len(traj_data) - 1)]
    target1 = traj_target[(t0 + dt + 1):len(traj_data)]

    # data shape
    data_shape = past_data0.shape[1:]

    n_data = len(past_data0)

    # 90% random test/train split
    p = np.random.permutation(n_data)
    past_data0 = past_data0[p]
    past_data1 = past_data1[p]
    target0 = target0[p]
    target1 = target1[p]

    train_past_data0 = past_data0[0: (8 * n_data) // 10]
    test_past_data0 = past_data0[(8 * n_data) // 10:]

    train_past_data1 = past_data1[0: (8 * n_data) // 10]
    test_past_data1 = past_data1[(8 * n_data) // 10:]

    train_target_data0 = target0[0: (8 * n_data) // 10]
    test_target_data0 = target0[(8 * n_data) // 10:]

    train_target_data1 = target1[0: (8 * n_data) // 10]
    test_target_data1 = target1[(8 * n_data) // 10:]

    return data_shape, train_past_data0, train_past_data1, train_target_data0, train_target_data1, \
           test_past_data0, test_past_data1, test_target_data0, test_target_data1

def sample_minibatch(past_data, data_labels, indices, device):
    sample_past_data = past_data[indices].to(device)
    sample_data_labels = data_labels[indices].to(device)

    return sample_past_data, sample_data_labels

def sample_pairwise_minibatch(past_data0, past_data1, target_data0, target_data1, indices, device):
    sample_past_data0 = past_data0[indices].to(device)
    sample_past_data1 = past_data1[indices].to(device)
    sample_target_data0 = target_data0[indices].to(device)
    sample_target_data1 = target_data1[indices].to(device)

    return sample_past_data0, sample_past_data1, sample_target_data0, sample_target_data1

def rand_projections(z_dim, num_samples=50):
    # This function generates `num_samples` random samples from the latent space's unit sphere
    projections = [w / np.sqrt((w**2).sum())
                   for w in np.random.normal(size=(num_samples, z_dim))]
    projections = torch.from_numpy(np.array(projections)).float()
    return projections

# Only used for unweighted samples
def sliced_wasserstein_distance(encoded_samples, prior_samples, projection_num=50, p=2, device='cpu'):
    # This function calculates the sliced-Wasserstein distance between the encoded samples and prior samples

    # derive latent space dimension size from random samples drawn from latent prior distribution
    z_dim = prior_samples.size(-1)

    # generate random projections in latent space
    projections = rand_projections(z_dim, projection_num).to(device)
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projections through the prior distribution random samples
    prior_projections = (prior_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections, dim=0)[0] -
                            torch.sort(prior_projections, dim=0)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()


@torch.no_grad()
def RegSpaceClustering(z_data, min_dist, max_centers=200, batch_size=128):
    '''
    Regular space clustering.
        Args:
            z_data: ndarray containing (n,d)-shaped float data
            max_centers: the maximum number of cluster centers to be determined, integer greater than 0 required
            min_dist: the minimal distances between cluster centers

        Returns:
            center_list: ndarray containing the cluster centers
    '''

    num_observations, d = z_data.shape

    p = np.random.permutation(num_observations)
    data = z_data[p]

    center_list = data[0:1, :].clone()

    i = 1
    while i < num_observations:
        x_active = data[i:i+batch_size, :]
        distances = torch.sqrt((torch.square(center_list.unsqueeze(0) - x_active.unsqueeze(1))).sum(dim=-1))
        indice = torch.nonzero(torch.all(distances > min_dist, dim=-1), as_tuple=True)[0]
        if len(indice) > 0:
            # the first element will be used
            center_list = torch.cat((center_list, x_active[indice[0]].reshape(1, d)), 0)
            i += indice[0]
        else:
            i += batch_size

        if len(center_list) >= max_centers:
            print("Exceed the maximum number of cluster centers!\n")
            print("Please increase dmin!\n")
            raise ValueError

    return center_list
