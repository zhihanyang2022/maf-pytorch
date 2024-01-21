import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gaussian import MultivariateStandardGaussian


def create_degrees(n_inputs, n_hiddens, input_order, mode):
    """
    (Copied from the official codebase; I only removed the dependency on rng and changed it to numpy random)

    Generates a degree for each hidden and input unit. A unit with degree d can only receive input from units with
    degree less than d.

    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param input_order: the order of the inputs; can be 'random', 'sequential', or an array of an explicit order
    :param mode: the strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
    :return: list of degrees
    """

    degrees = []

    # create degrees for inputs
    if isinstance(input_order, str):

        if input_order == 'random':
            degrees_0 = np.arange(1, n_inputs + 1)
            np.random.shuffle(degrees_0)

        elif input_order == 'sequential':
            degrees_0 = np.arange(1, n_inputs + 1)

        else:
            raise ValueError('invalid input order')

    else:
        input_order = np.array(input_order)
        assert np.all(np.sort(input_order) == np.arange(1, n_inputs + 1)), 'invalid input order'
        degrees_0 = input_order
    degrees.append(degrees_0)

    # create degrees for hiddens
    if mode == 'random':
        for N in n_hiddens:
            min_prev_degree = min(np.min(degrees[-1]), n_inputs - 1)
            degrees_l = np.random.randint(min_prev_degree, n_inputs, N)
            degrees.append(degrees_l)

    elif mode == 'sequential':
        for N in n_hiddens:
            degrees_l = np.arange(N) % max(1, n_inputs - 1) + min(1, n_inputs - 1)
            degrees.append(degrees_l)

    else:
        raise ValueError('invalid mode')

    return degrees


def create_masks(degrees):
    masks = []

    for d0, d1 in zip(degrees[:-1], degrees[1:]):
        masks.append(torch.IntTensor(d1.reshape(-1, 1) >= d0.reshape(1, -1)))

    masks.append(torch.IntTensor(degrees[0].reshape(-1, 1) > degrees[-1].reshape(1, -1)))

    return masks


class MaskedLinear(nn.Linear):

    def __init__(self, mask, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask.shape == (self.out_features, self.in_features)
        self.mask = mask

    def forward(self, inp):
        return F.linear(inp, self.weight * self.mask, self.bias)


class MADE(nn.Module):

    """
    This class must be able to do a few things:
    - Computing log prob of a sample (under its own base distribution)
    - Be able to generate (from its own base distribution or otherwise)
    - Be able to compute random numbers to be modelled by later layers and logabsdet
    """

    def __init__(self, data_dim, hidden_dims, input_order="sequential"):
        super().__init__()

        # create degrees and masks

        degrees = create_degrees(data_dim, hidden_dims, input_order=input_order, mode="sequential")
        weight_masks = create_masks(degrees)

        # create masked linear layers

        hidden_layers = [
            MaskedLinear(weight_masks[0], data_dim, hidden_dims[0]),
            nn.ReLU()
        ]

        for i, (h0, h1) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            hidden_layers.append(MaskedLinear(weight_masks[i + 1], h0, h1))
            hidden_layers.append(nn.ReLU())

        self.hidden = nn.Sequential(*hidden_layers)

        # parametrize the output distributions

        self.mean_layer = MaskedLinear(weight_masks[-1], hidden_dims[-1], data_dim)
        self.log_precision_layer = MaskedLinear(weight_masks[-1], hidden_dims[-1], data_dim)

        # base distribution

        self.base_dist = MultivariateStandardGaussian(D=data_dim)

    def calc_mean_and_log_precision(self, x):
        """
        x: (bs, D)
        h: (bs, H)
        mu: (bs, D)
        alpha: (bs, D)
        """
        h = self.hidden(x)
        return self.mean_layer(h), self.log_precision_layer(h)

    def calc_u_and_logabsdet(self, x):
        """
        Only call this method directly when stacking GaussianMADEs into an MAF

        I tried every trick I can think of to make this function as numerically stable as possible.
        (1) Model log_precision instead of log_std to avoid divisions
        (2) Use softplus instead of exp to avoid large gradients
        """
        mean, log_precision = self.calc_mean_and_log_precision(x)
        half_log_precision = 0.5 * log_precision
        one_over_std = F.softplus(half_log_precision)
        u = (x - mean) * one_over_std
        logabsdet = one_over_std.log().sum(dim=1)
        return u, logabsdet

    def log_prob(self, x):
        u, logabsdet = self.calc_u_and_logabsdet(x)
        log_prob_under_u = self.base_dist.log_prob(u)
        log_prob = log_prob_under_u + logabsdet
        return log_prob

    def sample(self, u=None):
        raise NotImplementedError


half_log_2pi = 0.5 * torch.log(torch.Tensor([2.]) * torch.pi)


def mog_1d_loglik(x, mean, log_precision, log_pi):
    """
    Compute the log likelihood of a one-dimensional mixture of Gaussians.

    :param x: ()
    :param mean: (number of components)
    :param log_precision: (number of components)
    :param log_pi: (number of components)
    :return: ()
    """
    return torch.logsumexp(
        log_pi + 0.5 * log_precision - half_log_2pi - 0.5 * ((x - mean) * torch.exp(0.5 * log_precision)) ** 2,
        dim=0
    )


# x: (bs, D)
# mu: (bs, D, C)
# log_std: (bs, D, C)
# log_pi: (bs, D, C)

mog_1d_loglik_batch = torch.vmap(torch.vmap(mog_1d_loglik, (0, 0, 0, 0), 0), (0, 0, 0, 0), 0)


class MADE_MOG(nn.Module):

    def __init__(self, data_dim, hidden_dims, num_components, input_order="sequential"):
        super().__init__()

        # create degrees and masks

        degrees = create_degrees(data_dim, hidden_dims, input_order=input_order, mode="sequential")
        weight_masks = create_masks(degrees)

        # create masked linear layers

        hidden_layers = [
            MaskedLinear(weight_masks[0], data_dim, hidden_dims[0]),
            nn.ReLU()
        ]

        for i, (h0, h1) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            hidden_layers.append(MaskedLinear(weight_masks[i + 1], h0, h1))
            hidden_layers.append(nn.ReLU())

        self.hidden = nn.Sequential(*hidden_layers)

        # parametrize the output distributions
        # empirically, if I initialize the biases to be zeros, training is wayyy slower for some reason, not sure why

        self.final_mask = weight_masks[-1].unsqueeze(-1)

        fan_in = hidden_dims[-1]

        self.mean_W = nn.Parameter(torch.randn(data_dim, hidden_dims[-1], num_components) / fan_in)
        self.mean_b = nn.Parameter(torch.randn(data_dim, num_components))

        self.log_precision_W = nn.Parameter(torch.randn(data_dim, hidden_dims[-1], num_components) / fan_in)
        self.log_precision_b = nn.Parameter(torch.randn(data_dim, num_components))

        self.logit_pi_W = nn.Parameter(torch.randn(data_dim, hidden_dims[-1], num_components) / fan_in)
        self.logit_pi_b = nn.Parameter(torch.randn(data_dim, num_components))

        # store useful info

        self.data_dim = data_dim
        self._formula = 'bi,idc->bdc'

    def calc_mean_and_log_precision_and_log_pi(self, x):
        """
        x: (bs, D)
        h: (bs, H)

        Overall logic (bs, H) @ (H, D, C) + (D, C) => (bs, D, C) + (D, C) => (bs, D, C)

        mu: (bs, D, C)
        alpha: (bs, D, C)
        pi: (bs, D, C)
        """

        h = self.hidden(x)

        mean = \
            torch.einsum(self._formula, h, torch.transpose(self.mean_W * self.final_mask, 0, 1)) + self.mean_b
        log_precision = \
            torch.einsum(self._formula, h, torch.transpose(self.log_precision_W * self.final_mask, 0, 1)) + \
            self.log_precision_b

        # logit_pi differs from log_pi in the sense that logit_pi is the log UNNORMALIZED probabilities

        log_pi = F.log_softmax(
            torch.einsum(self._formula, h, torch.transpose(self.logit_pi_W * self.final_mask, 0, 1)) + self.logit_pi_b,
            dim=2
        )

        return mean, log_precision, log_pi

    def log_prob(self, x):
        mean, log_precision, log_pi = self.calc_mean_and_log_precision_and_log_pi(x)
        return mog_1d_loglik_batch(x, mean, log_precision, log_pi).sum(dim=1)  # interpret dim=1 as an event dimension

    # def sample(self, n):
    #     """
    #     Not easy to do reparametrized sampling for mixture of Gaussians
    #
    #     :param n: number of samples to collect
    #     :return: samples
    #     """
    #
    #     with torch.no_grad():
    #
    #         x = torch.zeros(n, self.data_dim)
    #
    #         for d in range(self.data_dim):
    #
    #             # full forward pass
    #             mu, log_precision, log_pi = self.calc_mu_and_log_precision_and_log_pi(x)  # (n, D, C)
    #
    #             # select the parameters for the d-th dimension
    #             mu, log_precision, log_pi = mu[:, d, :], log_precision[:, d, :], log_pi[:, d, :]  # (n, C)
    #
    #             # ancestral sampling
    #             comp_indices = Categorical(logits=log_pi).sample()  # (n, )
    #             mu_selected = mu.gather(1, comp_indices.reshape(-1, 1)).reshape(-1)  # (n, )
    #             log_std_selected = log_precision.gather(1, comp_indices.reshape(-1, 1)).reshape(-1)  # (n, )
    #             # TODO: change this line of code to scaling
    #             x_d = Normal(loc=mu_selected, scale=log_std_selected.exp() + 1e-5).sample()  # (n, )
    #
    #             # store samples for the d-th dimension into x
    #             x[:, d] = x_d
    #
    #         return x
