import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from core.gaussian import MultivariateStandardGaussian


def create_degrees(n_inputs, n_hiddens, input_order, mode):
    """
    (Copied from the official codebase; I only removed the dependency on rng and changed it to numpy random.)

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

    def __init__(self, data_dim, hidden_dims, multiplier_max=10, input_order="sequential"):
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
        self.pre_one_over_std_layer = MaskedLinear(weight_masks[-1], hidden_dims[-1], data_dim)

        # base distribution

        self.base_dist = MultivariateStandardGaussian(D=data_dim)

        # store info

        self.data_dim = data_dim
        self.degrees = degrees
        self.multiplier_max = multiplier_max

    def calc_mean_and_pre_one_over_std(self, x):
        """
        x: (bs, D)
        h: (bs, H)
        mu: (bs, D)
        alpha: (bs, D)
        """
        h = self.hidden(x)
        return self.mean_layer(h), self.pre_one_over_std_layer(h)

    def calc_u_and_logabsdet(self, x):
        """Only call this method directly when stacking GaussianMADEs into an MAF"""
        mean, pre_one_over_std = self.calc_mean_and_pre_one_over_std(x)
        one_over_std = F.sigmoid(pre_one_over_std) * self.multiplier_max
        u = (x - mean) * one_over_std
        logabsdet = one_over_std.log().sum(dim=1)
        return u, logabsdet

    def log_prob(self, x):
        u, logabsdet = self.calc_u_and_logabsdet(x)
        log_prob_under_u = self.base_dist.log_prob(u)
        log_prob = log_prob_under_u + logabsdet
        return log_prob

    def sample(self, n, u=None):

        if u is None:
            u = self.base_dist.sample(n)

        with torch.no_grad():

            x = torch.zeros(n, self.data_dim)

            # if isinstance(self.input_order, str):
            #     if self.input_order == "sequential":
            #         d_iterator = range(self.data_dim)
            #     else:
            #         raise ValueError(f"{self.input_order} is not a recognized input order")
            # elif isinstance(self.input_order, np.ndarray):
            #     d_iterator = self.input_order - 1
            # else:
            #     raise ValueError(f"{self.input_order} does not belong to a recognized type")

            for d in self.degrees[0] - 1:

                # full forward pass
                mean, pre_one_over_std = self.calc_mean_and_pre_one_over_std(x)  # (n, D)

                # select the parameters for the d-th dimension
                mean, pre_one_over_std = mean[:, d], pre_one_over_std[:, d]

                # 1/std = (exp^(log(1/std^2)))^0.5 = exp(0.5 * log(1/std^2))
                # std = (1/std)^(-1) = exp(0.5 * log(1/std^2))^(-1) = exp(- 0.5 * log(1/std^2))

                std = (1 + torch.exp(-pre_one_over_std)) / self.multiplier_max  # 1/(multiplier_max * sigmoid)
                x_d = u[:, d] * std + mean

                # store samples for the d-th dimension into x
                x[:, d] = x_d

            return x


half_log_2pi = 0.5 * torch.log(torch.Tensor([2.]) * torch.pi)


def one_dim_mog_loglik(x, mean, log_precision, log_mixing_coeff):
    """
    Compute the log likelihood of a one-dimensional mixture of Gaussians.

    :param x: ()
    :param mean: (number of components)
    :param log_precision: (number of components)
    :param log_mixing_coeff: (number of components)
    :return: ()
    """
    return torch.logsumexp(
        log_mixing_coeff + 0.5 * log_precision - half_log_2pi - 0.5 * (x - mean).pow(2) * torch.exp(log_precision),
        dim=0
    )


# x: (bs, D)
# mu: (bs, D, C)
# log_std: (bs, D, C)
# log_pi: (bs, D, C)

one_dim_mog_loglik_batch = torch.vmap(torch.vmap(one_dim_mog_loglik, (0, 0, 0, 0), 0), (0, 0, 0, 0), 0)


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

        self.final_mask = weight_masks[-1].unsqueeze(-1)  # (data_dim, hidden_dims[-1], 1)

        fan_in = hidden_dims[-1]

        self.mean_W = nn.Parameter(torch.randn(data_dim, hidden_dims[-1], num_components) / fan_in)
        self.mean_b = nn.Parameter(torch.randn(data_dim, num_components))

        self.log_precision_W = nn.Parameter(torch.randn(data_dim, hidden_dims[-1], num_components) / fan_in)
        self.log_precision_b = nn.Parameter(torch.randn(data_dim, num_components))

        self.logit_mixing_coeff_W = nn.Parameter(torch.randn(data_dim, hidden_dims[-1], num_components) / fan_in)
        self.logit_mixing_coeff_b = nn.Parameter(torch.randn(data_dim, num_components))

        # store useful info

        self.data_dim = data_dim
        self.degrees = degrees
        self.formula = 'bi,idc->bdc'

    def calc_mean_and_log_precision_and_log_mixing_coeff(self, x):
        """
        x: (bs, D)
        h: (bs, H)

        mean: (bs, D, C)
        log_precision: (bs, D, C)
        log_mixing_coeff: (bs, D, C)
        """

        h = self.hidden(x)

        mean = torch.einsum(
            self.formula,  # 'bi,idc->bdc' represents (bs, H) @ (H, D, C) => (bs, D, C)
            h,  # (bs, H)
            torch.transpose(self.mean_W * self.final_mask, 0, 1)  # (D, H, C) =(transpose)=> (H, D, C)
        ) + self.mean_b

        log_precision = torch.einsum(
            self.formula,
            h,
            torch.transpose(self.log_precision_W * self.final_mask, 0, 1)
        ) + self.log_precision_b

        logit_mixing_coeff = torch.einsum(
            self.formula,
            h,
            torch.transpose(self.logit_mixing_coeff_W * self.final_mask, 0, 1)
        ) + self.logit_mixing_coeff_b

        log_mixing_coeff = F.log_softmax(logit_mixing_coeff, dim=2)

        return mean, log_precision, log_mixing_coeff

    def log_prob(self, x):
        mean, log_precision, log_mixing_coeff = self.calc_mean_and_log_precision_and_log_mixing_coeff(x)
        return one_dim_mog_loglik_batch(x, mean, log_precision, log_mixing_coeff).sum(dim=1)  # interpret dim 1 as event

    def sample(self, n):
        """
        Not easy to do reparametrized sampling for mixture of Gaussians, so samples here are not differentiable

        :param n: number of samples to collect
        :return: samples: (ns, data_Dim)
        """

        with torch.no_grad():

            x = torch.zeros(n, self.data_dim)

            for d in self.degrees[0] - 1:

                # full forward pass
                mean, log_precision, log_mixing_coeff = \
                    self.calc_mean_and_log_precision_and_log_mixing_coeff(x)  # (n, D, C)

                # select the parameters for the d-th dimension
                mean, log_precision, log_mixing_coeff = \
                    mean[:, d, :], log_precision[:, d, :], log_mixing_coeff[:, d, :]  # (n, C)

                # ancestral sampling
                comp_indices = Categorical(probs=log_mixing_coeff.exp()).sample()  # (n, )
                mean_selected = mean.gather(1, comp_indices.reshape(-1, 1)).reshape(-1)  # (n, )
                log_precision_selected = log_precision.gather(1, comp_indices.reshape(-1, 1)).reshape(-1)  # (n, )

                # 1/std = (exp^(log(1/std^2)))^0.5 = exp(0.5 * log(1/std^2))
                # std = (1/std)^(-1) = exp(0.5 * log(1/std^2))^(-1) = exp(- 0.5 * log(1/std^2))

                x_d = Normal(
                    loc=mean_selected,
                    scale=torch.exp(torch.min(-0.5 * log_precision_selected, torch.tensor([10.])))
                ).sample()  # (n, ), clipping as in as original theano code

                # store samples for the d-th dimension into x
                x[:, d] = x_d

            return x
