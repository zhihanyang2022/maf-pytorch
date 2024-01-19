import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from mades import MADE, MADE_MOG
from batch_norm import BatchNorm


class MAFBase(nn.Module):

    def __init__(self, data_dim, hidden_dims, num_ar_layers, alternate_input_order=True):
        super().__init__()

        self._current_input_order = np.arange(1, data_dim + 1)

        layers = []

        for _ in range(num_ar_layers):

            layers.append(MADE(data_dim, hidden_dims, input_order=self._current_input_order))
            layers.append(BatchNorm(data_dim))  # insert batch norm after every autoregressive layer

            if alternate_input_order:
                self._current_input_order = self._current_input_order[::-1]

        self.layers = nn.ModuleList(layers)

        self.base_dist = None

    def get_ms_and_vs(self, x):
        ms, vs = [], []
        temp = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BatchNorm):
                temp, m, v = layer.calc_u_and_logabsdet(temp, return_m_and_v=True)
                ms.append(m)
                vs.append(v)
            else:
                temp, _ = layer.calc_u_and_logabsdet(temp)
        return ms, vs

    def log_prob(self, x, ms=None, vs=None):
        log_prob = torch.zeros(x.shape[0])
        temp = x
        for i, layer in enumerate(self.layers):
            if (ms is not None) and (vs is not None) and isinstance(layer, BatchNorm):
                temp, logabsdet = layer.calc_u_and_logabsdet(temp, m=ms[i // 2], v=vs[i // 2])
            else:
                temp, logabsdet = layer.calc_u_and_logabsdet(temp)
            log_prob += logabsdet
        log_prob += self.base_dist.log_prob(temp)
        return log_prob

    def sample(self, x):
        pass


class MAF(MAFBase):

    """A stack of GaussianMADEs with the final u's modelled by a standard Gaussian"""

    def __init__(self, data_dim, hidden_dims, num_ar_layers, alternate_input_order=True):
        super().__init__(data_dim, hidden_dims, num_ar_layers, alternate_input_order)
        self.base_dist = Independent(Normal(torch.zeros(data_dim), torch.ones(data_dim)), 1)


class MAF_MOG(MAFBase):

    """A stack of GaussianMADEs with the final u's modelled by a MixtureOfGaussiansMADE"""

    def __init__(self, data_dim, hidden_dims, num_ar_layers, num_components, alternate_input_order=True):
        super().__init__(data_dim, hidden_dims, num_ar_layers, alternate_input_order)
        self.base_dist = MADE_MOG(
            data_dim, hidden_dims, num_components, self._current_input_order
        )
