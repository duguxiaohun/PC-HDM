import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.transforms import TanhTransform


@torch.no_grad()
def soft_update(target_net, current_net, tau: float):
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
        tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

def generate_embedding(dim, device='cuda'):
    return torch.randn(dim, device=device)

def add_embedding(state, embedding, p=0.5, modes=[]):
    num_state = state.shape[0]
    new_embedding = embedding.unsqueeze(0).repeat(num_state, 1)
    assert new_embedding.shape == (num_state, embedding.shape[0])

    s = int(num_state * p)
    if len(modes) != 0:

       if s % len(modes) != 0:
           e_s = [s // len(modes) for i in range(len(modes))]
           e_s[0] = e_s[0] + s % len(modes)
       else:
           e_s = [s // len(modes) for i in range(len(modes))]
       indices = torch.arange(s, device=state.device)
       embeds = []
       for i in range(len(modes)):
           embeds.append(modes[i].unsqueeze(0).repeat(e_s[i], 1))
       new_embedding[indices] = torch.cat(embeds)
    else:

        indices = np.random.choice(num_state, size=s, replace=False)
        indices = torch.tensor(indices).to(state.device)
        new_embedding[indices] = torch.zeros_like(embedding, device=embedding.device)

    new_state = torch.cat([state, new_embedding], dim=1)
    assert new_state.shape == (num_state, state.shape[1] + embedding.shape[0])
    return new_state


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        try:
            self.base_dist = pyd.Normal(loc, scale)
        except (AssertionError, ValueError) as e:
            raise ValueError("Invalid parameters for SquashedNormal") from e
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        return self.base_dist.entropy()


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):


        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):


        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class RunningMeanStd:

    def __init__(self, epsilon=1e-4, shape=(), device='cuda'):
        self.device = device
        self.mean = torch.zeros(shape, device=self.device)
        self.var = torch.ones(shape, device=self.device)
        self.epsilon = epsilon
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x):
        out = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return out

    def unnormalize(self, x):
        out = x * torch.sqrt(self.var + self.epsilon) + self.mean
        return out

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def get_states(self, device=None):
        if device is not None:
            return self.mean.to(device), self.var.to(device), self.epsilon
        else:
            return self.mean, self.var, self.epsilon

    def load_state_dict(self, info):
        self.mean = info[0]
        self.var = info[1]
        self.count = info[2]
