from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Independent
from torch.distributions import Normal

from ddiffpg.utils.torch_util import SquashedNormal


def create_simple_mlp(in_dim, out_dim, hidden_layers, act=nn.ELU):
    layer_nums = [in_dim, *hidden_layers, out_dim]
    model = []
    for idx, (in_f, out_f) in enumerate(zip(layer_nums[:-1], layer_nums[1:])):
        model.append(nn.Linear(in_f, out_f))
        if idx < len(layer_nums) - 2:
            model.append(act())
    return nn.Sequential(*model)


class MLPNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=None):
        super().__init__()
        if isinstance(in_dim, Sequence):
            in_dim = in_dim[0]
        if hidden_layers is None:
            hidden_layers = [512, 256, 128]
        self.net = create_simple_mlp(in_dim=in_dim,
                                     out_dim=out_dim,
                                     hidden_layers=hidden_layers)

    def forward(self, x):
        return self.net(x)


class DiagGaussianMLPPolicy(MLPNet):
    def __init__(self, state_dim, act_dim, hidden_layers=None,
                 init_log_std=0.):
        super().__init__(in_dim=state_dim,
                         out_dim=act_dim,
                         hidden_layers=hidden_layers)
        self.logstd = nn.Parameter(torch.full((act_dim,), init_log_std))

    def forward(self, x, sample=True):
        return self.get_actions(x, sample=sample)[0]

    def get_actions(self, x, sample=True):
        mean = self.net(x)
        log_std = self.logstd.expand_as(mean)
        std = torch.exp(log_std)
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        if sample:
            actions = action_dist.rsample()
        else:
            actions = mean
        return actions, action_dist

    def get_actions_logprob_entropy(self, state, sample=True):
        actions, action_dist = self.get_actions(state, sample=sample)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return actions, action_dist, log_prob, entropy

    def logprob_entropy(self, state, actions):
        _, action_dist = self.get_actions(state)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return actions, action_dist, log_prob, entropy


class TanhDiagGaussianMLPPolicy(MLPNet):
    def __init__(self, state_dim, act_dim, hidden_layers=None):
        super().__init__(in_dim=state_dim,
                         out_dim=act_dim * 2,
                         hidden_layers=hidden_layers)
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.log_std_min = -5
        self.log_std_max = 5

    def forward(self, state: Tensor, sample: bool = False) -> Tensor:
        return self.get_actions(state, sample=sample)

    def get_actions(self, state: Tensor, sample=True) -> Tensor:
        dist = self.get_action_dist(state)
        if sample:
            actions = dist.rsample()
        else:
            actions = dist.mean
        return actions

    def get_action_dist(self, state: Tensor):
        mu, log_std = self.net(state).chunk(2, dim=-1)
        std = log_std.clamp(self.log_std_min, self.log_std_max).exp()
        dist = SquashedNormal(mu, std)
        return dist

    def get_actions_logprob(self, state: Tensor):
        dist = self.get_action_dist(state)
        actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        return actions, dist, log_prob


class TanhMLPPolicy(MLPNet):
    def forward(self, state):
        return super().forward(state).tanh()


class DoubleQ(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.net_q1 = MLPNet(in_dim=state_dim + act_dim, out_dim=1)
        self.net_q2 = MLPNet(in_dim=state_dim + act_dim, out_dim=1)

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.min(*self.get_q1_q2(state, action))

    def get_q1_q2(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return self.net_q1(input_x), self.net_q2(input_x)

    def get_q1(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return self.net_q1(input_x)


class DistributionalDoubleQ(nn.Module):
    def __init__(self, state_dim, act_dim, v_min=-10, v_max=10, num_atoms=51, device="cuda"):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.device = device
        self.net_q1 = MLPNet(in_dim=state_dim + act_dim, out_dim=num_atoms)
        self.net_q2 = MLPNet(in_dim=state_dim + act_dim, out_dim=num_atoms)
        self.v_min = v_min
        self.v_max = v_max
        self.z_atoms = torch.linspace(v_min, v_max, num_atoms, device=device)

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        Q1, Q2 = self.get_q1_q2(state, action)
        Q1 = torch.sum(Q1 * self.z_atoms.to(self.device), dim=1)
        Q2 = torch.sum(Q2 * self.z_atoms.to(self.device), dim=1)
        return torch.min(Q1, Q2)

    def get_q1_q2(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return torch.softmax(self.net_q1(input_x), dim=1), torch.softmax(self.net_q2(input_x), dim=1)

    def get_q1(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return torch.softmax(self.net_q1(input_x), dim=1)


class DistributionalEnsembleQ(nn.Module):
    def __init__(self, state_dim, act_dim, v_min=-10, v_max=10, num_atoms=51, n_ensemble=3, device="cuda"):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.device = device
        self.n_ensemble = n_ensemble
        self.nets = nn.ModuleList([MLPNet(in_dim=state_dim + act_dim, out_dim=num_atoms) for _ in range(n_ensemble)])

        self.z_atoms = torch.linspace(v_min, v_max, num_atoms, device=device)

    def get_q_mean(self, state: Tensor, action: Tensor, need_q_list=False) -> Tensor:
        Qs = self.get_qs(state, action)
        Q = [torch.sum(Qs[i] * self.z_atoms.to(self.device), dim=1) for i in range(self.n_ensemble)]
        if need_q_list:
            return torch.mean(torch.stack(Q, dim=1), dim=1), Q
        else:
            return torch.mean(torch.stack(Q, dim=1), dim=1)

    def get_qs(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return [torch.softmax(self.nets[i](input_x), dim=1) for i in range(self.n_ensemble)]

    def get_q1(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return torch.softmax(self.nets[0](input_x), dim=1)


class MLPCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.critic = MLPNet(in_dim=state_dim, out_dim=1)

    def forward(self, state: Tensor) -> Tensor:
        return self.critic(state)


class DynamicModel(nn.Module):
    def __init__(self, state_dim, action_dim, n_ensemble):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.n_ensemble = n_ensemble
        self.nets = nn.ModuleList([MLPNet(in_dim=state_dim + action_dim, out_dim=state_dim) for _ in range(n_ensemble)])


    def get_states(self, state: Tensor, action: Tensor):
        input_x = torch.cat((state, action), dim=1)
        return [self.nets[i](input_x) for i in range(self.n_ensemble)]

    def get_reward(self, state, action):
        state_list = torch.stack(self.get_states(state, action), dim=1)
        var = torch.var(state_list, dim=1).detach()
        return var.mean(dim=1)

    def get_loss(self, memory, batch_size):
        loss = None
        for model in self.nets:
            state, action, _, next_state, _ = memory.sample_batch(batch_size)
            pred_next_state = model(torch.cat((state, action), dim=1))
            import torch.nn.functional as F
            if loss is None:
                loss = F.mse_loss(pred_next_state, next_state)
            else:
                loss += F.mse_loss(pred_next_state, next_state)
        return loss


class RNDModel(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]


        self.predictor = nn.Sequential(nn.Linear(state_dim, 512), nn.ELU(),
                                    nn.Linear(512, 256), nn.ELU(),
                                    nn.Linear(256, 128), nn.ELU(),
                                    nn.Linear(128, 128)
        )


        self.target = nn.Sequential(nn.Linear(state_dim, 512), nn.ELU(),
                                    nn.Linear(512, 256), nn.ELU(),
                                    nn.Linear(256, 128), nn.ELU(),
                                    nn.Linear(128, 128)
        )


        for p in self.modules():
            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()


        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, state):
        target_feature = self.target(state)
        predict_feature = self.predictor(state)

        return predict_feature, target_feature
