import numpy as np
from collections.abc import Sequence
import math
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionNet(nn.Module):
    def __init__(
        self,
        transition_dim,
        cond_dim,
        dim=256,
        num_blocks=3,
        act_fn=nn.Mish()
    ):
        super().__init__()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.transition_dim = transition_dim
        self.action_dim = transition_dim - cond_dim

        embed_dim = dim

        self.mlp = nn.Sequential(
                        nn.Linear(embed_dim + transition_dim, 1024),
                        act_fn,
                        nn.Linear(1024, 512),
                        act_fn,
                        nn.Linear(512, 256),
                        act_fn,
                        nn.Linear(256, self.action_dim),
                    )


    def forward(self, x, time, cond):


        t = self.time_mlp(time)

        inp = torch.cat([t, cond, x], dim=-1)
        out = self.mlp(inp)

        return out


class MLPResNetBlock(nn.Module):

    def __init__(self, features, act, dropout_rate=None, use_layer_norm=False):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.features = features
        self.act = act

        self.dense1 = nn.Linear(features, features * 4)
        self.dense2 = nn.Linear(features * 4, features)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x = self.dropout(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)

        if residual.shape != x.shape:
            residual = self.dense2(residual)

        return residual + x


class MLPResNet(nn.Module):
    def __init__(self, num_blocks, in_dim, out_dim, dropout_rate=0.1, use_layer_norm=True, hidden_dim=256, act=None):
        super().__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.act = act

        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim, act, dropout_rate, use_layer_norm) for _ in range(num_blocks)])
        self.dense2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.dense1(x)
        for block in self.blocks:
            x = block(x)
        x = self.act(x)
        x = self.dense2(x)
        return x


class EBMDiffusionModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def neg_logp_unnorm(self, x, t, obs):
        score = self.net(x, t, obs)
        return ((score - x) ** 2).sum(-1)

    def forward(self, x, t, obs):
        x.requires_grad_(True)
        neg_logp_unnorm_sum = self.neg_logp_unnorm(x, t, obs).sum()
        grad_outputs = torch.ones_like(neg_logp_unnorm_sum)
        gradients, = torch.autograd.grad(neg_logp_unnorm_sum, x, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)
        return gradients


class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, diffusion_iter, num_mode=0, tau1=0.4, tau2=0.9, noise_min=0.0,
                                    noise_max=0.25, noise_type="mixed", psi=1.0, energy=False, device="cuda"):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.diffusion_iter = diffusion_iter
        self.device = device


        self.net = DiffusionNet(
            transition_dim=state_dim + action_dim + num_mode,
            cond_dim=state_dim + num_mode)
        if energy:
            self.net = EBMDiffusionModel(self.net).to(self.device)


        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.diffusion_iter,
            beta_schedule='squaredcos_cap_v2',

            clip_sample=True,
            prediction_type='epsilon'
        )


        self.tau1 = tau1
        self.tau2 = tau2
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.noise_type = noise_type
        self.psi = psi
        self.rescale = True

    def forward(self, x, sample=True, add_noise=False):
        return self.get_actions(x, sample=sample, add_noise=add_noise)

    def add_noise(self, t, state):
        t = max(min(t / self.diffusion_iter, 1.0), 0.0)


        if t <= self.tau1:
            gamma = 1.0
        elif t >= self.tau2:
            gamma = 0.0
        else:
            gamma = (self.tau2 - t)/(self.tau2 - self.tau1)


        state_mean, state_std = torch.mean(state, dim=1, keepdim=True).to(state.device), torch.std(state, dim=1, keepdim=True).to(state.device)


        if self.noise_type == "mixed":
            noise_scale = torch.linspace(self.noise_min, self.noise_max, state.shape[0]).to(state.device)
        elif self.noise_type == "fixed":
            noise_scale = self.noise_max
        else:
            raise NotImplementedError

        state = np.sqrt(gamma) * state + noise_scale.unsqueeze(-1) * np.sqrt(1 - gamma) * torch.randn_like(state).to(state.device)

        if self.rescale:
            state_scaled = (state - torch.mean(state, dim=1, keepdim=True)) / torch.std(state, dim=1, keepdim=True) * state_std + state_mean
            if not torch.isnan(state_scaled).any():
                state = self.psi * state_scaled + (1 - self.psi) * state
        return state

    def get_actions(self, state, sample=True, add_noise=False):
        B = state.shape[0]

        noisy_action = torch.randn(
            (B, self.action_dim), device=self.device)

        self.noise_scheduler.set_timesteps(self.diffusion_iter)

        for k in self.noise_scheduler.timesteps:

            timesteps = torch.ones(B, device=self.device) * k

            if add_noise:
                state = self.add_noise(k, state)

            noise_pred = self.net(
                noisy_action,
                timesteps,
                state
            )
            if sample:
                noise_pred = noise_pred.detach()


            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action
            ).prev_sample
            if sample:
                noisy_action = noisy_action.detach()

        return noisy_action

    def get_actions_logprob_entropy(self, state, action_buf, sample=True):
        action = self.get_actions(state, sample=sample)
        log_prob, _ = self.logprob(state, action, action_buf)
        if sample:
            log_prob = log_prob.detach()
        return action, log_prob

    def logprob(self, state, action, action_buf, use_entropy=False):
        B1, B2 = state.shape[0], action_buf.shape[0]
        t1 = torch.zeros(B1, device=self.device)
        t2 = torch.zeros(B1*B2, device=self.device)


        E = self.net.neg_logp_unnorm(action, t1, state)


        a = action_buf.repeat(B1, 1)
        s = torch.repeat_interleave(state, repeats=B2, dim=0)
        pred = -self.net.neg_logp_unnorm(a, t2, s)
        Z = pred.exp()
        Z = Z.reshape(B1, B2).sum(-1)
        log_prob = -E - Z.log()
        if torch.any(torch.isnan(log_prob)):
            raise FloatingPointError("NaN encountered while computing action log probability")

        if use_entropy:
            entropy = self.entropy(state, Z.log())
        else:
            entropy = None
        return log_prob, entropy

    def entropy(self, state, logZ, num_action=50):

        s = torch.repeat_interleave(state, repeats=num_action, dim=0)
        action = self.get_actions(s, sample=False)
        t = torch.zeros(s.shape[0], device=self.device)
        E = self.net.neg_logp_unnorm(action, t, s)
        E = E.reshape(state.shape[0], num_action).mean(axis=1)
        return E + logZ

    def get_loss(self, state, action, noise=None, timesteps=None):
        B = action.shape[0]

        if noise is None:
            noise = torch.randn_like(action, device=self.device)


        if timesteps is None:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=self.device
            ).long()


        noisy_action = self.noise_scheduler.add_noise(
            action, noise, timesteps)


        noise_pred = self.net(
                noisy_action,
                timesteps,
                state
            )


        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss
