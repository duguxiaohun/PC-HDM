

from copy import deepcopy

import copy
import torch
import logging
log = logging.getLogger(__name__)

from model.diffusion.diffusion import DiffusionModel, Sample
from model.diffusion.sampling import make_timesteps, extract
from torch.distributions import Normal
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ddiffpg.utils.schedule_util import ExponentialSchedule
from ddiffpg.utils.schedule_util import LinearSchedule
from model.diffusion.policy_v1 import RLEncoder


from configs.init_configs import set_configs, get_argument
import gym

parser = get_argument()
args, _unknown = parser.parse_known_args()

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

args.scenario = 'carla'
args.algo = 'scene_rep'
args, params_cfg, runner_params = set_configs(args, test=False)
params_cfg = params_cfg['params']
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
OBSERVATION_SPACE = gym.spaces.Box(low=-1000, high=1000, shape=(args.neighbors + 1, args.n_steps, args.dim,))

class VPGDiffusion(DiffusionModel):

    def __init__(
        self,
        actor,
        critic,
        ft_denoising_steps,
        ft_denoising_steps_d=0,
        ft_denoising_steps_t=0,
        network_path=None,

        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,

        eta=None,
        learn_eta=False,
        noise=None,
        params=None,
        **kwargs,
    ):
        super().__init__(
            network=actor,
            network_path=network_path,
            **kwargs,
        )
        assert ft_denoising_steps <= self.denoising_steps
        assert ft_denoising_steps <= self.ddim_steps if self.use_ddim else True
        assert not (learn_eta and not self.use_ddim), "Cannot learn eta with DDPM."


        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_d = ft_denoising_steps_d
        self.ft_denoising_steps_t = ft_denoising_steps_t
        self.ft_denoising_steps_cnt = 0


        self.min_sampling_denoising_std = min_sampling_denoising_std


        self.min_logprob_denoising_std = min_logprob_denoising_std


        self.learn_eta = learn_eta
        if eta is not None:
            self.eta = eta.to(self.device)
            if not learn_eta:
                for param in self.eta.parameters():
                    param.requires_grad = False
                logging.info("Turned off gradients for eta")


        self.actor = self.network


        self.actor_ft = copy.deepcopy(self.actor)
        logging.info("Cloned model for fine-tuning")


        for param in self.actor.parameters():
            param.requires_grad = False
        logging.info("Turned off gradients of the pretrained network")
        logging.info(
            f"Number of finetuned parameters: {sum(p.numel() for p in self.actor_ft.parameters() if p.requires_grad)}"
        )


        self.critic = critic.to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=self.device, weights_only=True
            )
            if "ema" not in checkpoint:
                self.load_state_dict(checkpoint["model"], strict=False)
                logging.info("Loaded critic from %s", network_path)
        if noise is not None:
            if noise.decay == 'linear':
                self.noise_scheduler = LinearSchedule(start_val=self.cfg.algo.noise.std_max,
                                                      end_val=self.cfg.algo.noise.std_min,
                                                      total_iters=self.cfg.algo.noise.lin_decay_iters
                                                      )
            elif noise.decay == 'exp':
                self.noise_scheduler = ExponentialSchedule(start_val=self.cfg.algo.noise.std_max,
                                                           gamma=self.cfg.algo.exp_decay_rate,
                                                           end_val=self.cfg.algo.noise.std_min)
            else:
                self.noise_scheduler = None


        self.reward_mean = deque(maxlen=int(1e4))

        self.encoder = RLEncoder(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.shape, units=[256] * 3,
                                 trans=False,
                                 cnn_lstm=params_cfg['cnn_lstm'], ego_surr=params_cfg['ego_surr'],
                                 use_trans=params_cfg['use_trans'], neighbours=params_cfg['neighbours'],
                                 time_step=params_cfg['time_step'], debug=False,
                                 make_rotation=params_cfg['make_rotation'], make_prediction=params_cfg['make_prediction'],
                                 use_map=params_cfg['use_map'],
                                 num_traj=params_cfg['traj_nums'], path_length=params_cfg['path_length'], head_dim=params_cfg['head_num'],
                                 cnn=params_cfg['cnn'],
                                 use_hier=params_cfg['use_hier'], random_aug=params_cfg['random_aug'], carla=params_cfg['carla'],
                                 no_ego_fut=params_cfg['no_ego_fut'], no_neighbor_fut=params_cfg['no_neighbor_fut']).to(self.device)
        self.target_encoder = RLEncoder(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.shape, units=[256] * 3,
                                 trans=False,
                                 cnn_lstm=params_cfg['cnn_lstm'], ego_surr=params_cfg['ego_surr'],
                                 use_trans=params_cfg['use_trans'], neighbours=params_cfg['neighbours'],
                                 time_step=params_cfg['time_step'], debug=False,
                                 make_rotation=params_cfg['make_rotation'],
                                 make_prediction=params_cfg['make_prediction'],
                                 use_map=params_cfg['use_map'],
                                 num_traj=params_cfg['traj_nums'], path_length=params_cfg['path_length'],
                                 head_dim=params_cfg['head_num'],
                                 cnn=params_cfg['cnn'],
                                 use_hier=params_cfg['use_hier'], random_aug=params_cfg['random_aug'],
                                 carla=params_cfg['carla'],
                                 no_ego_fut=params_cfg['no_ego_fut'], no_neighbor_fut=params_cfg['no_neighbor_fut']).to(
            self.device)


    def step(self):


        if type(self.min_sampling_denoising_std) is not float:
            self.min_sampling_denoising_std.step()


        self.ft_denoising_steps_cnt += 1
        if (
            self.ft_denoising_steps_d > 0
            and self.ft_denoising_steps_t > 0
            and self.ft_denoising_steps_cnt % self.ft_denoising_steps_t == 0
        ):
            self.ft_denoising_steps = max(
                0, self.ft_denoising_steps - self.ft_denoising_steps_d
            )


            self.actor = self.actor_ft
            self.actor_ft = copy.deepcopy(self.actor)
            for param in self.actor.parameters():
                param.requires_grad = False
            logging.info(
                f"Finished annealing fine-tuning denoising steps to {self.ft_denoising_steps}"
            )

    def get_min_sampling_denoising_std(self):
        if type(self.min_sampling_denoising_std) is float:
            return self.min_sampling_denoising_std
        else:
            return self.min_sampling_denoising_std()


    def get_tgt_policy_actions(self, obs, deterministic=False, return_chain=False):
        samples = self.forward(
            cond=obs,
            deterministic=deterministic,
            return_chain=return_chain,
            use_base_policy=True,
        )

        return samples


    def bc_loss(self,
                x_start,
                cond,
                index=None,
                use_base_policy=False,
                deterministic=False,
                ):
        batch_size = len(x_start)
        t = torch.randint(
            0, self.denoising_steps, (batch_size,), device=x_start.device
        ).long()
        device = x_start.device
        noise_random = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise_random)
        noise = self.actor(x_noisy, t, cond=cond)
        if self.use_ddim:
            ft_indices = torch.where(
                index >= (self.ddim_steps - self.ft_denoising_steps)
            )[0]
        else:
            ft_indices = torch.where(t < self.ft_denoising_steps)[0]


        actor = self.actor if use_base_policy else self.actor_ft

        if len(ft_indices) > 0:

            cond_ft = cond[ft_indices]
            noise_ft = actor(x_noisy[ft_indices], t[ft_indices], cond=cond_ft)
            noise[ft_indices] = noise_ft

        return F.mse_loss(noise_random, noise, reduction="mean")


    def p_mean_var(
        self,
        x,
        t,
        cond,
        index=None,
        use_base_policy=False,
        deterministic=False,
    ):

        noise = self.actor(x, t, cond=cond)
        if self.use_ddim:
            ft_indices = torch.where(
                index >= (self.ddim_steps - self.ft_denoising_steps)
            )[0]
        else:
            ft_indices = torch.where(t < self.ft_denoising_steps)[0]


        actor = self.actor if use_base_policy else self.actor_ft

        if len(ft_indices) > 0:

            cond_ft = cond[ft_indices]
            noise_ft = actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            noise[ft_indices] = noise_ft


        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
                """
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:

                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha


        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)


        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε
            """
            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(cond).unsqueeze(1)
            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)
            dir_xt_coef = (1.0 - alpha_prev - sigma**2).clamp_(min=0).sqrt()
            mu = (alpha_prev**0.5) * x_recon + dir_xt_coef * noise
            var = sigma**2
            logvar = torch.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
            etas = torch.ones_like(mu).to(mu.device)
        return mu, logvar, etas


    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        return_chain=True,
        use_base_policy=False,
        mask=None,
        test=False,
        start_x=None,
        start_step=None,
        use_target_critic_for_improvement=False,
    ):


        device = self.betas.device
        if isinstance(cond, dict):
            cond = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond.items()}
        elif torch.is_tensor(cond):
            cond = cond.to(device)

        if mask is not None and torch.is_tensor(mask):
            mask = mask.to(device)

        if isinstance(cond, dict):
            cond, _ = self.encoder(
                cond['neighbor_trajs'],
                mask=mask,
                test=test,
                init_state=cond['ego_state'],
                map_state=cond['neighbor_waypoints']
            )

        B = cond.shape[0]


        min_sampling_denoising_std = self.get_min_sampling_denoising_std()


        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        x = torch.clamp(x, -1.0, 1.0)
        if start_step is not None:
            start_x = self.denoising_steps - int(start_step)
        elif start_x is None:
            start_x = self.denoising_steps
        elif not isinstance(start_x, int):
            if hasattr(start_x, 'item'):
                start_x = start_x.item()
            u = float(start_x)
            start_x = int((1.0 - u) * (self.denoising_steps - 1))
        start_x = max(0, min(int(start_x), self.denoising_steps))
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        if start_x < self.denoising_steps:
            t_all = t_all[start_x:]
        chain = [] if return_chain else None
        if return_chain:
            if not self.use_ddim and self.ft_denoising_steps == self.denoising_steps:
                chain.append(x)
            if self.use_ddim and self.ft_denoising_steps == self.ddim_steps:
                chain.append(x)
        x = self.update_targetaction(cond, x, use_target_critic=use_target_critic_for_improvement)
        if start_x < self.denoising_steps:
            for i in range(start_x):
                chain.append(x)
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar, _ = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
                use_base_policy=use_base_policy,
                deterministic=deterministic,
            )
            std = torch.exp(0.5 * logvar)


            if self.use_ddim:
                if deterministic:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=min_sampling_denoising_std)
            else:
                if deterministic and t == 0:
                    std = torch.zeros_like(std)
                elif deterministic:
                    std = torch.clip(std, min=1e-3)
                else:
                    std = torch.clip(std, min=min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise


            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )

            if return_chain:
                if not self.use_ddim and t <= self.ft_denoising_steps:
                    chain.append(x)
                elif self.use_ddim and i >= (
                    self.ddim_steps - self.ft_denoising_steps - 1
                ):
                    chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)

        return Sample(x, chain)

    def forward_again(
            self,
            cond,
            deterministic=False,
            return_chain=True,
            use_base_policy=False,
            start_x=None,
    ):


        device = self.betas.device
        if isinstance(cond, dict):
            cond = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond.items()}
        elif torch.is_tensor(cond):
            cond = cond.to(device)


        cond, _ = self.encoder(
            cond['neighbor_trajs'],
            mask=None,
            test=None,
            init_state=cond['ego_state'],
            map_state=cond['neighbor_waypoints']
        )
        B = cond.shape[0]


        min_sampling_denoising_std = self.get_min_sampling_denoising_std()


        if start_x == None:
            x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        else:
            x = start_x.to(device)

        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        chain = [] if return_chain else None
        if not self.use_ddim and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x)
        if self.use_ddim and self.ft_denoising_steps == self.ddim_steps:
            chain.append(x)
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar, _ = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
                use_base_policy=use_base_policy,
                deterministic=deterministic,
            )
            std = torch.exp(0.5 * logvar)


            if self.use_ddim:
                if deterministic:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=min_sampling_denoising_std)
            else:
                if deterministic and t == 0:
                    std = torch.zeros_like(std)
                elif deterministic:
                    std = torch.clip(std, min=1e-3)
                else:
                    std = torch.clip(std, min=min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise


            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )

            if return_chain:
                if not self.use_ddim and t <= self.ft_denoising_steps:
                    chain.append(x)
                elif self.use_ddim and i >= (
                        self.ddim_steps - self.ft_denoising_steps - 1
                ):
                    chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)

        return Sample(x, chain)


    def get_logprobs(
        self,
        cond,
        chains,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):


        if isinstance(cond, dict):
            cond_enc, _ = self.encoder(
                cond['neighbor_trajs'],
                mask=None,
                test=None,
                init_state=cond['ego_state'],
                map_state=cond['neighbor_waypoints']
            )


            cond_enc = cond_enc.flatten(start_dim=1)


            cond_enc = cond_enc.unsqueeze(1).repeat(1, self.ft_denoising_steps, 1)


            cond_enc = cond_enc.flatten(start_dim=0, end_dim=1)

            cond = cond_enc


        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = torch.arange(
                start=self.ft_denoising_steps - 1,
                end=-1,
                step=-1,
                device=self.device,
            )

        t_all = t_single.repeat(chains.shape[0], 1).flatten()
        if self.use_ddim:
            indices_single = torch.arange(
                start=self.ddim_steps - self.ft_denoising_steps,
                end=self.ddim_steps,
                device=self.device,
            )
            indices = indices_single.repeat(chains.shape[0])
        else:
            indices = None


        chains_prev = chains[:, :-1]
        chains_next = chains[:, 1:]


        chains_prev = chains_prev.reshape(-1, self.horizon_steps, self.action_dim)
        chains_next = chains_next.reshape(-1, self.horizon_steps, self.action_dim)


        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=indices,
            use_base_policy=use_base_policy,
        )
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        dist = Normal(next_mean, std)


        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    def get_logprobs_subsample(
        self,
        cond,
        chains_prev,
        chains_next,
        denoising_inds,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):


        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = torch.arange(
                start=self.ft_denoising_steps - 1,
                end=-1,
                step=-1,
                device=self.device,
            )

        t_all = t_single[denoising_inds]
        if self.use_ddim:
            ddim_indices_single = torch.arange(
                start=self.ddim_steps - self.ft_denoising_steps,
                end=self.ddim_steps,
                device=self.device,
            )
            ddim_indices = ddim_indices_single[denoising_inds]
        else:
            ddim_indices = None


        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=ddim_indices,
            use_base_policy=use_base_policy,
        )

        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        dist = Normal(next_mean, std)


        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    def loss(self, cond, chains, reward):


        with torch.no_grad():
            value = self.critic(cond).squeeze()
        advantage = reward - value


        logprobs, eta = self.get_logprobs(cond, chains, get_ent=True)


        logprobs = logprobs[:, :, : self.action_dim].sum(-1)


        logprobs = logprobs.reshape((-1, self.denoising_steps, self.horizon_steps))


        logprobs = logprobs.mean(-2)


        logprobs = logprobs.mean(-1)


        loss_actor = torch.mean(-logprobs * advantage)


        pred = self.critic(cond).squeeze()
        loss_critic = F.mse_loss(pred, reward)
        return loss_actor, loss_critic, eta
    def update_targetaction(self, obs, action, use_target_critic=False):
        import random
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        lim = 1 - 1e-5
        action = action.clamp(-lim, lim)


        q_net = self.critic_target if use_target_critic else self.critic
        q_net.requires_grad_(False)

        rangee = 0.2 + 0.4 * random.random()
        candidates = [action + delta for delta in [-rangee, 0.0, rangee]]
        best_action = action
        best_q = -float('inf')
        for cand in candidates:
            q = q_net.get_q_min(obs, cand).mean().item()
            if q > best_q:
                best_q = q
                best_action = cand

        q_net.requires_grad_(True)
        return deepcopy(best_action.clamp(-lim, lim).detach())
