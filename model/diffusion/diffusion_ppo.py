

from copy import deepcopy
from typing import Optional
import torch
import logging
import math
import torch.nn.functional as F
from ddiffpg.utils.distl_util import projection

log = logging.getLogger(__name__)
from model.diffusion.diffusion_vpg import VPGDiffusion


class PPODiffusion(VPGDiffusion):
    def __init__(
        self,
        gamma_denoising: float,
        clip_ploss_coef: float,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0,
        clip_advantage_upper_quantile: float = 1,
        norm_adv: bool = True,

        ratio_clip_max: float = 5.0,
        clip_ploss_dual: Optional[float] = 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)


        self.norm_adv = norm_adv


        self.ratio_clip_max = ratio_clip_max
        self.clip_ploss_dual = clip_ploss_dual


        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate


        self.clip_vloss_coef = clip_vloss_coef


        self.gamma_denoising = gamma_denoising


        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile

    def encode_observation(self, obs, target=False):
        if not isinstance(obs, dict):
            return obs
        encoder = self.target_encoder if target else self.encoder
        obs, _ = encoder(
            obs['neighbor_trajs'],
            mask=None,
            test=True,
            init_state=obs['ego_state'],
            map_state=obs['neighbor_waypoints']
        )
        return obs

    def expected_q_min(self, obs, action):
        return self.critic.get_q_min(obs, action).view(-1)

    def compute_q_guided_denoising_advantages(self, obs, final_action, denoising_inds):
        """
        Eq. (24): A_hat(Z_t, a_t^k) = gamma_den^k * (E[Q_phi(Z_t, a_t^0)] - V_psi(Z_t)).
        """
        obs = self.encode_observation(obs)
        with torch.no_grad():
            q_value = self.expected_q_min(obs, final_action)
            value = self.critic.get_v(obs).view(-1)
            advantages = q_value - value
            discount = torch.tensor(
                [
                    self.gamma_denoising ** (self.ft_denoising_steps - i - 1)
                    for i in denoising_inds
                ],
                device=self.device,
                dtype=advantages.dtype,
            )
        return advantages * discount

    def q_guided_action_refinement(self, obs, action):
        """
        Eq. (22): refine a_old by ascending the expected ordinal Q value.
        """
        return self.update_target_action(obs, action)

    def long_memory_actor_loss(self, obs, old_action):
        """
        Eq. (23): denoising behavior cloning toward the Q-refined action a*.
        """
        _, refined_action = self.q_guided_action_refinement(obs, old_action)
        return self.update_actor(obs, refined_action), refined_action

    def loss(
        self,
        obs,
        chains_prev,
        chains_next,
        denoising_inds,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        reward_horizon=4,
        advantages_include_denoising_discount=False,
    ):


        obs = self.encode_observation(obs)


        newlogprobs, eta = self.get_logprobs_subsample(
            obs,
            chains_prev,
            chains_next,
            denoising_inds,
            get_ent=True,

        )

        entropy_loss = -eta.mean()
        newlogprobs = newlogprobs.clamp(min=-5, max=2)
        oldlogprobs = oldlogprobs.clamp(min=-5, max=2)


        newlogprobs = newlogprobs[:, :reward_horizon, :]
        oldlogprobs = oldlogprobs[:, :reward_horizon, :]


        newlogprobs = newlogprobs.sum(dim=-1).mean(dim=-1).view(-1)
        oldlogprobs = oldlogprobs.sum(dim=-1).mean(dim=-1).view(-1)


        bc_loss = torch.zeros(1, device=self.device).squeeze()
        if use_bc_loss:


            samples = self.forward(
                cond=obs,
                deterministic=False,
                return_chain=True,
                use_base_policy=True,
            )

            bc_logprobs = self.get_logprobs(
                obs,
                samples.chains,
                get_ent=False,
                use_base_policy=False,
            )
            bc_logprobs = bc_logprobs.clamp(min=-5, max=2)
            bc_logprobs = bc_logprobs.mean(dim=(-1, -2)).view(-1)
            bc_loss = -bc_logprobs.mean()


        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        advantage_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        advantage_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=advantage_min, max=advantage_max)


        if not advantages_include_denoising_discount:
            discount = torch.tensor(
                [
                    self.gamma_denoising ** (self.ft_denoising_steps - i - 1)
                    for i in denoising_inds
                ]
            ).to(self.device)
            advantages *= discount


        logratio = newlogprobs - oldlogprobs


        ratio = logratio.exp()


        if self.ft_denoising_steps > 1:
            t = (denoising_inds.float() / (self.ft_denoising_steps - 1)).to(self.device)
            clip_ploss_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch.exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            clip_ploss_coef = torch.full_like(
                denoising_inds.float(), self.clip_ploss_coef, device=self.device
            )


        with torch.no_grad():


            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_ploss_coef).float().mean().item()


        if self.ratio_clip_max is not None:
            ratio_pg = ratio.clamp(
                max=self.ratio_clip_max, min=1.0 / self.ratio_clip_max
            )
        else:
            ratio_pg = ratio


        pg_loss1 = -advantages * ratio_pg
        pg_loss2 = -advantages * torch.clamp(
            ratio_pg, 1 - clip_ploss_coef, 1 + clip_ploss_coef
        )
        pg_loss_per = torch.max(pg_loss1, pg_loss2)


        if self.clip_ploss_dual is not None:
            pg_loss_per = torch.where(
                advantages < 0,
                torch.min(pg_loss_per, -self.clip_ploss_dual * advantages),
                pg_loss_per,
            )
        pg_loss = pg_loss_per.mean()


        returns = returns.clamp(min=self.critic.v_min, max=self.critic.v_max)
        newvalues = self.critic.get_v(obs).view(-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch.clamp(
                newvalues - oldvalues,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            bc_loss,
            eta.mean().item(),
        )


    def update_critic(self, obs, action, reward, next_obs, done):
        """
        Eq. (20)-(21): soft ordinal projection followed by cross entropy
        -sum_i m_i log p_i(Z_t, a_t).
        """

        with torch.no_grad():
            obs = self.encode_observation(obs)
            next_obs = self.encode_observation(next_obs, target=True)
            next_actions = self.forward(
                cond=next_obs, deterministic=True, return_chain=False,
                use_base_policy=False, use_target_critic_for_improvement=True,
            ).trajectories

        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target.get_q1_q2(next_obs, next_actions)
            target_Q1_projected = projection(next_dist=target_Q1,
                                             reward=reward,
                                             done=done,
                                             gamma=self.gamma ** self.nstep,
                                             v_min=self.critic.v_min,
                                             v_max=self.critic.v_max,
                                             num_atoms=self.critic.num_atoms,
                                             support=self.critic.z_atoms,
                                             device=self.device)
            target_Q2_projected = projection(next_dist=target_Q2,
                                             reward=reward,
                                             done=done,
                                             gamma=self.gamma ** self.nstep,
                                             v_min=self.critic.v_min,
                                             v_max=self.critic.v_max,
                                             num_atoms=self.critic.num_atoms,
                                             support=self.critic.z_atoms,
                                             device=self.device)


            z = self.critic.z_atoms.to(self.device)
            exp_Q1 = (target_Q1_projected * z).sum(dim=1, keepdim=True)
            exp_Q2 = (target_Q2_projected * z).sum(dim=1, keepdim=True)
            target_Q = torch.where(exp_Q1 < exp_Q2, target_Q1_projected, target_Q2_projected)


        logits_Q1, logits_Q2 = self.critic.get_q1_q2_logits(obs, action)
        log_prob1 = F.log_softmax(logits_Q1, dim=1)
        log_prob2 = F.log_softmax(logits_Q2, dim=1)
        critic_loss = -torch.sum(target_Q * log_prob1, dim=1).mean()\
                      - torch.sum(target_Q * log_prob2, dim=1).mean()

        return critic_loss


    def update_actor(self, obs, target_action):
        with torch.no_grad():
            obs = self.encode_observation(obs)
        actor_loss = self.bc_loss(target_action, obs)
        return actor_loss

    def update_target_action(self, obs, action):


        if isinstance(obs, dict):
            with torch.no_grad():
                obs = self.encode_observation(obs)

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.detach().clone().to(self.device)
        self.critic.requires_grad_(False)
        lim = 1 - 1e-5
        action.clamp_(-lim, lim)

        action_optimizer = torch.optim.Adam([action], lr=self.action_lr, eps=1e-5)

        for _ in range(self.update_times):
            action.requires_grad_(True)

            Q = self.critic.get_q_min(obs, action)
            loss = -Q.mean()


            self.optimizer_update(action_optimizer, loss)
            action.requires_grad_(False)
            action.clamp_(-lim, lim)

        target_action = action.detach().to(self.device)
        update = deepcopy(target_action)
        self.critic.requires_grad_(True)
        return torch.abs(action).mean().item(), update


    def optimizer_update(self, optimizer, objective):
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        from torch.nn.utils import clip_grad_norm_

        if self.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(parameters=optimizer.param_groups[0]["params"],
                                        max_norm=self.max_grad_norm)
        else:
            grad_norm = None
        optimizer.step()
        return grad_norm


    def offline_loss(
        self,
        obs,
        denoising_inds,
        returns,
        oldvalues,
        oldlogprobs,
        use_bc_loss=False,
        reward_horizon=4,
    ):


        if isinstance(obs, dict):
            obs, _ = self.encoder(
                obs['neighbor_trajs'],
                mask=None,
                test=True,
                init_state=obs['ego_state'],
                map_state=obs['neighbor_waypoints']
            )


        bc_loss = torch.zeros(1, device=self.device).squeeze()
        if use_bc_loss:


            samples = self.forward(
                cond=obs,
                deterministic=False,
                return_chain=True,
                use_base_policy=True,
            )

            bc_logprobs = self.get_logprobs(
                obs,
                samples.chains,
                get_ent=False,
                use_base_policy=False,
            )
            bc_logprobs = bc_logprobs.clamp(min=-5, max=2)
            bc_logprobs = bc_logprobs.mean(dim=(-1, -2)).view(-1)
            bc_loss = -bc_logprobs.mean()


        discount = torch.tensor(
            [
                self.gamma_denoising ** (self.ft_denoising_steps - i - 1)
                for i in denoising_inds
            ]
        ).to(self.device)


        t = (denoising_inds.float() / (self.ft_denoising_steps - 1)).to(self.device)
        if self.ft_denoising_steps > 1:
            clip_ploss_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch.exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            clip_ploss_coef = t


        newvalues = self.critic.get_v(obs).view(-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch.clamp(
                newvalues - oldvalues,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        return (
            v_loss,
            bc_loss,
        )
