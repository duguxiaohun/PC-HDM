from dataclasses import dataclass
from typing import Any

import torch
from omegaconf.dictconfig import DictConfig
from torch.nn.utils import clip_grad_norm_

from ddiffpg.utils.common import load_class_from_path
from ddiffpg.models import model_name_to_path
from ddiffpg.utils.common import Tracker
from ddiffpg.utils.torch_util import RunningMeanStd
from ddiffpg.models.diffusion_mlp import DiffusionPolicy


@dataclass
class ActorCriticBase:
    env: Any
    cfg: DictConfig

    def __post_init__(self):
        self.obs = None
        self.obs_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape[0]
        self.max_episode_len = self.env.max_episode_length
        act_class = load_class_from_path(self.cfg.algo.act_class,
                                         model_name_to_path[self.cfg.algo.act_class])
        cri_class = load_class_from_path(self.cfg.algo.cri_class,
                                         model_name_to_path[self.cfg.algo.cri_class])
        if self.cfg.algo.name == "DDiffPG":
            obs_dim = self.obs_dim[0] + self.cfg.algo.embedding_dim
            self.actor = DiffusionPolicy(obs_dim, self.action_dim, self.cfg.diffusion.diffusion_iter).to(self.cfg.device)
        elif self.cfg.algo.name == "DIPO":
            self.actor = DiffusionPolicy(self.obs_dim, self.action_dim, self.cfg.diffusion.diffusion_iter).to(self.cfg.device)
        elif self.cfg.algo.name == "DiffQ":
            from ddiffpg.models.baseline_models import Diffusion, Consistency, MLP
            self.model = MLP(state_dim=self.obs_dim[0], action_dim=self.action_dim, device=self.cfg.device)
            self.actor = Diffusion(state_dim=self.obs_dim[0], action_dim=self.action_dim, model=self.model, max_action=1.0,
                              beta_schedule="vp", n_timesteps=self.cfg.diffusion.diffusion_iter,).to(self.cfg.device)


        else:
            self.actor = act_class(self.obs_dim, self.action_dim).to(self.cfg.device)

        if self.cfg.algo.cri_class == "DistributionalDoubleQ":
            self.critic = cri_class(self.obs_dim, self.action_dim,
                                    v_min=self.cfg.algo.v_min,
                                    v_max=self.cfg.algo.v_max,
                                    num_atoms=self.cfg.algo.num_atoms,
                                    device=self.cfg.device).to(self.cfg.device)
        else:
            self.critic = cri_class(self.obs_dim, self.action_dim).to(self.cfg.device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), self.cfg.algo.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), self.cfg.algo.critic_lr)
        self.return_tracker = Tracker(self.cfg.algo.tracker_len)
        self.step_tracker = Tracker(self.cfg.algo.tracker_len)
        self.current_returns = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.cfg.device)
        self.current_lengths = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.cfg.device)

        self.device = torch.device(self.cfg.device)

        if self.cfg.algo.obs_norm:
            self.obs_rms = RunningMeanStd(shape=self.obs_dim, device=self.device)
        else:
            self.obs_rms = None

    def reset_agent(self):
        self.obs = self.env.reset()

    def update_tracker(self, reward, done):
        self.current_returns += reward
        self.current_lengths += 1
        env_done_indices = torch.where(done)[0]
        cumu_return = self.current_returns[env_done_indices]
        self.return_tracker.update(cumu_return)
        step = self.current_lengths.clone()
        self.step_tracker.update(self.current_lengths[env_done_indices])
        self.current_returns[env_done_indices] = 0
        self.current_lengths[env_done_indices] = 0
        return {'indices': env_done_indices,
                'cumulative_reward': cumu_return,
                'step': step}

    def optimizer_update(self, optimizer, objective):
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if self.cfg.algo.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(parameters=optimizer.param_groups[0]["params"],
                                        max_norm=self.cfg.algo.max_grad_norm)
        else:
            grad_norm = None
        optimizer.step()
        return grad_norm

    def get_noise_std(self):
        if self.noise_scheduler is None:
            return self.cfg.algo.noise.std_max
        else:
            return self.noise_scheduler.val()

    def update_noise(self):
        if self.noise_scheduler is not None:
            self.noise_scheduler.step()
