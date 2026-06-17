import glob
import os
import random
import re
from collections import deque

import hydra
import numpy as np
import torch
import torch.nn.functional as F

from env.gym_utils import make_async


class TestAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        env_type = cfg.env.get("env_type", None)
        self.venv = make_async(
            cfg.env.name,
            env_type=env_type,
            num_envs=cfg.env.n_envs,
            asynchronous=True,
            max_episode_steps=cfg.env.max_episode_steps,
            wrappers=cfg.env.get("wrappers", None),
            robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
            shape_meta=cfg.get("shape_meta", None),
            use_image_obs=cfg.env.get("use_image_obs", False),
            render=cfg.env.get("render", False),
            render_offscreen=cfg.env.get("save_video", False),
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            env_kwargs=dict(cfg.env.specific) if "specific" in cfg.env else None,
        )
        if env_type != "furniture":
            self.venv.seed([self.seed + i for i in range(cfg.env.n_envs)])
        self.n_envs = cfg.env.n_envs
        self.act_steps = cfg.act_steps
        self.model = hydra.utils.instantiate(cfg.model)

    def load_checkpoint(self, loadpath):
        print(f"[PC-HMD] Loading checkpoint: {loadpath}", flush=True)
        data = torch.load(loadpath, map_location=self.device, weights_only=True)
        self.model.load_state_dict(data["model"])

    def reset_env_all(self):
        obs_venv = self.venv.reset_arg(options_list=[{} for _ in range(self.n_envs)])
        return self._format_observation(obs_venv)

    @staticmethod
    def _format_observation(obs_venv):
        keys = ("neighbor_trajs", "ego_state", "neighbor_waypoints")
        if isinstance(obs_venv, list) and obs_venv:
            if not isinstance(obs_venv[0], tuple) or len(obs_venv[0]) != 3:
                raise TypeError(f"Expected a list of 3-tuples, got {type(obs_venv[0])}")
            return {
                key: np.stack([np.asarray(item[index]) for item in obs_venv], axis=0)
                for index, key in enumerate(keys)
            }
        if isinstance(obs_venv, tuple) and len(obs_venv) == 3:
            return {
                key: np.asarray(obs_venv[index])[None, ...]
                for index, key in enumerate(keys)
            }
        if isinstance(obs_venv, dict):
            missing = [key for key in keys if key not in obs_venv]
            if missing:
                raise KeyError(f"Missing observation keys: {missing}")
            return {key: np.asarray(obs_venv[key]) for key in keys}
        raise TypeError(f"Unsupported observation type: {type(obs_venv)}")


class PCHMDAgent(TestAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.run_cfg = cfg.run

    def process_prev_obs(self, prev_obs_venv):
        obs = self._format_observation(prev_obs_venv)
        cond = {
            key: torch.from_numpy(value).float().to(self.device)
            for key, value in obs.items()
        }
        return cond, obs

    @staticmethod
    def _extract_step(ckpt_path):
        match = re.search(r"state_(\d+)\.pt$", ckpt_path)
        return int(match.group(1)) if match else -1

    def _collect_checkpoints(self):
        checkpoint_path = self.run_cfg.get("checkpoint_path", "")
        checkpoint_dir = self.run_cfg.get("checkpoint_dir", "")
        checkpoint_root = self.run_cfg.get("checkpoint_root", "")

        if checkpoint_path:
            path = os.path.abspath(checkpoint_path)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            return [path]

        if checkpoint_dir:
            directory = os.path.abspath(checkpoint_dir)
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"Checkpoint directory not found: {directory}")
            step = int(self.run_cfg.get("checkpoint_step", -1))
            if step >= 0:
                path = os.path.join(directory, f"state_{step}.pt")
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Checkpoint not found: {path}")
                return [path]
            pattern = os.path.join(directory, self.run_cfg.checkpoint_pattern)
            checkpoints = [path for path in glob.glob(pattern) if os.path.isfile(path)]
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint matches: {pattern}")
            return [max(checkpoints, key=lambda path: (self._extract_step(path), path))]

        if checkpoint_root:
            root = os.path.abspath(checkpoint_root)
            pattern = os.path.join(root, "**", self.run_cfg.checkpoint_pattern)
            checkpoints = [
                path for path in glob.glob(pattern, recursive=True) if os.path.isfile(path)
            ]
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint matches: {pattern}")
            checkpoints.sort(key=lambda path: (self._extract_step(path), path))
            max_checkpoints = int(self.run_cfg.get("max_checkpoints", -1))
            return checkpoints[-max_checkpoints:] if max_checkpoints > 0 else checkpoints

        raise ValueError("Set run.checkpoint_path, run.checkpoint_dir, or run.checkpoint_root")

    def _run_one_episode(self, max_steps, episode):
        obs = self.reset_env_all()
        finish = False
        collision = False
        off_route = False

        for step in range(max_steps):
            with torch.no_grad():
                cond, obs = self.process_prev_obs(obs)
                samples = self.model(cond=cond, deterministic=True, return_chain=True)
                action = samples.trajectories.cpu().numpy()[:, : self.act_steps]

            if self.run_cfg.get("use_target_action_refine", True):
                _, action = self.model.update_target_action(obs, action)
                with torch.no_grad():
                    cond, obs = self.process_prev_obs(obs)
                    samples = self.model.forward_again(
                        cond=cond,
                        deterministic=True,
                        return_chain=True,
                        start_x=action,
                    )
                    action = samples.trajectories.cpu().numpy()[:, : self.act_steps]

            obs, _, terminated, info = self.venv.step(action)
            if not bool(np.asarray(terminated).reshape(-1)[0]):
                continue

            parsed = info
            if isinstance(parsed, (list, tuple)) and parsed and isinstance(parsed[0], (list, tuple)):
                parsed = parsed[0]
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
                finish = bool(parsed[0])
                collision = bool(parsed[1])
                off_route = bool(parsed[2])
            break

        return int(finish and not collision and not off_route)

    def run_checkpoint(self, checkpoint_path):
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        successes = []
        episodes = int(self.run_cfg.episodes_per_checkpoint)
        max_steps = int(self.run_cfg.max_episode_steps)

        for epoch in range(episodes):
            successes.append(self._run_one_episode(max_steps, epoch + 1))
            completion_mean = float(np.mean(successes))
            print(
                f"Epoch {epoch + 1}: completion_mean={completion_mean:.4f}",
                flush=True,
            )

    def run(self):
        for checkpoint_path in self._collect_checkpoints():
            self.run_checkpoint(checkpoint_path)


class AdaptivePCHMDAgent(PCHMDAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        adaptive_cfg = self.run_cfg.get("adaptive", {})
        total_steps = int(cfg.denoising_steps)
        self.u_low = float(adaptive_cfg.get("u_low", 0.0))
        self.u_high = float(adaptive_cfg.get("u_high", 1.0))
        self.k_min = int(round(float(adaptive_cfg.get("k_min_ratio", 0.25)) * total_steps))
        self.k_max = int(round(float(adaptive_cfg.get("k_max_ratio", 0.75)) * total_steps))
        self.k_min = max(1, min(self.k_min, total_steps))
        self.k_max = max(self.k_min, min(self.k_max, total_steps))
        self.history_horizon = max(1, int(adaptive_cfg.get("history_horizon", 1)))
        self.use_target_encoder = bool(adaptive_cfg.get("use_target_encoder", True))
        self.log_steps = bool(adaptive_cfg.get("log_steps", True))

    def _encode_latent(self, cond, target=False):
        encoder = self.model.target_encoder if target else self.model.encoder
        latent, _ = encoder(
            cond["neighbor_trajs"],
            mask=None,
            test=None,
            init_state=cond["ego_state"],
            map_state=cond["neighbor_waypoints"],
        )
        if latent.ndim == 3:
            latent = latent[:, 0]
        return latent.detach()

    def _uncertainty_from_history(self, history, current_latent):
        if not history:
            return 1.0
        errors = []
        for predicted_latent in history:
            cosine = F.cosine_similarity(predicted_latent, current_latent, dim=-1)
            normalized_similarity = (cosine + 1.0) * 0.5
            errors.append(1.0 - normalized_similarity.clamp(0.0, 1.0))
        return float(torch.stack(errors, dim=0).mean().item())

    def _kinf_from_uncertainty(self, uncertainty):
        denom = max(self.u_high - self.u_low, 1e-8)
        scaled = np.clip((uncertainty - self.u_low) / denom, 0.0, 1.0)
        return int(round(self.k_min + (self.k_max - self.k_min) * scaled))

    @staticmethod
    def _parse_terminal_info(info):
        parsed = info
        if isinstance(parsed, (list, tuple)) and parsed and isinstance(parsed[0], (list, tuple)):
            parsed = parsed[0]
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
            return bool(parsed[0]), bool(parsed[1]), bool(parsed[2])
        return False, False, False

    def _run_one_episode(self, max_steps, episode):
        obs = self.reset_env_all()
        finish = False
        collision = False
        off_route = False
        latent_history = deque(maxlen=self.history_horizon)
        uncertainties = []
        denoising_steps = []

        for step in range(max_steps):
            with torch.no_grad():
                cond, obs_np = self.process_prev_obs(obs)
                current_latent = self._encode_latent(
                    cond,
                    target=self.use_target_encoder and len(latent_history) > 0,
                )
                uncertainty = self._uncertainty_from_history(
                    latent_history,
                    current_latent,
                )
                k_inf = self._kinf_from_uncertainty(uncertainty)
                samples = self.model(
                    cond=cond,
                    deterministic=True,
                    return_chain=True,
                    start_step=k_inf,
                )
                action = samples.trajectories.cpu().numpy()[:, : self.act_steps]

            obs, _, terminated, info = self.venv.step(action)
            latent_history.append(current_latent)
            uncertainties.append(uncertainty)
            denoising_steps.append(k_inf)

            if not bool(np.asarray(terminated).reshape(-1)[0]):
                continue

            finish, collision, off_route = self._parse_terminal_info(info)
            break

        success = int(finish and not collision and not off_route)
        mean_uncertainty = float(np.mean(uncertainties)) if uncertainties else 1.0
        mean_k = float(np.mean(denoising_steps)) if denoising_steps else float(self.k_max)
        if self.log_steps:
            print(
                f"Epoch {episode}: adaptive_u_mean={mean_uncertainty:.4f}, "
                f"adaptive_k_mean={mean_k:.2f}, k_range=[{self.k_min},{self.k_max}]",
                flush=True,
            )
        return success, mean_uncertainty, mean_k

    def run_checkpoint(self, checkpoint_path):
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        successes = []
        uncertainty_means = []
        k_means = []
        episodes = int(self.run_cfg.episodes_per_checkpoint)
        max_steps = int(self.run_cfg.max_episode_steps)

        for epoch in range(episodes):
            success, uncertainty_mean, k_mean = self._run_one_episode(max_steps, epoch + 1)
            successes.append(success)
            uncertainty_means.append(uncertainty_mean)
            k_means.append(k_mean)
            print(
                f"Epoch {epoch + 1}: completion_mean={float(np.mean(successes)):.4f}, "
                f"adaptive_u_running={float(np.mean(uncertainty_means)):.4f}, "
                f"adaptive_k_running={float(np.mean(k_means)):.2f}",
                flush=True,
            )
