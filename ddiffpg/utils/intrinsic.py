import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from ddiffpg.models.mlp import RNDModel
from ddiffpg.utils.torch_util import RunningMeanStd


class IntrinsicM:
    def __init__(self, obs_dim, type='noveld', normalize=True, warm_up=1000, device='cuda'):
        self.obs_dim = obs_dim
        self.type = type
        self.normalize = normalize
        self.device = device
        self.update_step = 0
        self.warm_up = warm_up

        self.rnd_model = RNDModel(self.obs_dim).to(self.device)
        self.rnd_optimizer = torch.optim.AdamW(self.rnd_model.parameters(), 1e-4)
        self.rnd_rms = RunningMeanStd(shape=(1), device=self.device)

    def compute_reward(self, obs, next_obs=None):
        if len(obs.shape) > 2:
            obs = obs.view(obs.shape[0], -1)
        if len(next_obs.shape) > 2:
            next_obs = next_obs.view(next_obs.shape[0], -1)


        if self.type == 'rnd':
            novelty_obs = self.get_novelty(obs)
            if self.normalize and self.update_step > self.warm_up:
                self.rnd_rms.update(novelty_obs)
                novelty_obs = self.rnd_rms.normalize(novelty_obs)
            reward_intrinsic = novelty_obs.unsqueeze(1)
            return reward_intrinsic

        elif self.type == 'noveld':
            assert next_obs is not None
            novelty_obs = self.get_novelty(obs)
            novelty_nextobs = self.get_novelty(next_obs)

            if self.normalize and self.update_step > self.warm_up:
                self.rnd_rms.update(novelty_obs)
                self.rnd_rms.update(novelty_nextobs)
                novelty_obs = self.rnd_rms.normalize(novelty_obs)
                novelty_nextobs = self.rnd_rms.normalize(novelty_nextobs)

            intrinsic = novelty_nextobs - 0.5 * novelty_obs
            reward_intrinsic = 0.01 * torch.max(intrinsic, torch.zeros(intrinsic.shape, device=intrinsic.device)).unsqueeze(1)
            return reward_intrinsic

        else:
            raise NotImplementedError

    def get_novelty(self, obs):
        predict_obs, target_obs = self.rnd_model(obs)
        novelty = torch.norm(predict_obs - target_obs, dim=1, p=2).detach()
        return novelty

    def update(self, obs):


        predict_feature, target_feature = self.rnd_model(obs)
        dynamic_loss = F.mse_loss(predict_feature, target_feature.detach())
        dynamic_grad_norm = self.optimizer_update(self.rnd_optimizer, dynamic_loss)
        self.update_step += 1
        return dynamic_loss.item(), dynamic_grad_norm.item()

    def optimizer_update(self, optimizer, objective):
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        grad_norm = clip_grad_norm_(parameters=optimizer.param_groups[0]["params"],
                                    max_norm=1.0)
        optimizer.step()
        return grad_norm
