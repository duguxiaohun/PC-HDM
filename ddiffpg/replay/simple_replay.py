import numpy as np
import torch
from copy import deepcopy


def create_buffer(capacity, obs_dim, action_dim, cond_steps=1, horizon_steps=4, device='cuda'):
    if isinstance(capacity, int):
        capacity = (capacity,)
    buf_obs_size = (*capacity, cond_steps, obs_dim) if isinstance(obs_dim, int) else (*capacity, *obs_dim)
    buf_obs = torch.empty(buf_obs_size,
                          dtype=torch.float32, device=device)
    buf_action = torch.empty((*capacity, horizon_steps, int(action_dim)),
                             dtype=torch.float32, device=device)
    buf_reward = torch.empty((*capacity, 1),
                             dtype=torch.float32, device=device)
    buf_next_obs = torch.empty(buf_obs_size,
                               dtype=torch.float32, device=device)
    buf_done = torch.empty((*capacity, 1),
                           dtype=torch.bool, device=device)
    return buf_obs, buf_action, buf_next_obs, buf_reward, buf_done


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, cond_steps=1, horizon_steps=4, device='cpu'):
        self.obs_dim = obs_dim
        if isinstance(obs_dim, int):
            self.obs_dim = (self.obs_dim,)
        self.action_dim = action_dim
        self.device = device
        self.next_p = 0
        self.if_full = False
        self.cur_capacity = 0
        self.capacity = int(capacity)
        self.total_samples = 0
        self.sample_idx = None
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps


        ret = create_buffer(capacity=self.capacity, obs_dim=obs_dim, action_dim=action_dim, cond_steps=cond_steps, horizon_steps=horizon_steps, device=device)
        self.buf_obs, self.buf_action, self.buf_next_obs, self.buf_reward, self.buf_done = ret
        self.buf_target_action = torch.empty_like(self.buf_action)

    @torch.no_grad()
    def add_to_buffer(self, trajectory):
        obs, actions, rewards, next_obs, dones = trajectory
        obs = obs.reshape(-1, self.cond_steps, *self.obs_dim)
        actions = actions.reshape(-1, self.horizon_steps, self.action_dim)
        rewards = rewards.reshape(-1, 1)
        next_obs = next_obs.reshape(-1, self.cond_steps, *self.obs_dim)
        dones = dones.reshape(-1, 1).bool()

        p = self.next_p + rewards.shape[0]
        self.total_samples += rewards.shape[0]

        if p > self.capacity:
            self.if_full = True
            overflow = self.capacity - self.next_p

            self.buf_obs[self.next_p:self.capacity] = obs[:overflow]
            self.buf_action[self.next_p:self.capacity] = actions[:overflow]
            self.buf_target_action[self.next_p:self.capacity] = actions[:overflow]
            self.buf_reward[self.next_p:self.capacity] = rewards[:overflow]
            self.buf_next_obs[self.next_p:self.capacity] = next_obs[:overflow]
            self.buf_done[self.next_p:self.capacity] = dones[:overflow]

            remain = rewards.shape[0] - overflow
            self.buf_obs[0:remain] = obs[-remain:]
            self.buf_action[0:remain] = actions[-remain:]
            self.buf_target_action[0:remain] = actions[-remain:]
            self.buf_reward[0:remain] = rewards[-remain:]
            self.buf_next_obs[0:remain] = next_obs[-remain:]
            self.buf_done[0:remain] = dones[-remain:]
            p = remain
        else:
            self.buf_obs[self.next_p:p] = obs
            self.buf_action[self.next_p:p] = actions
            self.buf_target_action[self.next_p:p] = actions
            self.buf_reward[self.next_p:p] = rewards
            self.buf_next_obs[self.next_p:p] = next_obs
            self.buf_done[self.next_p:p] = dones

        self.next_p = p
        self.cur_capacity = self.capacity if self.if_full else self.next_p


    @torch.no_grad()
    def sample_batch(self, batch_size, device='cuda'):
        indices = torch.randint(self.cur_capacity, size=(batch_size,), device=device)
        self.sample_idx = indices

        return (
            self.buf_obs[indices].to(device),
            self.buf_action[indices].to(device),
            self.buf_target_action[indices].to(device),
            self.buf_reward[indices].to(device),
            self.buf_next_obs[indices].to(device),
            self.buf_done[indices].to(device).float()
        )

    @torch.no_grad()
    def update_target_action(self, new_action):
        self.buf_target_action[self.sample_idx] = new_action


class DiffusionReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, cond_steps=1, horizon_steps=4, device='cpu'):
        self.obs_dim = obs_dim if not isinstance(obs_dim, int) else (obs_dim,)
        self.action_dim = action_dim
        self.device = device
        self.capacity = int(capacity)
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        self.ptr = 0
        self.size = 0
        self.indices = None


        self.buf_neighbor_trajs = torch.zeros((capacity, 6, 10, 5), dtype=torch.float32, device=device)
        self.buf_ego_state = torch.zeros((capacity, 5), dtype=torch.float32, device=device)
        self.buf_neighbor_waypoints = torch.zeros((capacity, 18, 10, 2), dtype=torch.float32, device=device)

        self.buf_action = torch.zeros((capacity, horizon_steps, action_dim), dtype=torch.float32, device=device)
        self.buf_target_action = torch.zeros((capacity, horizon_steps, action_dim), dtype=torch.float32, device=device)
        self.buf_reward = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.buf_next_neighbor_trajs = torch.zeros((capacity, 6, 10, 5), dtype=torch.float32, device=device)
        self.buf_next_ego_state = torch.zeros((capacity, 5), dtype=torch.float32, device=device)
        self.buf_next_neighbor_waypoints = torch.zeros((capacity, 18, 10, 2), dtype=torch.float32, device=device)
        self.buf_done = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    @torch.no_grad()
    def add_to_buffer(self, trajectory):
        obs, actions, target_actions, rewards, next_obs, dones = trajectory
        neighbor_trajs, ego_state, neighbor_waypoints = obs['neighbor_trajs'], obs['ego_state'], obs['neighbor_waypoints']
        neighbor_trajs = torch.from_numpy(neighbor_trajs).to(self.device).float().reshape(-1, 6, 10, 5)
        ego_state = torch.from_numpy(ego_state).to(self.device).float().reshape(-1, 5)
        neighbor_waypoints = torch.from_numpy(neighbor_waypoints).to(self.device).float().reshape(-1, 18, 10, 2)

        actions = torch.from_numpy(actions).to(self.device).float().reshape(-1, self.horizon_steps, self.action_dim)
        target_actions = torch.from_numpy(target_actions).to(self.device).float().reshape(-1, self.horizon_steps,
                                                                                          self.action_dim)
        rewards = torch.from_numpy(rewards).to(self.device).float().reshape(-1, 1)

        next_neighbor_trajs, next_ego_state, next_neighbor_waypoints = next_obs['neighbor_trajs'], next_obs['ego_state'], next_obs['neighbor_waypoints']
        next_neighbor_trajs = torch.from_numpy(next_neighbor_trajs).to(self.device).float().reshape(-1, 6, 10, 5)
        next_ego_state = torch.from_numpy(next_ego_state).to(self.device).float().reshape(-1, 5)
        next_neighbor_waypoints = torch.from_numpy(next_neighbor_waypoints).to(self.device).float().reshape(-1, 18, 10, 2)
        dones = torch.from_numpy(dones).to(self.device).float().reshape(-1, 1)

        batch_size = neighbor_trajs.shape[0]


        insert_idx = (self.ptr + torch.arange(batch_size)) % self.capacity

        self.buf_neighbor_trajs[insert_idx] = neighbor_trajs
        self.buf_ego_state[insert_idx] = ego_state
        self.buf_neighbor_waypoints[insert_idx] = neighbor_waypoints

        self.buf_action[insert_idx] = actions
        self.buf_target_action[insert_idx] = target_actions
        self.buf_reward[insert_idx] = rewards
        self.buf_next_neighbor_trajs[insert_idx] = next_neighbor_trajs
        self.buf_next_ego_state[insert_idx] = next_ego_state
        self.buf_next_neighbor_waypoints[insert_idx] = next_neighbor_waypoints
        self.buf_done[insert_idx] = dones

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    @torch.no_grad()
    def sample_batch(self, batch_size, device='cuda'):
        if self.size < batch_size:
            raise ValueError(f"Not enough samples to sample: have {self.size}, need {batch_size}")
        device = torch.device(device)
        indices = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        self.indices = indices
        return (
            {
                "neighbor_trajs": self.buf_neighbor_trajs[indices].to(device),
                "ego_state": self.buf_ego_state[indices].to(device),
                "neighbor_waypoints": self.buf_neighbor_waypoints[indices].to(device),
            },
            self.buf_action[indices].to(device),
            self.buf_target_action[indices].to(device),
            self.buf_reward[indices].to(device),
            {
                "neighbor_trajs": self.buf_next_neighbor_trajs[indices].to(device),
                "ego_state": self.buf_next_ego_state[indices].to(device),
                "neighbor_waypoints": self.buf_next_neighbor_waypoints[indices].to(device),
            },
            self.buf_done[indices].to(device),
        )

    @torch.no_grad()
    def sample_sequence_batch(self, batch_size, pred_step=3, device='cuda'):
        pred_step = int(pred_step)
        if pred_step < 1:
            raise ValueError(f"pred_step must be at least 1, got {pred_step}")
        if pred_step == 1:
            return self.sample_batch(batch_size, device=device)
        if self.size < pred_step:
            raise ValueError(
                f"Not enough samples for {pred_step}-step sequences: have {self.size}"
            )

        device = torch.device(device)
        arange = torch.arange(pred_step, device=self.device)

        if self.size < self.capacity:
            starts = torch.arange(0, self.size - pred_step + 1, device=self.device)
        else:
            starts_all = torch.arange(0, self.capacity, device=self.device)
            no_wrap_before_ptr = starts_all + pred_step <= self.ptr
            no_wrap_after_ptr = (starts_all >= self.ptr) & (starts_all + pred_step <= self.capacity)
            starts = starts_all[no_wrap_before_ptr | no_wrap_after_ptr]

        if starts.numel() == 0:
            raise ValueError(f"No valid {pred_step}-step sequences are available")

        seq = starts.unsqueeze(1) + arange.unsqueeze(0)
        done_seq = self.buf_done[seq[:, :-1]].squeeze(-1)
        valid = ~done_seq.bool().any(dim=1)
        starts = starts[valid]
        if starts.numel() == 0:
            raise ValueError(
                f"No valid {pred_step}-step sequences remain after done-boundary filtering"
            )

        choice = torch.randint(0, starts.numel(), size=(batch_size,), device=self.device)
        start_indices = starts[choice]
        seq_indices = start_indices.unsqueeze(1) + arange.unsqueeze(0)
        self.indices = start_indices

        return (
            {
                "neighbor_trajs": self.buf_neighbor_trajs[start_indices].to(device),
                "ego_state": self.buf_ego_state[start_indices].to(device),
                "neighbor_waypoints": self.buf_neighbor_waypoints[start_indices].to(device),
            },
            self.buf_action[seq_indices, 0].to(device),
            self.buf_target_action[seq_indices, 0].to(device),
            self.buf_reward[start_indices].to(device),
            {
                "neighbor_trajs": self.buf_next_neighbor_trajs[seq_indices].to(device),
                "ego_state": self.buf_next_ego_state[seq_indices].to(device),
                "neighbor_waypoints": self.buf_next_neighbor_waypoints[seq_indices].to(device),
            },
            self.buf_done[start_indices].to(device),
        )

    def get_buffer_size(self):
        return self.size

    @torch.no_grad()
    def clear(self):
        self.ptr = 0
        self.size = 0
        self.indices = None

    @torch.no_grad()
    def update_target_action(self, new_action):
        if self.indices is None:
            raise RuntimeError("sample_batch must be called before update_target_action")
        self.buf_target_action[self.indices] = new_action.to(self.device)


class HybridMemoryDiffusionReplayBuffer:
    """
    Paper-aligned Hybrid Memory wrapper.

    B_long stores broad off-policy rollouts for the ordinal critic and
    Q-guided value distillation. B_short stores recent on-policy rollouts for
    denoising PPO and can be cleared after each policy update.
    """

    def __init__(
        self,
        long_capacity: int,
        short_capacity: int,
        obs_dim: int,
        action_dim: int,
        cond_steps=1,
        horizon_steps=4,
        device='cpu',
    ):
        if long_capacity <= short_capacity:
            raise ValueError(
                "Hybrid memory expects asymmetric capacities: "
                f"long_capacity={long_capacity}, short_capacity={short_capacity}"
            )
        self.long = DiffusionReplayBuffer(
            capacity=long_capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            cond_steps=cond_steps,
            horizon_steps=horizon_steps,
            device=device,
        )
        self.short = DiffusionReplayBuffer(
            capacity=short_capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            cond_steps=cond_steps,
            horizon_steps=horizon_steps,
            device=device,
        )

    @torch.no_grad()
    def add_to_buffer(self, trajectory):
        self.long.add_to_buffer(trajectory)
        self.short.add_to_buffer(trajectory)

    def sample_long_batch(self, batch_size, device='cuda'):
        return self.long.sample_batch(batch_size, device=device)

    def sample_short_batch(self, batch_size, device='cuda'):
        return self.short.sample_batch(batch_size, device=device)

    def sample_target_action_batch(self, batch_size, device='cuda'):
        return self.long.sample_batch(batch_size, device=device)

    def sample_representation_sequence(self, batch_size, pred_step=3, device='cuda'):
        return self.long.sample_sequence_batch(batch_size, pred_step=pred_step, device=device)

    @torch.no_grad()
    def update_long_target_action(self, new_action):
        self.long.update_target_action(new_action)

    @torch.no_grad()
    def update_target_action_memory(self, new_action):
        self.long.update_target_action(new_action)

    @torch.no_grad()
    def clear_short(self):
        self.short.clear()

    def get_long_size(self):
        return self.long.get_buffer_size()

    def get_short_size(self):
        return self.short.get_buffer_size()

    def get_target_action_size(self):
        return self.long.get_buffer_size()


class DualDiffusionReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, cond_steps=1, horizon_steps=4, device='cpu'):
        self.main = DiffusionReplayBuffer(
            capacity=capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            cond_steps=cond_steps,
            horizon_steps=horizon_steps,
            device=device,
        )
        self.success = DiffusionReplayBuffer(
            capacity=capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            cond_steps=cond_steps,
            horizon_steps=horizon_steps,
            device=device,
        )

    @torch.no_grad()
    def add_to_buffer(self, trajectory, success=False):
        self.main.add_to_buffer(trajectory)
        if success:
            self.success.add_to_buffer(trajectory)

    @torch.no_grad()
    def add_success_to_buffer(self, trajectory):
        self.success.add_to_buffer(trajectory)

    def sample_batch(self, batch_size, device='cuda'):
        return self.main.sample_batch(batch_size, device=device)

    def sample_sequence_batch(self, batch_size, pred_step=3, device='cuda'):
        return self.main.sample_sequence_batch(batch_size, pred_step=pred_step, device=device)

    def sample_success_batch(self, batch_size, device='cuda'):
        return self.success.sample_batch(batch_size, device=device)

    @torch.no_grad()
    def update_target_action(self, new_action):
        self.main.update_target_action(new_action)

    def get_buffer_size(self):
        return self.main.get_buffer_size()

    def get_success_buffer_size(self):
        return self.success.get_buffer_size()
