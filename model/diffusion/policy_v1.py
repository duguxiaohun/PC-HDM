import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Rotater(nn.Module):
    def __init__(self, name='rotater', mode='traj', aug=False, carla=False):
        super().__init__()
        self.name = name
        self.mode = mode
        self.random_aug = aug
        self.carla = carla

    def forward(self, states, mask, curr_frame=None, aug=True, random_angle=None):
        if self.mode == 'traj':
            return self._make_rotations(states, mask, curr_frame, aug)
        elif self.mode == 'fut':
            return self._rotate_fut(states, curr_frame, aug)
        else:
            return self._make_map_rotations(states, mask, curr_frame, aug, random_angle)

    def _rotate_fut(self, states, curr_frames, aug=True):


        mask = (states != 0)[:, :, 0]

        mask = mask.float()
        yaw = curr_frames[:, 2]
        cos_a = torch.cos(yaw).unsqueeze(-1)
        sin_a = torch.sin(yaw).unsqueeze(-1)

        x = states[:, :, 0] - curr_frames[:, 0].unsqueeze(-1)
        y = states[:, :, 1] - curr_frames[:, 1].unsqueeze(-1)
        new_x = cos_a * x + sin_a * y
        new_y = -sin_a * x + cos_a * y
        rotated_state = torch.stack([new_x, new_y], dim=-1)
        mask = mask.unsqueeze(-1)
        return rotated_state * mask, mask

    def _make_rotations(self, states, mask, curr_frame=None, aug=True):


        mask = mask.to(torch.int32)
        ind = mask[:, 0, :].sum(dim=-1)

        ind = torch.clamp(ind - 1, 0, 100)


        if curr_frame is None:
            batch_indices = torch.arange(mask.size(0)).to(mask.device)
            curr_frames = states[batch_indices, 0, ind]

        else:

            curr_frames = curr_frame
        if self.random_aug:
            r_range = np.pi / 2
            random_angle = torch.empty_like(curr_frames[:, 2]).uniform_(-r_range, r_range)
            cos_r = torch.cos(random_angle).unsqueeze(-1).unsqueeze(-1)
            sin_r = torch.sin(random_angle).unsqueeze(-1).unsqueeze(-1)


        yaw = curr_frames[:, 2]
        cos_a = torch.cos(yaw).unsqueeze(-1).unsqueeze(-1)
        sin_a = torch.sin(yaw).unsqueeze(-1).unsqueeze(-1)
        x = states[:, :, :, 0] - curr_frames[:, 0].unsqueeze(-1).unsqueeze(-1)

        y = states[:, :, :, 1] - curr_frames[:, 1].unsqueeze(-1).unsqueeze(-1)
        angle = states[:, :, :, 2] - yaw.unsqueeze(-1).unsqueeze(-1)
        if aug:
            new_x = cos_a * x + sin_a * y
            new_y = -sin_a * x + cos_a * y
            if self.random_aug:
                n_x = cos_r * new_x + sin_r * new_y
                n_y = -sin_r * new_x + cos_r * new_y
                new_x, new_y = n_x, n_y
        else:
            new_x, new_y = x, y
        vx = states[:, :, :, 3] - curr_frames[:, 3].unsqueeze(-1).unsqueeze(-1)
        vy = states[:, :, :, 4] - curr_frames[:, 4].unsqueeze(-1).unsqueeze(-1)
        if aug:
            new_vx = cos_a * vx + sin_a * vy
            new_vy = -sin_a * vx + cos_a * vy
            if self.random_aug:
                n_vx = cos_r * new_vx + sin_r * new_vy
                n_vy = -sin_r * new_vx + cos_r * new_vy
                new_vx, new_vy = n_vx, n_vy
        else:
            new_vx, new_vy = vx, vy
        rotated_state = torch.stack([-new_x, new_y, angle, -new_vx, new_vy], dim=-1)
        if not self.carla:
            rotated_state = torch.stack([new_x, new_y, angle, new_vx, new_vy], dim=-1)
        mask = mask.unsqueeze(-1).float()
        if self.random_aug:
            return rotated_state * mask, curr_frames, random_angle
        return rotated_state * mask, curr_frames, None

    def _make_map_rotations(self, states, mask, curr_frames, aug=True, random_angle=None):


        yaw = curr_frames[:, 2]

        cos_a = torch.cos(yaw).unsqueeze(-1).unsqueeze(-1)
        sin_a = torch.sin(yaw).unsqueeze(-1).unsqueeze(-1)


        x = states[:, :, :, 0] - curr_frames[:, 0].unsqueeze(-1).unsqueeze(-1)
        y = states[:, :, :, 1] - curr_frames[:, 1].unsqueeze(-1).unsqueeze(-1)

        if self.random_aug:
            cos_r = torch.cos(random_angle).unsqueeze(-1).unsqueeze(-1)
            sin_r = torch.sin(random_angle).unsqueeze(-1).unsqueeze(-1)

        if aug:
            new_x = cos_a * x + sin_a * y
            new_y = -sin_a * x + cos_a * y
            if self.random_aug:
                n_x = cos_r * new_x + sin_r * new_y
                n_y = -sin_r * new_x + cos_r * new_y
                new_x, new_y = n_x, n_y
        else:
            new_x, new_y = x, y
        rotated_state = torch.stack([-new_x, new_y], dim=-1)

        if not self.carla:
            angle = states[:, :, :, 2] - yaw.unsqueeze(-1).unsqueeze(-1)
            rotated_state = torch.stack([new_x, new_y, angle, states[:, :, :, 3], states[:, :, :, 4]], dim=-1)
        mask = mask.unsqueeze(-1).float()

        return rotated_state * mask


class MapEncoder(nn.Module):
    def __init__(self, return_attention_scores=False, carla=False):
        super(MapEncoder, self).__init__()
        self.return_attention_scores = return_attention_scores
        self.carla = carla
        if self.carla:
            self.self_line = nn.Linear(2, 3*64)
        else:
            self.self_line = nn.Linear(3, 128 + 64)


        self.node_attention = nn.MultiheadAttention(embed_dim=3*64, num_heads=2, dropout=0, batch_first=True)
        self.flatten = nn.AdaptiveMaxPool1d(1)
        self.vector_feature = nn.Linear(2, 64)
        self.sublayer = nn.Linear(64*4, 128)

    def forward(self, inputs, mask, test):

        inputs = inputs.to(device)
        mask = mask.to(device)

        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.bool)


        if self.carla:
            nodes = inputs[:, :, :2]

        else:
            nodes = inputs[:, :, :3]

        nodes = self.self_line(nodes)


        mask = (~mask).float()

        if self.return_attention_scores:
            nodes, attention_weights = self.node_attention(
                nodes, nodes, nodes,
                key_padding_mask=mask,

            )

        else:
            nodes, _ = self.node_attention(
                nodes, nodes, nodes,
                key_padding_mask=mask,
            )

        nodes = F.relu(nodes)
        nodes = self.flatten(nodes.transpose(1,2)).squeeze(-1)


        vector = self.vector_feature(inputs[:, 0, -2:])


        out = torch.cat([nodes, vector], dim=1)

        polyline_feature = self.sublayer(out)

        if self.return_attention_scores:

            attention_weights = attention_weights.mean(dim=1)


            return polyline_feature, attention_weights
        return polyline_feature


class MultiModal_Attention(nn.Module):
    def __init__(self, num_modes, key_dim, head_num=1):
        super(MultiModal_Attention, self).__init__()
        self.num_modes = num_modes
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=key_dim, num_heads=head_num, batch_first=True)
            for _ in range(num_modes)
        ])
        self.norm1 = nn.LayerNorm(key_dim)
        self.norm2 = nn.LayerNorm(key_dim)

        self.dropout1 = nn.Dropout(0.1)
        self.FFN2 = nn.Linear(key_dim, key_dim)
        self.dropout2 = nn.Dropout(0.1)


    def forward(self, query, key, mask=None, training=True):
        output = []
        for i in range(self.num_modes):
            value, _ = self.attention[i](query, key, key, key_padding_mask=~mask)

            output.append(value.squeeze(1))

        value = F.relu(torch.stack(output, dim=1))

        return value, None


import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(d_model, out_dim)
        self.relu = nn.LeakyReLU(0.01)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class Hierachial_Transformer(nn.Module):
    def __init__(self, state_shape, name='hier_encoder', units=256,
                 use_trans_encode=False, num_heads=2, drop_rate=0.1, neighbours=5,
                 make_rotation=True, time_step=8, num_modes=1, final_head_num=2,
                 random_aug=False, no_ego_fut=False, no_neighbor_fut=False, carla=False,
                 leaky_relu_slope=0.01, rezero=True, n_heads=2):
        super(Hierachial_Transformer, self).__init__()
        self.inp_dim = state_shape[-1]

        self.n_heads = n_heads
        self.map_layer = MapEncoder(return_attention_scores=True, carla=carla)

        self.neighbours = neighbours
        self.make_rotation = make_rotation
        self.time_step = time_step
        self.embedder = nn.Linear(self.inp_dim, units)


        self.time_layer = nn.MultiheadAttention(embed_dim=units, num_heads=num_heads, dropout=0, batch_first=True)
        self.time_pooling = nn.AdaptiveMaxPool1d(1)

        self.use_trans = use_trans_encode
        self.rel_layer = nn.MultiheadAttention(embed_dim=units, num_heads=num_heads, dropout=0, batch_first=True)

        if self.make_rotation:
            self.rotater = Rotater(mode='traj', aug=random_aug, carla=carla)
            self.map_rotater = Rotater(mode='map', aug=random_aug, carla=carla)

        self.map_attention = nn.MultiheadAttention(embed_dim=units, num_heads=num_heads, dropout=0, batch_first=True)

        self.final_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=units, num_heads=final_head_num, dropout=0, batch_first=True)
            for _ in range(num_modes)
        ])
        self.num_modes = num_modes

        self.no_ego_fut = no_ego_fut
        self.no_neighbor_fur = no_neighbor_fut
        self.carla = carla

        self.att_proj = nn.Linear(self.inp_dim + 1,  1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.Rezero = nn.Parameter(torch.zeros(units)) if rezero else None
        self.LayerNorm = nn.LayerNorm(units)
        self.rezero_enabled = rezero
        self.FF = FeedForwardNetwork(units, units)
        self.edge_attr_proj = nn.Linear(6, units)
        self.D_head = units // self.n_heads
        self.att_proj_emb = nn.Parameter(torch.randn(self.n_heads, 3 * self.D_head))

    def compute_edge_attr(self, states):


        x = states[..., 0]
        y = states[..., 1]
        vx = states[..., 2]
        vy = states[..., 3]
        yaw = states[..., 4]


        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        vx = vx.permute(0, 2, 1)
        vy = vy.permute(0, 2, 1)
        yaw = yaw.permute(0, 2, 1)


        dx = x.unsqueeze(2) - x.unsqueeze(3)
        dy = y.unsqueeze(2) - y.unsqueeze(3)
        dvx = vx.unsqueeze(2) - vx.unsqueeze(3)
        dvy = vy.unsqueeze(2) - vy.unsqueeze(3)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        dyaw = yaw.unsqueeze(2) - yaw.unsqueeze(3)


        edge_attr = torch.stack([dx, dy, dvx, dvy, dist, dyaw], dim=-1)
        return edge_attr


    def forward(self, states, test=False, map_state=None, aug=True):

        training = not test
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float().to(device)
        else:
            states = states.to(device).float()
        if isinstance(map_state, np.ndarray):
            map_state = torch.from_numpy(map_state).float().to(device)
        else:
            map_state = map_state.to(device).float()


        mask = (states != 0)[:, :, :, 0]


        if self.make_rotation:
            states, curr_frames, rg = self.rotater(states, mask, aug=aug)


        embedder_state = self.embedder(states)

        ego_states, neighbor_states = states[:, 0, :, :], states[:, 1:, :, :]


        ego_mask, neighbor_mask = mask[:, 0, :], mask[:, 1:, :]


        ego_embedder, neighbor_embedder = embedder_state[:, 0, :, :], embedder_state[:, 1:, :, :]

        actor_mask = (torch.cat([torch.ones_like(ego_states).unsqueeze(1), neighbor_states], dim=1) != 0)[:, :, 0, 0]


        ego = self._timestep_attention(ego_embedder, training, ego_mask)


        neighbors = [
            self._timestep_attention(neighbor_embedder[:, i, :, :], training, neighbor_mask[:, i, :])
            for i in range(self.neighbours)
        ]


        map_mask = (map_state != 0)[:, :, :, 0]

        map_traj_mask = (map_state != 0)[:, :, 0, 0]

        if self.make_rotation:
            map_state = self.map_rotater(map_state, map_mask, curr_frames, aug, rg)

        map = []
        val = []
        for i in range(map_state.size(1)):
            m, v = self.map_layer(map_state[:, i], map_mask[:, i, :], test)


            map.append(m)
            if test:
                val.append(v)
        map = torch.stack(map, dim=1)


        if test:
            val = torch.stack(val, dim=1)


        if self.carla:
            ego_map, neighbor_map = map[:, :3, :], map[:, 3:, :]

            ego_map_traj_mask, neighbor_map_traj_mask = map_traj_mask[:, :3], map_traj_mask[:, 3:]


        else:
            ego_map, neighbor_map = map[:, :2, :], map[:, 2:, :]
            ego_map_traj_mask, neighbor_map_traj_mask = map_traj_mask[:, :2], map_traj_mask[:, 2:]

        neighbor_rel_val = [
            self._map_vehicle_rel(neighbors[i], neighbor_map, neighbor_map_traj_mask, i * 3)[0]
            for i in range(self.neighbours)


        ]

        if self.no_neighbor_fur:
            neighbor_rel_val = neighbors

        if True:
            neighbor_val = [
                self._map_vehicle_rel(neighbors[i], neighbor_map, neighbor_map_traj_mask, i * 3)[1]
                for i in range(self.neighbours)
            ]


        actor = torch.cat([ego.unsqueeze(1), torch.stack(neighbor_rel_val, dim=1)], dim=1)


        edge_attr = self.compute_edge_attr(states)
        edge_attr = edge_attr[:, -1, :, :]

        states = states[:, :, -1, :]

        B, N, _ = states.shape
        D = actor.size(-1)
        H = self.n_heads
        D_head = D // H


        actor_multi = actor.view(B, N, H, D_head)


        h_i = actor_multi.unsqueeze(2).expand(B, N, N, H, D_head)
        h_j = actor_multi.unsqueeze(1).expand(B, N, N, H, D_head)


        edge_attr_proj = self.edge_attr_proj(edge_attr).view(B, N, N, H, D_head)


        att_input = torch.cat([h_i, edge_attr_proj, h_j], dim=-1)


        att_logits = torch.einsum("bjnhd,hd->bjnh", att_input, self.att_proj_emb)
        att_logits = self.leaky_relu(att_logits)


        valid = (states != 0)[..., 0]
        att_logits = att_logits.masked_fill(~valid.unsqueeze(1).unsqueeze(-1), -1e9)


        att_weights = torch.softmax(att_logits, dim=2).unsqueeze(-1)


        v_j = actor_multi.unsqueeze(1).expand(B, N, N, H, D_head)


        att_output = (att_weights * v_j).sum(dim=2)
        actor = att_output.reshape(B, N, D)


        if self.rezero_enabled:
            actor = actor + self.Rezero * self.FF(actor)
        else:
            actor = actor + self.FF(actor)
        actor = self.LayerNorm(actor)

        actor_rel, _ = self.rel_layer(
            ego.unsqueeze(1),
            actor,
            actor,
            key_padding_mask=(~actor_mask).float()
        )

        actor_rel = F.relu(actor_rel.squeeze(1))


        goals, ego_val = self._goal_layer(actor_rel.unsqueeze(1), ego_map, ego_map_traj_mask.unsqueeze(1))


        ego_states = actor_rel.unsqueeze(1).repeat(1, self.num_modes, 1)

        if self.no_ego_fut:
            states = ego_states
        else:
            states = goals + ego_states
        if test:
            neighbor_val = [ego_val] + neighbor_val
            neighbor_val = torch.cat(neighbor_val, dim=-1).unsqueeze(-1)
            return states, neighbor_val

        return states

    def _timestep_attention(self, states, training, mask):


        t = states.shape[1]


        key_padding_mask = (~mask).float()


        causal_mask = torch.triu(torch.ones(t, t), diagonal=1).bool().to(states.device)


        attn_output, _ = self.time_layer(
            states, states, states,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask
        )


        attn_output = F.relu(attn_output)
        state_val = self.time_pooling(attn_output.transpose(1, 2)).squeeze(-1)

        return state_val

    def _map_vehicle_rel(self, value, map_state, map_mask, i):

        use_map = map_state[:, i:i + 3, :]
        use_map_mask = torch.cat([
            torch.ones_like(map_mask[:, 0]).unsqueeze(1),
            map_mask[:, i:i + 3]
        ], dim=1)

        mv_rel = torch.cat([value.unsqueeze(1), use_map], dim=1)

        key_padding_mask = (~use_map_mask).float()
        mv_val, val = self.map_attention(
            value.unsqueeze(1),
            mv_rel,
            mv_rel,
            key_padding_mask=key_padding_mask,

        )

        val = val.squeeze(-2)[:, 1:]
        mv_val = F.relu(mv_val.squeeze(1))


        return mv_val, val

    def _goal_layer(self, query, key, mask=None, training=True):


        output = []
        key_padding_mask = (~mask.squeeze(1)).float()


        v = []
        for i in range(self.num_modes):
            value, val = self.final_attention[i](
                query,
                key,
                key,
                key_padding_mask=key_padding_mask if mask is not None else None,

            )


            output.append(value.squeeze(1))
            v.append(val.squeeze(1))

        v = torch.stack(v, dim=1).mean(dim=-2)

        value = F.relu(torch.stack(output, dim=1))

        return value, v


class Represent_Learner(nn.Module):
    def __init__(
        self,
        encoder=None,
        target_encoder=None,
        pred_step=3,
        latent_dim=128,
        action_dim=2,
        hidden_dim=256,
        similarity_weight=1.0,
        info_nce_weight=0.1,
        cycle_weight=0.2,
        temperature=0.1,
        ema_tau=0.005,
    ):
        super().__init__()
        if pred_step < 1:
            raise ValueError("pred_step must be at least 1")
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.pred_step = int(pred_step)
        self.similarity_weight = float(similarity_weight)
        self.info_nce_weight = float(info_nce_weight)
        self.cycle_weight = float(cycle_weight)
        self.temperature = float(temperature)
        self.ema_tau = float(ema_tau)
        transition_dim = latent_dim + action_dim
        self.forward_dynamics = nn.Sequential(
            nn.Linear(transition_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.backward_dynamics = nn.Sequential(
            nn.Linear(transition_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, latent_dim),
        )

    @staticmethod
    def _transition(latent, action, dynamics):
        delta = dynamics(torch.cat([latent, action], dim=-1))
        return latent + delta

    def _encode(self, states, map_state, mask, test, init_state):
        if self.encoder is None:
            raise RuntimeError("encoder is required")
        latent, _ = self.encoder(
            states,
            mask=mask,
            test=test,
            init_state=init_state,
            map_state=map_state,
        )
        if latent.ndim == 3:
            latent = latent[:, 0]
        if latent.ndim != 2:
            raise ValueError(f"encoder output must be [B, D] or [B, 1, D], got {latent.shape}")
        return latent

    def _prepare_actions(self, actions, pred_steps):
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        if actions.ndim != 3:
            raise ValueError(f"actions must be [B, T, A], got {actions.shape}")
        if actions.shape[1] < pred_steps:
            raise ValueError(
                f"actions provides {actions.shape[1]} steps, but {pred_steps} are required"
            )
        return actions[:, :pred_steps]

    @staticmethod
    def _prepare_future_sequence(values, current_values, steps, name):
        if values.ndim == current_values.ndim:
            values = values.unsqueeze(1)
        if values.ndim != current_values.ndim + 1:
            raise ValueError(f"{name} must include a prediction-step dimension")
        if values.shape[1] < steps:
            raise ValueError(
                f"{name} provides {values.shape[1]} steps, but {steps} are required"
            )
        return values[:, :steps]

    def _encode_targets(
        self,
        next_states,
        next_map_state,
        mask,
        test,
        init_state,
        steps,
    ):
        if self.target_encoder is None:
            raise RuntimeError("target_encoder is required for representation learning")

        target_predictions = []
        with torch.no_grad():
            for step in range(steps):
                target_latent, _ = self.target_encoder(
                    next_states[:, step],
                    mask=mask,
                    test=test,
                    init_state=init_state,
                    map_state=next_map_state[:, step],
                )
                if target_latent.ndim == 3:
                    target_latent = target_latent[:, 0]
                if target_latent.ndim != 2:
                    raise ValueError(
                        "target encoder output must be [B, D] or [B, 1, D]"
                    )
                target_predictions.append(target_latent)
        return torch.stack(target_predictions, dim=1)

    def _info_nce_loss(self, predictions, targets):
        predictions = F.normalize(predictions, dim=-1)
        targets = F.normalize(targets, dim=-1)
        losses = []
        labels = torch.arange(predictions.shape[0], device=predictions.device)
        for step in range(predictions.shape[1]):
            logits = predictions[:, step] @ targets[:, step].T
            losses.append(F.cross_entropy(logits / self.temperature, labels))
        return torch.stack(losses).mean()

    def _make_autonomous_forward_cycle_loss(
        self,
        states,
        map_state,
        actions,
        next_states=None,
        next_map_state=None,
        mask=None,
        test=False,
        init_state=None,
        pred_steps=None,
    ):
        steps = self.pred_step if pred_steps is None else int(pred_steps)
        if steps < 1:
            raise ValueError("pred_steps must be at least 1")
        actions = self._prepare_actions(actions, steps)
        initial_latent = self._encode(states, map_state, mask, test, init_state)

        latent = initial_latent
        forward_predictions = []
        for step in range(steps):
            latent = self._transition(latent, actions[:, step], self.forward_dynamics)
            forward_predictions.append(latent)
        forward_predictions = torch.stack(forward_predictions, dim=1)

        latent = forward_predictions[:, -1]
        backward_predictions = []
        for step in reversed(range(steps)):
            latent = self._transition(latent, actions[:, step], self.backward_dynamics)
            backward_predictions.append(latent)
        backward_predictions = torch.stack(backward_predictions, dim=1)
        reconstructed_latent = backward_predictions[:, -1]
        cycle_error = 1.0 - F.cosine_similarity(
            initial_latent,
            reconstructed_latent,
            dim=-1,
        )

        result = {
            "initial_latent": initial_latent,
            "forward_predictions": forward_predictions,
            "backward_predictions": backward_predictions,
            "reconstructed_latent": reconstructed_latent,
            "cycle_error": cycle_error,
            "cycle_loss": cycle_error.mean(),
        }
        if next_states is not None:
            if next_map_state is None:
                raise ValueError("next_map_state is required when next_states is provided")
            next_states = self._prepare_future_sequence(
                next_states, states, steps, "next_states"
            )
            next_map_state = self._prepare_future_sequence(
                next_map_state, map_state, steps, "next_map_state"
            )
            targets = self._encode_targets(
                next_states,
                next_map_state,
                mask,
                test,
                init_state,
                steps,
            )
            similarity_error = 1.0 - F.cosine_similarity(
                forward_predictions,
                targets,
                dim=-1,
            )
            similarity_loss = similarity_error.mean()
            info_nce_loss = self._info_nce_loss(forward_predictions, targets)
            result.update(
                {
                    "target_predictions": targets,
                    "similarity_loss": similarity_loss,
                    "info_nce_loss": info_nce_loss,
                    "loss": (
                        self.similarity_weight * similarity_loss
                        + self.info_nce_weight * info_nce_loss
                        + self.cycle_weight * result["cycle_loss"]
                    ),
                }
            )
        return result

    @torch.no_grad()
    def update_target_encoder(self, tau=None):
        if self.target_encoder is None:
            raise RuntimeError("target_encoder is required for EMA updates")
        tau = self.ema_tau if tau is None else float(tau)
        for target_param, source_param in zip(
            self.target_encoder.parameters(), self.encoder.parameters()
        ):
            target_param.data.lerp_(source_param.data, tau)

    def forward(
        self,
        prev_obs,
        actions,
        next_obs,
        mask=None,
        test=False,
        pred_steps=None,
        return_components=False,
    ):
        result = self._make_autonomous_forward_cycle_loss(
            states=prev_obs["neighbor_trajs"],
            map_state=prev_obs["neighbor_waypoints"],
            actions=actions,
            next_states=next_obs["neighbor_trajs"],
            next_map_state=next_obs["neighbor_waypoints"],
            mask=mask,
            test=test,
            init_state=prev_obs.get("ego_state"),
            pred_steps=pred_steps,
        )
        return result if return_components else result["loss"]


class RLEncoder(nn.Module):
    def __init__(self, state_shape, action_dim, units=[256] * 3, hidden_activation="relu", name='rl_encoder',
                 lstm=False, trans=False, cnn_lstm=False, ego_surr=False,
                 use_trans=False, neighbours=5, time_step=8, debug=False, make_rotation=True, make_prediction=False,
                 use_mask=False, use_map=False, num_traj=5, cnn=False, path_length=0, head_dim=1, use_hier=False,
                 random_aug=False, no_ego_fut=False, no_neighbor_fut=False, carla=False):
        super().__init__()
        self.lstm = lstm
        self.cnn = cnn
        self.cnn_lstm = cnn_lstm
        self.ego_surr = ego_surr
        self.trans = trans
        self.debug = debug
        self.use_map = use_map
        self.neighbours = neighbours
        self.num_traj = num_traj
        self.use_mask = use_mask
        self.use_hier = use_hier

        self.h_layer = Hierachial_Transformer(state_shape, units=128, use_trans_encode=True, num_heads=2,
                                                  drop_rate=0, neighbours=neighbours, make_rotation=make_rotation,
                                                  time_step=time_step, num_modes=num_traj,
                                                  final_head_num=head_dim, random_aug=random_aug,
                                                  no_ego_fut=no_ego_fut, no_neighbor_fut=no_neighbor_fut, carla=carla)

    def forward(self, states, mask=None, test=False, init_state=None, map_state=None, curr_frames=None, aug=True):


        if test:
            states, val = self.h_layer(states, test, map_state, aug)
            return states, val
        states = self.h_layer(states, test, map_state, aug)


        return states, None
