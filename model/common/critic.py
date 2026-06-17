

from collections.abc import Sequence
from typing import Union
import torch
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.common.modules import SpatialEmb, RandomShiftsAug

class MLPNet(torch.nn.Module):
    def __init__(
            self,
            cond_dim,
            mlp_dims,
            activation_type="Mish",
            use_layernorm=False,
            residual_style=False,
            out_dim=1,
            **kwargs,
    ):
        super().__init__()
        mlp_dims = [cond_dim] + mlp_dims + [out_dim]
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.net = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, x):
        return self.net(x)


class CriticObs(torch.nn.Module):


    def __init__(
        self,
        cond_dim,
        mlp_dims,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [cond_dim] + mlp_dims + [1]
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.Q1 = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, cond: Union[dict, torch.Tensor]):


        if isinstance(cond, dict):
            B = len(cond["state"])


            state = cond["state"].view(B, -1)
        else:
            state = cond
        q1 = self.Q1(state)
        return q1


class CriticObsAct(torch.nn.Module):


    def __init__(
        self,
        cond_dim,
        mlp_dims,
        action_dim,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_tyle=False,
        double_q=True,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [cond_dim + action_dim * action_steps] + mlp_dims + [1]
        if residual_tyle:
            model = ResidualMLP
        else:
            model = MLP
        self.Q1 = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )
        if double_q:
            self.Q2 = model(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )

    def forward(self, cond: dict, action):


        if isinstance(cond, dict):
            B = len(cond["state"])


            state = cond["state"].view(B, -1)
        else:
            B = cond.shape[0]

            state = cond.view(B, -1)


        action = action.view(B, -1)

        x = torch.cat((state, action), dim=-1)
        if hasattr(self, "Q2"):
            q1 = self.Q1(x)
            q2 = self.Q2(x)
            return q1.squeeze(1), q2.squeeze(1)
        else:
            q1 = self.Q1(x)
            return q1.squeeze(1)


class ViTCritic(CriticObs):


    def __init__(
        self,
        backbone,
        cond_dim,
        img_cond_steps=1,
        spatial_emb=128,
        dropout=0,
        augment=False,
        num_img=1,
        **kwargs,
    ):

        mlp_obs_dim = spatial_emb * num_img + cond_dim
        super().__init__(cond_dim=mlp_obs_dim, **kwargs)
        self.backbone = backbone
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if num_img > 1:
            self.compress1 = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            self.compress2 = deepcopy(self.compress1)
        else:
            self.compress = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment

    def forward(
        self,
        cond: dict,
        no_augment=False,
    ):


        B, T_rgb, C, H, W = cond["rgb"].shape


        state = cond["state"].view(B, -1)


        rgb = cond["rgb"][:, -self.img_cond_steps :]


        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")


        rgb = rgb.float()


        if self.num_img > 1:
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment and not no_augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:
            if self.augment and not no_augment:
                rgb = self.aug(rgb)
            feat = self.backbone(rgb)
            feat = self.compress.forward(feat, state)
        feat = torch.cat([feat, state], dim=-1)
        return super().forward(feat)


class DoubleQ(torch.nn.Module):
    def __init__(
        self,
        cond_dim,
        action_dim,
        mlp_dims,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        **kwargs,
    ):
        super().__init__()
        self.net_q1 = MLPNet(
            cond_dim=cond_dim + action_dim * action_steps,
            mlp_dims=mlp_dims,
            activation_type=activation_type,
            use_layernorm=use_layernorm,
            residual_style=residual_style,
        )
        self.net_q2 = MLPNet(
            cond_dim=cond_dim + action_dim * action_steps,
            mlp_dims=mlp_dims,
            activation_type=activation_type,
            use_layernorm=use_layernorm,
            residual_style=residual_style,
        )

        self.net_v = MLPNet(
            cond_dim=cond_dim,
            mlp_dims=mlp_dims,
            activation_type=activation_type,
            use_layernorm=use_layernorm,
            residual_style=residual_style,
        )


    def get_q_min(self, state, action):

        return torch.min(*self.get_q1_q2(state, action))


    def forward(self, state, action):

        return torch.min(*self.get_q1_q2(state, action))

    def get_q1_q2(self, cond, action):
        if isinstance(cond, dict):
            B = len(cond["state"])


            state = cond["state"].view(B, -1)
        else:
            B = cond.shape[0]


            state = cond.view(B, -1)

        action = action.view(B, -1)
        input_x = torch.cat((state, action), dim=1)
        q1 = self.net_q1(input_x)
        q2 = self.net_q2(input_x)
        return q1.squeeze(1), q2.squeeze(1)

    def get_q1(self, cond, action):
        if isinstance(cond, dict):
            B = len(cond["state"])


            state = cond["state"].view(B, -1)
        else:
            B = cond.shape[0]

            state = cond.view(B, -1)

        action = action.view(B, -1)
        input_x = torch.cat((state, action), dim=1)
        q1 = self.net_q1(input_x)
        return q1.squeeze(1)

    def get_v(self, cond):


        if isinstance(cond, dict):
            B = len(cond["state"])


            state = cond["state"].view(B, -1)
        else:


            state = cond
        v = self.net_v(state)
        return v


class DistributionalDoubleQ(torch.nn.Module):
    def __init__(
        self,
        cond_dim,
        action_dim,
        mlp_dims,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        v_min=-10,
        v_max=10,
        num_atoms=51,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.net_q1 = MLPNet(
            cond_dim=cond_dim + action_dim * action_steps,
            mlp_dims=mlp_dims,
            activation_type=activation_type,
            use_layernorm=use_layernorm,
            residual_style=residual_style,
            out_dim=num_atoms,
        )
        self.net_q2 = MLPNet(
            cond_dim=cond_dim + action_dim * action_steps,
            mlp_dims=mlp_dims,
            activation_type=activation_type,
            use_layernorm=use_layernorm,
            residual_style=residual_style,
            out_dim=num_atoms,
        )

        self.net_v = MLPNet(
            cond_dim=cond_dim,
            mlp_dims=mlp_dims,
            activation_type=activation_type,
            use_layernorm=use_layernorm,
            residual_style=residual_style,
            out_dim=1,
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.register_buffer("z_atoms", torch.linspace(v_min, v_max, num_atoms).to(torch.float32))


    def get_q_min(self, cond, action):


        q1, q2 = self.get_q1_q2(cond, action)

        z_atoms = self.z_atoms.to(self.device)

        expected_q1 = torch.sum(q1 * z_atoms, dim=1)
        expected_q2 = torch.sum(q2 * z_atoms, dim=1)


        q_min = torch.min(expected_q1, expected_q2)


        return q_min

    def forward(self, state, action):

        return torch.min(*self.get_q1_q2(state, action))

    def get_q1_q2(self, cond, action):

        if isinstance(cond, dict):
            B = len(cond["state"])
            state = cond["state"].view(B, -1)
        else:
            B = cond.shape[0]
            state = cond.view(B, -1)
        action = action.view(B, -1)


        input_x = torch.cat((state, action), dim=1)


        q1 = self.net_q1(input_x)
        q2 = self.net_q2(input_x)

        prob1 = torch.softmax(q1, dim=1)
        prob2 = torch.softmax(q2, dim=1)

        return prob1, prob2

    def get_q1_q2_logits(self, cond, action):

        if isinstance(cond, dict):
            B = len(cond["state"])
            state = cond["state"].view(B, -1)
        else:
            B = cond.shape[0]
            state = cond.view(B, -1)
        action = action.view(B, -1)
        input_x = torch.cat((state, action), dim=1)
        return self.net_q1(input_x), self.net_q2(input_x)


    def get_q1(self, cond, action):
        if isinstance(cond, dict):
            B = len(cond["state"])


            state = cond["state"].view(B, -1)
        else:
            B = cond.shape[0]

            state = cond.view(B, -1)

        action = action.view(B, -1)
        input_x = torch.cat((state, action), dim=1)
        q1 = self.net_q1(input_x)
        return torch.softmax(q1, dim=1)

    def get_v(self, cond):


        if isinstance(cond, dict):
            B = len(cond["state"])


            state = cond["state"].view(B, -1)
        else:


            state = cond
        v = self.net_v(state)
        return v
