from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from termcolor import cprint
import time
import pytorch3d.ops as torch3d_ops
import einops

from bridge_policy_3d.model.common.normalizer import LinearNormalizer
from bridge_policy_3d.policy.base_policy import BasePolicy
from bridge_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from bridge_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from bridge_policy_3d.common.pytorch_util import dict_apply
from bridge_policy_3d.common.model_util import print_params
from bridge_policy_3d.model.vision.pointnet_extractor import BridgePolicyEncoder



class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()


class ClipLoss(nn.Module):
    """
    非分布式实现的ClipLoss，只支持单机单卡。
    """
    def __init__(self, cache_labels=False):
        super().__init__()
        self.cache_labels = cache_labels
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        if self.cache_labels:
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                self.labels[device] = labels
                self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]
        else:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class ActionAligner(torch.nn.Module):
    def __init__(self, input_dim, horizon, action_dim, hidden_dim=512, dropout=0.05):
        super().__init__()
        # 编码器

        self.input_dim = input_dim
        self.horizon = horizon
        self.action_dim = action_dim
        self.output_dim = horizon * action_dim

        self.obs_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

        self.action_encoder = nn.Linear(horizon * action_dim, self.output_dim)

        self.fc_mu = torch.nn.Linear(hidden_dim, self.output_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dim, self.output_dim)

    def encode(self, x):
        h = self.obs_encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def obs_forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


    def action_forward(self, x):
        x = x.reshape(-1, self.output_dim)
        return self.action_encoder(x)
    

    def forward(self, x, action_encoder=False, obs_encoder=False):
        if action_encoder:
            return self.action_forward(x)
        elif obs_encoder:
            return self.obs_forward(x)
        else:
            raise NotImplementedError('not implement for other modalities')



class BridgePolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        obs_encoder = BridgePolicyEncoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")



        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model

        # 将 aligner 定义为一个可直接调用的 torch.nn.Sequential 模块
        self.aligner = ActionAligner(global_cond_dim, horizon, action_dim)
        self.clip_loss = ClipLoss()

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        print_params(self)
        self.sample_type = 'sample' # self.sample_type = 'noise'
        # self.sde = sde_util.UniDB(lambda_square=opt["sde"]["lambda_square"], gamma=opt["sde"]["gamma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)

        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None, sde=None, 
            # keyword arguments to scheduler.step
            **kwargs
            ):

        condition_features, _, _ = self.aligner(global_cond, obs_encoder=True)
        condition_features = condition_features / condition_features.norm(dim=-1, keepdim=True)
        # condition_features torch.Size([128, 416]) action_features torch.Size([128, 416])
        condition_features = einops.rearrange(condition_features, 'b (h d) -> b h d', h=self.horizon)


        if self.sample_type == 'sample':
            pass
        else:
            pass

        # breakpoint()
        sde.set_mu(condition_features)
        sde.set_mu_prim(global_cond)

        self.model.eval()
        with torch.no_grad():
            # trajectory = sde.reverse_sde(condition_data)
            trajectory = sde.unidb_sde_solver_data_prediction(condition_data)

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], sde=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            sde=sde, 
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, sde):
        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        condition_features, mu, logvar = self.aligner(global_cond, obs_encoder=True)
        condition_features = condition_features / condition_features.norm(dim=-1, keepdim=True)
        action_features = self.aligner(nactions, action_encoder=True)
        action_features = action_features / action_features.norm(dim=-1, keepdim=True)
        clip_loss = self.clip_loss(condition_features, action_features, logit_scale=1.0)

        kld_loss = -0.5 * torch.sum(1 + logvar - (0.3 * mu) ** 6 - logvar.exp(), dim = 1) # slightly different KL loss function: mu -> 0 [(0.3*mu) ** 6] and var -> 1
        kld_loss_weight = 1e-2 # 0.0005
        loss_mlp = clip_loss + kld_loss * kld_loss_weight

        # condition_features torch.Size([128, 416]) action_features torch.Size([128, 416])

        condition_features = einops.rearrange(condition_features, 'b (h d) -> b h d', h=horizon)


        timesteps, states = sde.generate_random_states(x0=nactions, mu=condition_features)


        sde.set_mu(condition_features)
        sde.set_mu_prim(global_cond)

        data = sde.noise_fn(states, timesteps.squeeze())
        # breakpoint()
        # noise = self.model(self.state, timesteps.squeeze())
        # score = sde.get_score_from_noise(noise, timesteps)

        xt_1_expection = data
        xt_1_optimum = nactions
        matching_loss = F.l1_loss(xt_1_expection, xt_1_optimum)
        loss = matching_loss + loss_mlp
        loss = loss.mean()
        # breakpoint()
        # loss = F.mse_loss(pred, target, reduction='none')
        # loss = loss * loss_mask.type(loss.dtype)
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss.mean()
        # print(f"matching_loss: {matching_loss.mean().item()}", f"loss_mlp: {loss_mlp.mean().item()}")

        loss_dict = {
                'bc_loss': loss.item(),
            }
        
        return loss, loss_dict
