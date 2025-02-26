# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algos.info_nce import InfoNCE
import utils
from utils import attribution_augmentation, random_overlay, random_color_jitter

class RewardPredictor(nn.Module):
    def __init__(self, encoder, action_shape, accumulate_steps, hidden_dim):
        super().__init__()


        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.repr_dim + action_shape[0] * accumulate_steps, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(utils.weight_init)

    def forward(self, x, action):
        x = self.encoder(x)
        x = torch.cat([x, action], dim=1)
        x = self.mlp(x)
        return x



class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, inchannel):
        super().__init__()
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(inchannel, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h
    
class Masker(nn.Module):
    def __init__(self, inchannel):
        super().__init__()

        self.convnet = nn.Sequential(nn.Conv2d(inchannel//3, 32, 3, stride=1, padding=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1, padding=1),
                                    nn.ReLU(), nn.Conv2d(32, 1, 3, stride=1, padding=1),
                                    nn.Sigmoid()
                                    #  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                    #  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                    #  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                    )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        return h
    
class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class MadiAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.att = None
        self.num_masks = 3
        # models
        self.encoder = Encoder(obs_shape[0]-3).to(device)

        self.mask_predictor = Masker(obs_shape[0]-3).to(device)

        self.reward_predictor = RewardPredictor(self.encoder, action_shape, 1,
                                                    1024).to(device)
        self.reward_predictor_optimizer = torch.optim.Adam(
            self.reward_predictor.parameters(), lr=lr
        )

        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.mask_predictor_opt = torch.optim.Adam(self.mask_predictor.parameters(), lr=1e-3)
        self.disentanglement_opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-6)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.disentanglement_opt, milestones=[4e4,8e4,12e4,16e4,20e4], gamma=0.5)
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.infonce = InfoNCE()
        print('Setting up the Madi Model')
        self.train()
        self.critic_target.train()

    def apply_mask(self, obs):
        # obs: tensor shaped as (B, 9, H, W)
        frames = obs.chunk(self.num_masks, dim=1)  # frames: list of tensors [ (B,3,H,W) , (B,3,H,W) , (B,3,H,W) ]
        frames_cat = torch.cat(frames, dim=0)  # concat in batch dim. frames_cat: tensor shaped (B*3, 3, H, W)
        masks_cat = self.mask_predictor(frames_cat)  # apply MaskerNet just once. masks_cat: (B*3, 1, H, W)
        self.att = masks_cat.clone()[-1,0,:,:]
        masks = masks_cat.chunk(self.num_masks, dim=0)  # split the batch dim back into channel dim. masks: list of tensors [ (B,1,H,W) , (B,1,H,W) , (B,1,H,W) ]
        masked_frames = [m * f for m, f in zip(masks, frames)]  # element-wise multiplication, uses broadcasting over the 3 RGB channels within 1 frame. masked_frames: list of tensors [ (B,3,H,W) , (B,3,H,W) , (B,3,H,W) ]
        return torch.cat(masked_frames, dim=1)  # concat in channel dim. returns: tensor shaped (B, 9, H, W)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.mask_predictor.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        
        obs = torch.as_tensor(obs, device=self.device)
        pixel_obs = obs[:9,:,:]
        pixel_obs = self.apply_mask(pixel_obs.unsqueeze(0))
        obs = self.encoder(pixel_obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step, aug_obs):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        aug_Q1, aug_Q2 = self.critic(aug_obs, action)
        aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)
        
        critic_loss = 0.5 * (critic_loss + aug_loss)
        metrics['train_critic_mask_loss'] = aug_loss
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        self.mask_predictor_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.mask_predictor.parameters(), max_norm=20.0)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=20.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=20.0)
        self.critic_opt.step()
        self.encoder_opt.step()
        self.mask_predictor_opt.step()

        return metrics
    

    def threshold_tensor_by_quantile(self, tensor: torch.Tensor) -> torch.Tensor:
        # Calculate the 75th quantile of the tensor
        quantile = torch.quantile(tensor.flatten(1), 0.90,1).to(self.device)

        # Create a new tensor where values greater than the 75th quantile are set to 1, others to 0
        output_tensor = torch.where(tensor >= quantile[:,None,None,None], torch.tensor(1.0).to(self.device), torch.tensor(0.0).to(self.device))

        return output_tensor
    
    def update_reward_predictor(self, obs, action, reward, step):
        metrics = dict()
        predicted_reward = self.reward_predictor(obs, action)
        predict_loss = F.mse_loss(reward, predicted_reward)

        self.reward_predictor_optimizer.zero_grad()
        predict_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_predictor.parameters(), 20)
        self.reward_predictor_optimizer.step()
        metrics['train_rewardpred_loss'] = predict_loss
        
        return metrics


    
    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        # encode
        pixel_obs = obs[:,:9]
        overlay = random_overlay(random_color_jitter(pixel_obs.clone()))
        overlay = self.apply_mask(overlay)
        pixel_obs = self.apply_mask(pixel_obs)
        obs = self.encoder(pixel_obs)
        overlayed_feature = self.encoder(overlay)
        with torch.no_grad():
            next_pixel_obs = next_obs[:,:9]
            next_pixel_obs = self.apply_mask(next_pixel_obs)
            # next_aug_obs = next_pixel_obs.clone()
            next_obs = self.encoder(next_pixel_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step, overlayed_feature))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
            

        return metrics
