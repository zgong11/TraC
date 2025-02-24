import datetime
import functools
import os
import sys
import tempfile
from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from research.envs.base import EmptyEnv
from research.networks.base import ModuleContainer
from research.utils import utils

from .base import Algorithm


class OffPolicyAlgorithm(Algorithm):
    def __init__(
        self,
        *args,
        offline_steps: int = 0,  # Run fully offline by setting to -1
        random_steps: int = 1000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.offline_steps = offline_steps
        self.random_steps = random_steps

    def setup_datasets(self, env: gym.Env, target_cost: float, total_steps: int):
        super().setup_datasets(env, target_cost, total_steps)
        # Assign the correct update function based on what is passed in.
        if env is None or isinstance(env, EmptyEnv) or self.offline_steps < 0:
            self.env_step = self._empty_step
        elif isinstance(env, gym.Env):
            # Setup Env Metrics
            self._current_obs = env.reset()
            self._episode_reward = 0
            self._episode_length = 0
            self._num_ep = 0
            self._env_steps = 0
            # Note that currently the very first (s, a) pair is thrown away because
            # we don't add to the dataset here.
            # This was done for better compatibility for offline to online learning.
            self.dataset.add(obs=self._current_obs)  # add the first observation.
            self.env_step = self._env_step
        else:
            raise ValueError("Invalid env passed")

    def _empty_step(self, env: gym.Env, step: int, total_steps: int) -> Dict:
        return dict()

    def _env_step(self, env: gym.Env, step: int, total_steps: int) -> Dict:
        # Return if env is Empty or we we aren't at every env_freq steps
        if step <= self.offline_steps:
            # Purposefully set to nan so we write CSV log.
            return dict(steps=self._env_steps, reward=-np.inf, length=np.inf, num_ep=self._num_ep)

        if step < self.random_steps:
            action = env.action_space.sample()
        else:
            self.eval()
            action = self._get_train_action(self._current_obs, step, total_steps)
            self.train()
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, reward, done, info = env.step(action)
        self._env_steps += 1
        self._episode_length += 1
        self._episode_reward += reward

        if "discount" in info:
            discount = info["discount"]
        elif hasattr(env, "_max_episode_steps") and self._episode_length == env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        # Store the consequences.
        self.dataset.add(obs=next_obs, action=action, reward=reward, done=done, discount=discount)

        if done:
            self._num_ep += 1
            # Compute metrics
            metrics = dict(
                steps=self._env_steps, reward=self._episode_reward, length=self._episode_length, num_ep=self._num_ep
            )
            # Reset the environment
            self._current_obs = env.reset()
            self.dataset.add(obs=self._current_obs)  # Add the first timestep
            self._episode_length = 0
            self._episode_reward = 0
            return metrics
        else:
            self._current_obs = next_obs
            return dict(steps=self._env_steps)

    @abstractmethod
    def _get_train_action(self, obs: Any, step: int, total_steps: int) -> np.ndarray:
        raise NotImplementedError

    @functools.cached_property
    def action_range(self):
        action_range = (self.processor.action_space.low, self.processor.action_space.high)
        return utils.to_device(utils.to_tensor(action_range), self.device)

    def _predict(
        self, batch: Dict, sample: bool = False, noise: float = 0.0, noise_clip: Optional[float] = None, temperature=1.0
    ) -> torch.Tensor:
        with torch.no_grad():
            if isinstance(self.network, ModuleContainer) and "encoder" in self.network.CONTAINERS:
                obs = self.network.encoder(batch["obs"])
            else:
                obs = batch["obs"]

            # Could be: Logits (discrete), Float (continuous), or torch Dist
            dist = self.network.actor(obs)

            if isinstance(self.processor.action_space, gym.spaces.Box):
                if isinstance(dist, torch.distributions.Independent):
                    # Guassian Distribution
                    action = dist.sample() if sample else dist.base_dist.loc

                elif isinstance(dist, torch.distributions.MixtureSameFamily):
                    # Mixture of Gaussians.
                    if sample:
                        action = dist.sample()
                    else:
                        # Robomimic always samples from the Categorical, but then does the mixture deterministically.
                        loc = dist.component_distribution.base_dist.loc
                        category = dist.mixture_distribution.sample()

                        # Expand to add Mixture Dim, Action Dim
                        es = dist.component_distribution.event_shape
                        mix_sample_r = category.reshape(category.shape + torch.Size([1] * (len(es) + 1)))
                        mix_sample_r = mix_sample_r.repeat(torch.Size([1] * len(category.shape)) + torch.Size([1]) + es)
                        action = torch.gather(loc, len(dist.batch_shape), mix_sample_r)
                        action = action.squeeze(len(dist.batch_shape))

                elif torch.is_tensor(dist):
                    action = dist

                else:
                    raise ValueError("Model output incompatible with default _predict.")

                if noise > 0.0:
                    eps = noise * torch.randn_like(action)
                    if noise_clip is not None:
                        eps = torch.clamp(eps, -noise_clip, noise_clip)
                    action = action + eps
                action = action.clamp(*self.action_range)
                return action

            elif isinstance(self.processor.action_space, gym.spaces.Discrete):
                logits = dist.logits if isinstance(dist, torch.distributions.Categorical) else dist
                if sample:
                    action = torch.distributions.Categorical(logits=logits / temperature).sample()
                else:
                    action = logits.argmax(dim=-1)

                return action

            else:
                raise ValueError("Complex action_space incompatible with default _predict.")