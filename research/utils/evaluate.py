import collections
import os
from typing import Any, Dict, List, Optional

import gymnasium as gym
import dsrl
import imageio
import numpy as np
import torch


class EvalMetricTracker(object):

    def __init__(self):
        self.metrics = collections.defaultdict(list)
        self.ep_length = 0
        self.ep_reward = 0
        self.ep_cost = 0

    def reset(self) -> None:
        if self.ep_length > 0:
            # Add the episode to overall metrics
            self.metrics["cost"].append(self.ep_cost)
            self.metrics["reward"].append(self.ep_reward)
            self.metrics["length"].append(self.ep_length)

            self.ep_length = 0
            self.ep_reward = 0
            self.ep_cost = 0

    def step(self, reward: float, info: Dict) -> None:
        self.ep_length += 1
        self.ep_reward += reward
        self.ep_cost += info["cost"]

    def add(self, k: str, v: Any):
        self.metrics[k].append(v)

    def export(self) -> Dict:
        if self.ep_length > 0:
            # We have one remaining episode to log, make sure to get it.
            self.reset()
        metrics = {k: np.mean(v) for k, v in self.metrics.items()}
        metrics["reward_std"] = np.std(self.metrics["reward"])
        metrics["cost_std"] = np.std(self.metrics["cost"])
        metrics["normalized_reward_std"] = np.std(self.metrics["normalized_reward"])
        metrics["normalized_cost_std"] = np.std(self.metrics["normalized_cost"])
        return metrics

def eval_policy(
    env: gym.Env,
    model,
    target_cost: float,
    path: str,
    step: int,
    num_ep: int = 10,
    num_gifs: int = 0,
    width=200,
    height=200,
    every_n_frames: int = 2,
    predict_kwargs: Optional[Dict] = None,
) -> Dict:
    metric_tracker = EvalMetricTracker()
    predict_kwargs = {} if predict_kwargs is None else predict_kwargs
    assert num_gifs <= num_ep, "Cannot save more gifs than eval ep."

    env.set_target_cost(target_cost)

    for i in range(num_ep):
        # Reset Metrics
        done = False
        ep_length, ep_reward, ep_cost = 0, 0, 0
        frames = []
        save_gif = i < num_gifs
        render_kwargs = dict(mode="rgb_array", width=width, height=height) if save_gif else dict()
        obs, info = env.reset()

        # Metadrive reset gives tuple for obs.
        if "Metadrive" in env.unwrapped.spec.id:
            obs = obs[0]

        if save_gif:
            frames.append(env.render(**render_kwargs))
        metric_tracker.reset()

        while not done:
            batch = dict(obs=obs)
            with torch.no_grad():
                action = model.predict(batch, **predict_kwargs)
            next_obs, reward, terminal, timeout, info = env.step(action)
            done = terminal or timeout
            ep_reward += reward
            ep_cost += info["cost"]
            metric_tracker.step(reward, info)
            ep_length += 1
            if save_gif and ep_length % every_n_frames == 0:
                frames.append(env.render(**render_kwargs))
            obs = next_obs

        normalized_reward, normalized_cost = env.get_normalized_score(ep_reward, ep_cost)
        metric_tracker.add("normalized_reward", normalized_reward)
        metric_tracker.add("normalized_cost", normalized_cost)

        if save_gif:
            gif_name = "vis-{}_ep-{}.gif".format(step, i)
            imageio.mimsave(os.path.join(path, gif_name), frames)

    return metric_tracker.export()
