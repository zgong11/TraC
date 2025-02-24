import os
import random
import tempfile
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from research.envs.base import EmptyEnv

from . import evaluate
from .logger import Logger

LOG_LAST_METRICS = {"step", "steps"}


def log_from_dict(logger: Logger, metric_lists: Dict[str, Union[List, float]], prefix: str) -> None:
    keys_to_remove = []
    for metric_name, metric_value in metric_lists.items():
        if isinstance(metric_value, list) and len(metric_value) > 0:
            v = metric_value[-1] if metric_name in LOG_LAST_METRICS else np.mean(metric_value)
            logger.record(prefix + "/" + metric_name, v)
            keys_to_remove.append(metric_name)
        else:
            logger.record(prefix + "/" + metric_name, metric_value)
            keys_to_remove.append(metric_name)
    for key in keys_to_remove:
        del metric_lists[key]


def log_wrapper(fn: Callable, metric_lists: Dict[str, List]):
    def wrapped_fn(*args, **kwargs):
        metrics = fn(*args, **kwargs)
        for name, value in metrics.items():
            metric_lists[name].append(value)

    return wrapped_fn


def time_wrapper(fn: Callable, name: str, profile_lists: Dict[str, List]):
    def wrapped_fn(*args, timeit=False, **kwargs):
        if timeit:
            start_time = time.time()
            output = fn(*args, **kwargs)
            end_time = time.time()
            profile_lists[name].append(end_time - start_time)
        else:
            output = fn(*args, **kwargs)
        return output

    return wrapped_fn


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.utils.data.get_worker_info().seed
    seed = seed % (2**32 - 1)  # Reduce to valid 32bit unsigned range
    np.random.seed(seed)
    random.seed(seed)


class Trainer(object):
    def __init__(
        self,
        model,
        env_fn: Optional[Callable] = None,
        eval_env_fn: Optional[Callable] = None,
        env_runner: Optional[str] = None,
        eval_env_runner: Optional[str] = None,
        total_steps: int = 1000,
        log_freq: int = 100,
        env_freq: int = 1,
        eval_freq: int = 1000,
        profile_freq: int = -1,
        checkpoint_freq: Optional[int] = None,
        target_cost: float = 20.0,
        eval_fn: Optional[Any] = None,
        eval_kwargs: Optional[Dict] = None,
        train_dataloader_kwargs: Optional[Dict] = None,
    ) -> None:
        self.model = model

        # Environment parameters.
        self._env = None
        self.env_fn = env_fn
        self.env_runner = env_runner
        self._eval_env = None
        self.eval_env_fn = eval_env_fn
        self.eval_env_runner = eval_env_runner

        # Logging parameters
        self.total_steps = total_steps
        self.log_freq = log_freq
        self.env_freq = env_freq
        self.eval_freq = eval_freq
        self.profile_freq = profile_freq
        self.checkpoint_freq = checkpoint_freq
        self.target_cost = target_cost

        # Eval parameters
        self.eval_fn = eval_fn
        self.eval_kwargs = {} if eval_kwargs is None else eval_kwargs

        # Dataloader parameters
        self._train_dataloader = None
        self.train_dataloader_kwargs = {} if train_dataloader_kwargs is None else train_dataloader_kwargs

    @property
    def env(self):
        if self._env is None and self.env_fn is not None:
            env_runner = self.env_runner
            if env_runner is None:
                self._env = self.env_fn()
            else:
                self._env = env_runner(
                    self.env_fn, observation_space=self.model.observation_space, action_space=self.model.action_space
                )
        return self._env

    @property
    def eval_env(self):
        if self._eval_env is None and self.eval_env_fn is not None:
            env_runner = self.eval_env_runner
            if env_runner is None:
                self._eval_env = self.eval_env_fn()
            else:
                self._eval_env = env_runner(
                    self.eval_env_fn,
                    observation_space=self.model.observation_space,
                    action_space=self.model.action_space,
                )
        return self._eval_env

    @property
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if not hasattr(self.model, "dataset"):
            raise ValueError("Must call model.setup_datasets before get dataloader!")
        if self.model.dataset is None:
            return None
        if self._train_dataloader is None:
            shuffle = not isinstance(self.model.dataset, torch.utils.data.IterableDataset)
            pin_memory = self.model.device.type == "cuda"
            self._train_dataloader = torch.utils.data.DataLoader(
                self.model.dataset,
                shuffle=shuffle,
                pin_memory=pin_memory,
                worker_init_fn=_worker_init_fn,
                **self.train_dataloader_kwargs,
            )
        return self._train_dataloader

    def train(self, path: str):
        # Prepare the model for training by initializing the optimizers and the schedulers
        self.model.setup_optimizers()
        self.model.setup_schedulers(do_nothing=True)
        print("[research] Training a model with", self.model.num_params, "trainable parameters.")
        print("[research] Estimated size: {:.2f} GB".format(self.model.nbytes / 1024**3))

        step, epoch = 0, 0
        # Setup datasets
        self.model.setup_datasets(self.env, self.target_cost, self.total_steps)
        self.model.setup_gammalst() # get gammas
        print(f"Desired vs Undesired: {np.sum(self.model.dataset.data['label'] == 1.0)} vs {np.sum(self.model.dataset.data['label'] == 0.0)}")

        # Setup logging
        writers = ["tb", "csv"]
        try:
            # Detect if wandb has been setup. If so, log it.
            import wandb

            if wandb.run is not None:
                writers.append("wandb")
        except ImportError:
            pass

        logger = Logger(path=path, writers=writers)

        # Construct all of the metric lists to be used during training
        # Construct all the metric lists to be used during training
        train_metric_lists = defaultdict(list)
        env_metric_lists = defaultdict(list)
        profiling_metric_lists = defaultdict(list)
        # Wrap the functions we use in logging and profile wrappers
        train_step = log_wrapper(self.model.train_step, train_metric_lists)
        train_step = time_wrapper(train_step, "train_step", profiling_metric_lists)
        env_step = log_wrapper(self.model.env_step, env_metric_lists)
        env_step = time_wrapper(env_step, "env_step", profiling_metric_lists)
        format_batch = time_wrapper(self.model.format_batch, "processor", profiling_metric_lists)

        best_norm_reward = -1 * float("inf")
        best_norm_cost = float("inf")

        # Setup Loop values
        env_freq = int(self.env_freq) if self.env_freq >= 1 else 1
        if self.env is None or isinstance(self.env, EmptyEnv):
            env_freq = 1000000  # choose a really large value arbitrarily.
        env_iters = int(1 / self.env_freq) if self.env_freq < 1 else 1

        # Set model to train
        self.model.train()

        profile = True if self.profile_freq > 0 else False  # must profile to get all keys for csv log
        start_time = time.time()
        current_time = start_time

        while step <= self.total_steps:
            for batch in self.train_dataloader:
                if profile:
                    profiling_metric_lists["dataset"].append(time.time() - current_time)

                # Run the environment step.
                if step % env_freq == 0:
                    for _ in range(env_iters):
                        env_step(self.env, step, self.total_steps, timeit=profile)

                # Next, format the batch
                batch = format_batch(batch, timeit=profile)

                # Run the train step
                train_step(batch, step, timeit=profile)

                # Update the schedulers
                for scheduler in self.model.schedulers.values():
                    scheduler.step()

                # Now determine if we should dump the logs
                if step % self.log_freq == 0:
                    # Record timing metrics
                    current_time = time.time()
                    logger.record("time/step", step)
                    logger.record("time/epoch", epoch)
                    logger.record("time/steps_per_second", self.log_freq / (current_time - start_time))
                    log_from_dict(logger, profiling_metric_lists, "time")
                    start_time = current_time
                    # Record learning rates
                    for name, scheduler in self.model.schedulers.items():
                        logger.record("lr/" + name, scheduler.get_last_lr()[0])
                    # Record training metrics
                    log_from_dict(logger, env_metric_lists, "env")
                    log_from_dict(logger, train_metric_lists, "train")
                    logger.dump(step=step)
                    # Update the last time we logged.

                # Run eval and validation, but skip if benchmark is on.
                if step % self.eval_freq == 0 and not step == 0:
                    self.model.eval()
                    model_metadata = dict(step=step, epoch=epoch)

                    # Run and time eval step
                    current_time = time.time()
                    eval_metrics = self.evaluate(path, step)
                    logger.record("time/eval", time.time() - current_time)
                    current_norm_reward = eval_metrics["normalized_reward"]
                    current_norm_cost = eval_metrics["normalized_cost"]
                    log_from_dict(logger, eval_metrics, "eval")

                    if (best_norm_cost >= 1 and current_norm_cost < best_norm_cost) or (
                        best_norm_cost < 1 and current_norm_cost < 1 and current_norm_reward > best_norm_reward
                    ):
                        best_norm_cost = current_norm_cost
                        best_norm_reward = current_norm_reward
                        self.model.save(path, "best_model", model_metadata)

                    # Eval Logger dump to CSV
                    logger.dump(step=step, eval=True)  # Mark True on the eval flag
                    # Save the final model
                    self.model.save(path, "final_model", model_metadata)  # Also save the final model every eval period.
                    # Put the model back in train mode.
                    self.model.train()

                if self.checkpoint_freq is not None and step % self.checkpoint_freq == 0:
                    # Save a checkpoint
                    model_metadata = dict(step=step, epoch=epoch)
                    self.model.save(path, "model_" + str(step), model_metadata)

                step += 1
                if step > self.total_steps:
                    break  # We need to break in the middle of an epoch.

                profile = self.profile_freq > 0 and step % self.profile_freq == 0
                if profile:
                    current_time = time.time()  # update current time only, not start time

            epoch += 1

        # Cleanup!
        model_metadata = dict(step=step, epoch=epoch)
        self.model.save(path, "final_model", model_metadata)
        if self._env is not None:
            self._env.close()
        if self._eval_env is not None:
            self._eval_env.close()


    def evaluate(self, path: str, step: int):
        assert not self.model.training
        eval_fn = None if self.eval_fn is None else vars(evaluate)[self.eval_fn]
        if eval_fn is None:
            return dict()
        eval_metrics = eval_fn(self.eval_env, self.model, self.target_cost, path, step, **self.eval_kwargs)
        return eval_metrics
