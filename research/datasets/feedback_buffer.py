import math
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from research.utils import utils

import h5py

class FeedbackBuffer(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        target_cost: float,
        path: Optional[str] = None,
        discount: float = 0.99,
        action_eps: float = 1e-5,
        segment_length: Optional[int] = None,
        seg_ratio: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
        kappa: float = 0.1,
        safe_top_perc: float = 1.0,
        safe_bottom_perc: float = 0.0,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
    ):
        self.discount = discount
        self.batch_size = 1 if batch_size is None else batch_size
        self.target_cost = target_cost
        self.segment_length = segment_length
        self.seg_ratio = seg_ratio
        self.kappa = kappa
        self.safe_top_perc = safe_top_perc
        self.safe_bottom_perc = safe_bottom_perc

        assert path is not None, "Must provide dataset file."
        data = self.load_segdsrl_data(path)
        dataset_size = data["action"].shape[0]  # The number of segments in the dataset
        assert capacity is None or capacity <= dataset_size, "Capacity is set larger than dataset!"
        if capacity is not None and dataset_size > capacity:
            # Trim the dataset down
            data = utils.get_from_batch(data, 0, capacity)

        # preprocess the data
        data = utils.remove_float64(data)
        lim = 1 - action_eps
        data["action"] = np.clip(data["action"], a_min=-lim, a_max=lim)

        # Save the data
        self.data = data

    def __len__(self):
        return self.data["label"].shape[0]
    
    def __iter__(self):
        size = len(self)
        my_inds = np.arange(size)
        idxs = np.random.permutation(my_inds)
        for i in range(math.ceil(len(idxs) / self.batch_size)):  # Need to use ceil to get all data points.
            if self.batch_size == 1:
                data_idxs = idxs[i]
            else:
                # Might be some overlap here but its probably OK.
                data_idxs = idxs[i * self.batch_size : min((i + 1) * self.batch_size, len(self))]

            batch = {
                "obs": self.data["obs"][data_idxs],
                "action": self.data["action"][data_idxs],
                "weight": self.data["weight"][data_idxs],
                "label": self.data["label"][data_idxs],
            }

            yield batch


    def load_segdsrl_data(self, datapath):
        with h5py.File(datapath, 'r') as h5pydata:
            observations = np.array(h5pydata['observations'])
            actions = np.array(h5pydata['actions'])
            rewards = np.array(h5pydata['rewards'])
            costs = np.array(h5pydata['costs'])
            terminals = np.array(h5pydata['terminals'])
            timeouts = np.array(h5pydata['timeouts'])

            # Get index of all the rollouts
            terminals_idx = np.where(terminals==True)[0]
            timeouts_idx = np.where(timeouts==True)[0]
            rollout_endidx_list = np.sort(np.concatenate((terminals_idx, timeouts_idx))) + 1
            rollout_startidx_list = np.insert(rollout_endidx_list[:-1], 0, 0)
            self.max_rollout_len = max(rollout_endidx_list - rollout_startidx_list)
            rollout_num = rollout_startidx_list.shape[0]
            obs_dim = observations.shape[1]
            act_dim = actions.shape[1]

            self.segment_length = int(self.max_rollout_len * self.seg_ratio)   # set segment_length to be min(64, max_rollout_len // 5)
            segment_threshold = self.target_cost * self.segment_length / self.max_rollout_len
            segments_num = ((self.max_rollout_len - self.segment_length) // 6) * rollout_num if self.seg_ratio < 1.0 \
                else np.count_nonzero(rollout_endidx_list - rollout_startidx_list == self.max_rollout_len) 
            segments_num = min(segments_num, 1000000)

            segment_obs = np.empty((segments_num, self.segment_length, obs_dim))
            segment_act = np.empty((segments_num, self.segment_length, act_dim))
            seg_rewsum = np.empty(segments_num)
            seg_costsum = np.empty(segments_num)
            seg_safety = np.empty(segments_num)
            segment_weights = np.empty(segments_num)
            segment_labels = np.empty(segments_num)

            unsatisfied_rollout_startidx_list = np.where((rollout_startidx_list < rollout_endidx_list - self.segment_length + 1) == False)[0]

            # Uniformly select segments
            rollout_idxes = np.random.choice(np.delete(np.arange(rollout_startidx_list.size), unsatisfied_rollout_startidx_list), segments_num, replace=True)
            seg_startidx = np.random.randint(rollout_startidx_list[rollout_idxes], rollout_endidx_list[rollout_idxes] - self.segment_length + 1)

            for i in range(segments_num):
                segment_obs[i] = observations[seg_startidx[i]:seg_startidx[i] + self.segment_length]
                segment_act[i] = actions[seg_startidx[i]:seg_startidx[i] + self.segment_length]

                # Get comparison labels
                seg_rewsum[i] = rewards[seg_startidx[i]:seg_startidx[i] + self.segment_length].sum()
                seg_costsum[i] = costs[seg_startidx[i]:seg_startidx[i] + self.segment_length].sum()

                seg_safety[i] = 0 if seg_costsum[i] > segment_threshold else 1

            # Get index of safe and unsafe segments
            safe_seg_idx = np.where(seg_safety == 1)[0]
            unsafe_seg_idx = np.where(seg_safety == 0)[0]

            # For safe segments, compute the weights
            safe_seg_rewsum = seg_rewsum[safe_seg_idx]
            safe_seg_rewsum_max, safe_seg_rewsum_min = safe_seg_rewsum.max(), safe_seg_rewsum.min()

            w_safe_seg = (1 - self.kappa) * (safe_seg_rewsum - safe_seg_rewsum_min) / (safe_seg_rewsum_max - safe_seg_rewsum_min) + self.kappa
            w_safe_seg_reverse = (1 - self.kappa) * (1 - (safe_seg_rewsum - safe_seg_rewsum_min) / (safe_seg_rewsum_max - safe_seg_rewsum_min)) + self.kappa
            w_safe_seg_sortidxidx = np.argsort(w_safe_seg)

            top_perc = self.safe_top_perc   # test with only top safe trajs
            least_perc = self.safe_bottom_perc
            # Get idx of idx of desired trajetories
            desired_seg_idxidx = w_safe_seg_sortidxidx[-math.ceil(w_safe_seg_sortidxidx.shape[0] * top_perc):]
            # Get idx of idx of undesired trajectories from safe trajctories
            undesired_safe_seg_idxidx = w_safe_seg_sortidxidx[:math.ceil(w_safe_seg_sortidxidx.shape[0] * least_perc)]

            # Get weights and lables for the data
            segment_weights[safe_seg_idx[desired_seg_idxidx]] = w_safe_seg[desired_seg_idxidx]
            segment_labels[safe_seg_idx[desired_seg_idxidx]] = 1
            segment_weights[safe_seg_idx[undesired_safe_seg_idxidx]] = w_safe_seg_reverse[undesired_safe_seg_idxidx]
            segment_labels[safe_seg_idx[undesired_safe_seg_idxidx]] = 0

            segment_weights[unsafe_seg_idx] = 1
            segment_labels[unsafe_seg_idx] = 0

            # Remove unlabled data
            labeled_idx = np.concatenate((unsafe_seg_idx, safe_seg_idx[desired_seg_idxidx], safe_seg_idx[undesired_safe_seg_idxidx]))
            abandon_idx = np.delete(np.arange(segments_num), labeled_idx)

            if abandon_idx.shape[0] > 0:
                segment_obs = np.delete(segment_obs, abandon_idx, axis=0)
                segment_act = np.delete(segment_act, abandon_idx, axis=0)
                segment_weights = np.delete(segment_weights, abandon_idx)
                segment_labels = np.delete(segment_labels, abandon_idx)

            data = {
                "obs": segment_obs,
                "action": segment_act,
                "weight": segment_weights,
                "label": segment_labels,
            }
        return data
