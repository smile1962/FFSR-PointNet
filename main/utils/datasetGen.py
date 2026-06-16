# Standard library imports
import os
import json

# Third-party library imports
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

"""
PyTorch Dataset classes for paired low-resolution / high-resolution flow field super-resolution tasks.

This module provides two HDF5-based Dataset implementations:
1. FlowFieldDataset (active): Full version with geometric features (DEM, slope).
   Applies z-score normalization to hydraulic variables and min-max scaling to terrain features.
2. FlowFieldDataset (commented): Lightweight version without geometric features,
   designed for small-scale scenarios where super-resolution is performed using flow fields only.

Shared features:
- Configurable sample subset, coordinate output, and feature normalization
- Automatic computation and disk caching of normalization statistics
- Multiple metadata loading strategies (JSON dataset, file attributes, per-dataset attributes)
- Optional return of sample metadata and original index for debugging and visualization
"""


# ==============================================================
# Dataset with geometric features (DEM, slope)
# Use this class when terrain information is included in the dataset
# ==============================================================
class FlowFieldDataset(Dataset):
    def __init__(
        self,
        low_h5_path: str,
        high_h5_path: str,
        num_samples: int = None,
        return_coords: bool = False,
        normalize: bool = True,
        mean_std_file: str = 'mean_std.npz',
        x_range: tuple = (-9999999, 9999999999999999999),
        y_range: tuple = (-9999999, 9999999999999999999),
        return_meta: bool = False,
        return_index: bool = False
    ):
        """
        Args:
            low_h5_path: File path to low-resolution flow field HDF5 dataset
            high_h5_path: File path to high-resolution flow field HDF5 dataset
            num_samples: Number of samples to load (first N samples); use all if None
            return_coords: If True, return original X/Y coordinates for visualization
            normalize: If True, apply normalization to hydraulic and terrain features
            mean_std_file: Path to save/load precomputed mean and standard deviation values
            x_range: Coordinate filtering range (min_x, max_x) for X axis
            y_range: Coordinate filtering range (min_y, max_y) for Y axis
            return_meta: If True, return sample metadata dictionary in __getitem__
            return_index: If True, return original sample integer index in __getitem__
            Note: return_meta and return_index can be enabled simultaneously.
        """
        # Store configuration parameters
        self.x_range = x_range
        self.y_range = y_range
        self.return_coords = return_coords
        self.normalize = normalize
        self.mean_std_file = mean_std_file
        self.return_meta = return_meta
        self.return_index = return_index

        # Open HDF5 files (kept open throughout Dataset lifecycle for efficient access)
        self.low_h5 = h5py.File(low_h5_path, 'r')
        self.high_h5 = h5py.File(high_h5_path, 'r')

        # Retrieve sorted sample keys
        self.low_keys = sorted(list(self.low_h5.keys()))
        self.high_keys = sorted(list(self.high_h5.keys()))
        if num_samples is not None:
            self.low_keys = self.low_keys[:num_samples]
            self.high_keys = self.high_keys[:num_samples]
        assert len(self.low_keys) == len(self.high_keys), \
            "Mismatch in number of low-resolution and high-resolution samples"

        # Infer dimensions from first sample
        low_sample = self.low_h5[self.low_keys[0]][:]
        high_sample = self.high_h5[self.high_keys[0]][:]
        self.N_low = low_sample.shape[0]
        self.N_high = high_sample.shape[0]
        self.total_input_dim = low_sample.shape[1]
        self.total_output_dim = high_sample.shape[1]
        # Subtract 2 coordinate channels to get feature-only dimensions
        self.input_dim = self.total_input_dim - 2
        self.output_dim = self.total_output_dim - 2

        # Compute or load normalization statistics
        if self.normalize:
            if os.path.exists(self.mean_std_file):
                data = np.load(self.mean_std_file)
                self.mean_depth = data['mean_depth']
                self.std_depth = data['std_depth']
                self.mean_velocity = data['mean_velocity']
                self.std_velocity = data['std_velocity']
                self.dem_min = data['dem_min']
                self.dem_max = data['dem_max']
                self.slope_min = data['slope_min']
                self.slope_max = data['slope_max']
                self.mean_out = data['mean_out']
                self.std_out = data['std_out']
            else:
                (self.mean_depth, self.std_depth,
                 self.mean_velocity, self.std_velocity,
                 self.dem_min, self.dem_max,
                 self.slope_min, self.slope_max,
                 self.mean_out, self.std_out) = self._compute_mean_std()
                np.savez(
                    self.mean_std_file,
                    mean_depth=self.mean_depth,
                    std_depth=self.std_depth,
                    mean_velocity=self.mean_velocity,
                    std_velocity=self.std_velocity,
                    dem_min=self.dem_min,
                    dem_max=self.dem_max,
                    slope_min=self.slope_min,
                    slope_max=self.slope_max,
                    mean_out=self.mean_out,
                    std_out=self.std_out
                )
        else:
            self.mean_in = np.zeros(4, dtype=np.float32)
            self.std_in = np.ones(4, dtype=np.float32)
            self.mean_out = np.zeros(4, dtype=np.float32)
            self.std_out = np.ones(4, dtype=np.float32)

        # Load metadata list, aligned to the order of low_keys / high_keys
        # Three loading strategies supported (in priority order):
        # 1. Per-dataset attributes (time/event fields) for each sample
        # 2. Dedicated 'meta_json' dataset storing full JSON string
        # 3. 'meta' field in file-level attributes (legacy implementation)
        self.meta_list = self._load_meta_from_h5(self.low_h5, len(self.low_keys))

        # Fallback: try high-res file if low-res metadata length mismatches
        if len(self.meta_list) != len(self.low_keys):
            alt = self._load_meta_from_h5(self.high_h5, len(self.low_keys))
            if len(alt) == len(self.low_keys):
                self.meta_list = alt
            else:
                # Pad with empty dicts to match sample count for compatibility
                print("Warning: metadata length mismatches sample count, padding with empty dicts")
                self.meta_list = (self.meta_list + [{}] * len(self.low_keys))[:len(self.low_keys)]

    def _load_meta_from_h5(self, h5file: h5py.File, expected_len: int) -> list[dict]:
        """
        Attempt to load sample metadata from an HDF5 file.

        Args:
            h5file: Opened HDF5 file object
            expected_len: Expected number of samples (not strictly enforced)

        Returns:
            List of metadata dictionaries, one per sample
        """
        metas = []

        # Strategy 1: dedicated 'meta_json' dataset
        if 'meta_json' in h5file:
            raw = h5file['meta_json'][()]
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8')
            try:
                metas = json.loads(raw)
                return metas
            except Exception:
                pass

        # Strategy 2: file-level 'meta' attribute
        if 'meta' in h5file.attrs:
            raw = h5file.attrs['meta']
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8')
            try:
                metas = json.loads(raw)
                return metas
            except Exception:
                pass

        # Strategy 3: per-dataset attributes (e.g. sample1.attrs['time'])
        ks = sorted(list(h5file.keys()))
        for k in ks:
            attrs = {}
            for a_key, a_val in h5file[k].attrs.items():
                # Decode bytes-typed attributes
                if isinstance(a_val, bytes):
                    try:
                        attrs[a_key] = a_val.decode('utf-8')
                    except Exception:
                        attrs[a_key] = a_val
                else:
                    attrs[a_key] = a_val
            if attrs:
                metas.append(attrs)

        return metas

    def _compute_mean_std(self) -> tuple:
        """
        Compute normalization statistics across the full dataset.
        - Depth and velocity: mean and standard deviation (for z-score normalization)
        - DEM and slope: min and max (for min-max scaling)

        Returns:
            Tuple of statistical values in fixed order
        """
        all_depth = []
        all_velocity = []
        all_dem = []
        all_slope = []
        all_out = []

        for lk, hk in zip(self.low_keys, self.high_keys):
            low_data = self.low_h5[lk][:]
            high_data = self.high_h5[hk][:]

            # Low-res columns: [X, Y, depth, velocity, dem, slope]
            all_depth.append(low_data[:, 2:3])
            all_velocity.append(low_data[:, 3:4])
            all_dem.append(low_data[:, 4:5])
            all_slope.append(low_data[:, 5:6])

            # High-res columns: [X, Y, depth, velocity]
            all_out.append(high_data[:, 2:4])

        all_depth = np.concatenate(all_depth, axis=0)
        all_velocity = np.concatenate(all_velocity, axis=0)
        all_dem = np.concatenate(all_dem, axis=0)
        all_slope = np.concatenate(all_slope, axis=0)
        all_out = np.concatenate(all_out, axis=0)

        # Input feature statistics
        mean_depth = np.mean(all_depth)
        std_depth = np.std(all_depth) + 1e-8
        mean_velocity = np.mean(all_velocity)
        std_velocity = np.std(all_velocity) + 1e-8
        dem_min = np.min(all_dem)
        dem_max = np.max(all_dem)
        slope_min = np.min(all_slope)
        slope_max = np.max(all_slope)

        # Output feature statistics
        mean_out = np.mean(all_out, axis=0)
        std_out = np.std(all_out, axis=0) + 1e-8

        return (
            mean_depth, std_depth,
            mean_velocity, std_velocity,
            dem_min, dem_max,
            slope_min, slope_max,
            mean_out, std_out
        )

    def __len__(self) -> int:
        return len(self.low_keys)

    def __getitem__(self, idx: int) -> tuple:
        low_data = self.low_h5[self.low_keys[idx]][:]
        high_data = self.high_h5[self.high_keys[idx]][:]

        low_tensor = torch.tensor(low_data, dtype=torch.float32)
        high_tensor = torch.tensor(high_data, dtype=torch.float32)

        # Extract feature channels: [depth, velocity, dem, slope]
        low_features = low_tensor[:, 2:6]
        # Extract target channels: [depth, velocity]
        high_features = high_tensor[:, 2:4]

        if self.normalize:
            # Input normalization
            low_features[:, 0] = (low_features[:, 0] - self.mean_depth) / self.std_depth
            low_features[:, 1] = (low_features[:, 1] - self.mean_velocity) / self.std_velocity
            low_features[:, 2] = (low_features[:, 2] - self.dem_min) / (self.dem_max - self.dem_min + 1e-8)
            low_features[:, 3] = (low_features[:, 3] - self.slope_min) / (self.slope_max - self.slope_min + 1e-8)

            # Output normalization (z-score)
            for ch in range(2):
                high_features[:, ch] = (high_features[:, ch] - self.mean_out[ch]) / self.std_out[ch]

        to_return = [low_features, high_features]

        if self.return_coords:
            coords_low = low_tensor[:, :2]
            coords_high = high_tensor[:, :2]
            to_return.extend([coords_low, coords_high])

        if self.return_meta:
            meta = self.meta_list[idx] if idx < len(self.meta_list) else {}
            to_return.append(meta)

        if self.return_index:
            to_return.append(idx)

        return tuple(to_return)

    def __del__(self) -> None:
        try:
            self.low_h5.close()
            self.high_h5.close()
        except Exception:
            pass


# ==============================================================
# Dataset without geometric features (commented out)
# Use for small-scale problems where only LR flow fields are used
# to reconstruct corresponding HR flow fields without terrain data
# ==============================================================
# class FlowFieldDataset(Dataset):
#     def __init__(
#         self,
#         low_h5_path: str,
#         high_h5_path: str,
#         num_samples: int = None,
#         return_coords: bool = False,
#         normalize: bool = True,
#         mean_std_file: str = 'mean_std.npz',
#         x_range: tuple = (-9999999, 9999999999999999999),
#         y_range: tuple = (-9999999, 9999999999999999999),
#         return_meta: bool = False,
#         return_index: bool = False
#     ):
#         """
#         Args:
#             low_h5_path: File path to low-resolution flow field HDF5 dataset
#             high_h5_path: File path to high-resolution flow field HDF5 dataset
#             num_samples: Number of samples to load (first N samples); use all if None
#             return_coords: If True, return original X/Y coordinates for visualization
#             normalize: If True, apply z-score normalization to flow features
#             mean_std_file: Path to save/load precomputed mean and standard deviation values
#             x_range: Coordinate filtering range (min_x, max_x) for X axis
#             y_range: Coordinate filtering range (min_y, max_y) for Y axis
#             return_meta: If True, return sample metadata dictionary in __getitem__
#             return_index: If True, return original sample integer index in __getitem__
#             Note: return_meta and return_index can be enabled simultaneously.
#         """
#         # Store configuration parameters
#         self.x_range = x_range
#         self.y_range = y_range
#         self.return_coords = return_coords
#         self.normalize = normalize
#         self.mean_std_file = mean_std_file
#         self.return_meta = return_meta
#         self.return_index = return_index
#
#         # Open HDF5 files (kept open throughout Dataset lifecycle for efficient access)
#         self.low_h5 = h5py.File(low_h5_path, 'r')
#         self.high_h5 = h5py.File(high_h5_path, 'r')
#
#         # Retrieve sorted sample keys
#         self.low_keys = sorted(list(self.low_h5.keys()))
#         self.high_keys = sorted(list(self.high_h5.keys()))
#         if num_samples is not None:
#             self.low_keys = self.low_keys[:num_samples]
#             self.high_keys = self.high_keys[:num_samples]
#         assert len(self.low_keys) == len(self.high_keys), \
#             "Mismatch in number of low-resolution and high-resolution samples"
#
#         # Infer dimensions from first sample
#         low_sample = self.low_h5[self.low_keys[0]][:]
#         high_sample = self.high_h5[self.high_keys[0]][:]
#         self.N_low = low_sample.shape[0]
#         self.N_high = high_sample.shape[0]
#         self.total_input_dim = low_sample.shape[1]
#         self.total_output_dim = high_sample.shape[1]
#         # Subtract 2 coordinate channels to get feature-only dimensions
#         self.input_dim = self.total_input_dim - 2
#         self.output_dim = self.total_output_dim - 2
#
#         # Compute or load normalization statistics
#         if self.normalize:
#             if os.path.exists(self.mean_std_file):
#                 data = np.load(self.mean_std_file)
#                 self.mean_in = data['mean_in']   # shape: (2,)
#                 self.std_in = data['std_in']     # shape: (2,)
#                 self.mean_out = data['mean_out']
#                 self.std_out = data['std_out']
#             else:
#                 self.mean_in, self.std_in, self.mean_out, self.std_out = self._compute_mean_std()
#                 np.savez(
#                     self.mean_std_file,
#                     mean_in=self.mean_in,
#                     std_in=self.std_in,
#                     mean_out=self.mean_out,
#                     std_out=self.std_out
#                 )
#         else:
#             self.mean_in = np.zeros(2, dtype=np.float32)
#             self.std_in = np.ones(2, dtype=np.float32)
#             self.mean_out = np.zeros(2, dtype=np.float32)
#             self.std_out = np.ones(2, dtype=np.float32)
#
#         # Load metadata list, aligned to the order of low_keys / high_keys
#         # Three loading strategies supported (in priority order):
#         # 1. Per-dataset attributes (time/event fields) for each sample
#         # 2. Dedicated 'meta_json' dataset storing full JSON string
#         # 3. 'meta' field in file-level attributes (legacy implementation)
#         self.meta_list = self._load_meta_from_h5(self.low_h5, len(self.low_keys))
#
#         # Fallback: try high-res file if low-res metadata length mismatches
#         if len(self.meta_list) != len(self.low_keys):
#             alt = self._load_meta_from_h5(self.high_h5, len(self.low_keys))
#             if len(alt) == len(self.low_keys):
#                 self.meta_list = alt
#             else:
#                 # Pad with empty dicts to match sample count for compatibility
#                 print("Warning: metadata length mismatches sample count, padding with empty dicts")
#                 self.meta_list = (self.meta_list + [{}] * len(self.low_keys))[:len(self.low_keys)]
#
#     def _load_meta_from_h5(self, h5file: h5py.File, expected_len: int) -> list[dict]:
#         """
#         Attempt to load sample metadata from an HDF5 file.
#
#         Args:
#             h5file: Opened HDF5 file object
#             expected_len: Expected number of samples (not strictly enforced)
#
#         Returns:
#             List of metadata dictionaries, one per sample
#         """
#         metas = []
#
#         # Strategy 1: dedicated 'meta_json' dataset
#         if 'meta_json' in h5file:
#             raw = h5file['meta_json'][()]
#             if isinstance(raw, bytes):
#                 raw = raw.decode('utf-8')
#             try:
#                 metas = json.loads(raw)
#                 return metas
#             except Exception:
#                 pass
#
#         # Strategy 2: file-level 'meta' attribute
#         if 'meta' in h5file.attrs:
#             raw = h5file.attrs['meta']
#             if isinstance(raw, bytes):
#                 raw = raw.decode('utf-8')
#             try:
#                 metas = json.loads(raw)
#                 return metas
#             except Exception:
#                 pass
#
#         # Strategy 3: per-dataset attributes (e.g. sample1.attrs['time'])
#         ks = sorted(list(h5file.keys()))
#         for k in ks:
#             attrs = {}
#             for a_key, a_val in h5file[k].attrs.items():
#                 # Decode bytes-typed attributes
#                 if isinstance(a_val, bytes):
#                     try:
#                         attrs[a_key] = a_val.decode('utf-8')
#                     except Exception:
#                         attrs[a_key] = a_val
#                 else:
#                     attrs[a_key] = a_val
#             if attrs:
#                 metas.append(attrs)
#
#         return metas
#
#     def _compute_mean_std(self) -> tuple:
#         """
#         Compute z-score normalization statistics for depth and velocity features.
#
#         Returns:
#             Tuple of (mean_in, std_in, mean_out, std_out)
#         """
#         all_in = []
#         all_out = []
#         for lk, hk in zip(self.low_keys, self.high_keys):
#             low_data = self.low_h5[lk][:]
#             high_data = self.high_h5[hk][:]
#
#             all_in.append(low_data[:, 2:4])
#             all_out.append(high_data[:, 2:4])
#
#         all_in = np.concatenate(all_in, axis=0)
#         all_out = np.concatenate(all_out, axis=0)
#         mean_in = np.mean(all_in, axis=0)
#         std_in = np.std(all_in, axis=0) + 1e-8  # Prevent division by zero
#         mean_out = np.mean(all_out, axis=0)
#         std_out = np.std(all_out, axis=0) + 1e-8
#         return mean_in, std_in, mean_out, std_out
#
#     def __len__(self) -> int:
#         return len(self.low_keys)
#
#     def __getitem__(self, idx: int) -> tuple:
#         # Load low-resolution and high-resolution data from HDF5 files
#         low_data = self.low_h5[self.low_keys[idx]][:]
#         high_data = self.high_h5[self.high_keys[idx]][:]
#
#         # Filter rows based on x_range and y_range (disabled by default)
#         # mask_low = (low_data[:, 0] >= self.x_range[0]) & (low_data[:, 0] <= self.x_range[1]) & \
#         #            (low_data[:, 1] >= self.y_range[0]) & (low_data[:, 1] <= self.y_range[1])
#         # low_data_filtered = low_data[mask_low]
#         # mask_high = (high_data[:, 0] >= self.x_range[0]) & (high_data[:, 0] <= self.x_range[1]) & \
#         #             (high_data[:, 1] >= self.y_range[0]) & (high_data[:, 1] <= self.y_range[1])
#         # high_data_filtered = high_data[mask_high]
#
#         # Convert data to torch tensors
#         low_tensor = torch.tensor(low_data, dtype=torch.float32)
#         high_tensor = torch.tensor(high_data, dtype=torch.float32)
#
#         # Extract flow feature channels (depth, velocity)
#         low_features = low_tensor[:, 2:4]
#         high_features = high_tensor[:, 2:4]
#
#         # Apply optional z-score normalization
#         if self.normalize:
#             for c in range(2):
#                 low_features[:, c] = (low_features[:, c] - self.mean_in[c]) / self.std_in[c]
#             for c in range(2):
#                 high_features[:, c] = (high_features[:, c] - self.mean_out[c]) / self.std_out[c]
#
#         # Return coordinates if enabled
#         if self.return_coords:
#             coords_low = low_tensor[:, :2]
#             coords_high = high_tensor[:, :2]
#             return low_features, high_features, coords_low, coords_high
#         else:
#             return low_features, high_features
#
#     def __del__(self) -> None:
#         # Close HDF5 files to release file handles and memory
#         self.low_h5.close()
#         self.high_h5.close()