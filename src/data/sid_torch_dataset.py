# =============================================================================
# Project: RAW ISP | Task: SID Packed RAW Patch Dataset
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase4-Data-v0.1.0
# =============================================================================
"""PyTorch dataset for aligned SID short/long packed RAW patches."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import rawpy
import torch
from torch.utils.data import Dataset

from .sid_manifest import SidPair, load_sid_list
from isp.baseline import pack_bayer


@lru_cache(maxsize=4)
def _load_normalized_packed(path_text: str) -> np.ndarray:
    """Decode one ARW and return read-only H/2 x W/2 x 4 normalized RAW."""
    path = Path(path_text)
    with rawpy.imread(str(path)) as image:
        raw = image.raw_image_visible.astype(np.float32)
        white = float(image.white_level)
        black_rgbg = np.asarray(image.black_level_per_channel[:4], dtype=np.float32)
        pattern = "".join("RGBG"[int(value)] for value in image.raw_pattern.flat)

    packed = pack_bayer(raw, pattern)
    # rawpy metadata order is R,G,B,G; packed order is R,G1,G2,B.
    black = black_rgbg[[0, 1, 3, 2]].reshape(1, 1, 4)
    denominator = white - black
    if np.any(denominator <= 0):
        raise ValueError(f"invalid black/white level metadata: {path}")
    normalized = np.clip((packed - black) / denominator, 0.0, 1.0).astype(np.float32)
    normalized.setflags(write=False)
    return normalized


def sid_scene_id(path: Path) -> str:
    """Return the official five-digit SID scene identifier."""
    return path.stem.split("_")[0]


def _augment_pair(
    input_patch: np.ndarray,
    target_patch: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, bool]]:
    horizontal_flip = bool(rng.integers(0, 2))
    vertical_flip = bool(rng.integers(0, 2))
    transpose = bool(rng.integers(0, 2))
    if horizontal_flip:
        input_patch = input_patch[:, ::-1]
        target_patch = target_patch[:, ::-1]
    if vertical_flip:
        input_patch = input_patch[::-1]
        target_patch = target_patch[::-1]
    if transpose:
        input_patch = input_patch.transpose(1, 0, 2)
        target_patch = target_patch.transpose(1, 0, 2)
    return input_patch, target_patch, {
        "horizontal_flip": horizontal_flip,
        "vertical_flip": vertical_flip,
        "transpose": transpose,
    }


class SidRawPatchDataset(Dataset[dict[str, object]]):
    """Yield aligned exposure-corrected short RAW and long RAW patches.

    Tensor outputs use float32 ``C x H x W`` layout with channels ordered as
    R,G1,G2,B. Both input and target are clipped to [0, 1]. Crop coordinates
    are in packed-RAW pixels; one packed pixel corresponds to a 2x2 Bayer cell.
    """

    def __init__(
        self,
        list_path: str | Path,
        patch_size: int = 256,
        patches_per_pair: int = 1,
        augment: bool = True,
        max_exposure_gain: float | None = 300.0,
        seed: int = 20260920,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if patches_per_pair <= 0:
            raise ValueError("patches_per_pair must be positive")
        if max_exposure_gain is not None and max_exposure_gain <= 0:
            raise ValueError("max_exposure_gain must be positive or None")
        self.pairs = load_sid_list(list_path)
        self.patch_size = patch_size
        self.patches_per_pair = patches_per_pair
        self.augment = augment
        self.max_exposure_gain = max_exposure_gain
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_pair

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic crop/augmentation choices between epochs."""
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = epoch

    def _random_generator(self, index: int) -> np.random.Generator:
        seed = np.random.SeedSequence([self.seed, self.epoch, index])
        return np.random.default_rng(seed)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        pair_position = index // self.patches_per_pair
        pair: SidPair = self.pairs[pair_position]
        short = _load_normalized_packed(str(pair.short_path))
        target = _load_normalized_packed(str(pair.long_path))
        if short.shape != target.shape:
            raise ValueError(
                f"short/long packed shapes differ: {short.shape} vs {target.shape}"
            )
        height, width, channels = short.shape
        if channels != 4 or height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"packed RAW {short.shape} is smaller than patch size {self.patch_size}"
            )

        rng = self._random_generator(index)
        top = int(rng.integers(0, height - self.patch_size + 1))
        left = int(rng.integers(0, width - self.patch_size + 1))
        bottom = top + self.patch_size
        right = left + self.patch_size
        exposure_gain = pair.exposure_gain
        if self.max_exposure_gain is not None:
            exposure_gain = min(exposure_gain, self.max_exposure_gain)
        input_patch = np.clip(short[top:bottom, left:right] * exposure_gain, 0.0, 1.0)
        target_patch = target[top:bottom, left:right]

        augmentation = {
            "horizontal_flip": False,
            "vertical_flip": False,
            "transpose": False,
        }
        if self.augment:
            input_patch, target_patch, augmentation = _augment_pair(
                input_patch, target_patch, rng
            )

        input_tensor = torch.from_numpy(
            np.ascontiguousarray(input_patch.transpose(2, 0, 1), dtype=np.float32)
        )
        target_tensor = torch.from_numpy(
            np.ascontiguousarray(target_patch.transpose(2, 0, 1), dtype=np.float32)
        )
        return {
            "input": input_tensor,
            "target": target_tensor,
            "pair_index": pair_position,
            "scene_id": sid_scene_id(pair.short_path),
            "iso": pair.iso,
            "physical_exposure_gain": float(pair.exposure_gain),
            "applied_exposure_gain": float(exposure_gain),
            "crop_top": top,
            "crop_left": left,
            "augmentation": augmentation,
            "short_path": str(pair.short_path),
            "long_path": str(pair.long_path),
        }
