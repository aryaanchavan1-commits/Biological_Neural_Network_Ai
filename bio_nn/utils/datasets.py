"""Dataset loading, splitting, and continual-learning task creation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


def load_dataset(
    name: str,
    root: str = "./data",
    **kwargs: Any,
) -> Tuple[Dataset, Dataset]:
    """Load a dataset by name.

    Supports ``"mnist"``, ``"fashion_mnist"``, and ``"cifar10"``.
    Data is auto-downloaded to *root* if not already present.

    Args:
        name: Dataset identifier (case-insensitive).
        root: Local directory for cached data.

    Returns:
        ``(train_dataset, test_dataset)`` tuple.
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    if name_lower == "mnist":
        return _load_torchvision("MNIST", root, **kwargs)
    elif name_lower in ("fashion_mnist", "fashionmnist"):
        return _load_torchvision("FashionMNIST", root, **kwargs)
    elif name_lower in ("cifar10", "cifar_10"):
        return _load_torchvision("CIFAR10", root, **kwargs)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: mnist, fashion_mnist, cifar10"
        )


def _load_torchvision(
    ds_name: str, root: str, **kwargs: Any
) -> Tuple[Dataset, Dataset]:
    """Helper to load a torchvision dataset with transform defaults."""
    from torchvision import datasets as tv_datasets, transforms

    normalize = kwargs.pop("normalize", True)
    flatten = kwargs.pop("flatten", True)

    if ds_name == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        to_tensor = transforms.ToTensor()
        transform = (
            transforms.Compose([to_tensor, transforms.Normalize(mean, std)])
            if normalize
            else to_tensor
        )
        train_ds = getattr(tv_datasets, ds_name)(
            root=root, train=True, download=True, transform=transform
        )
        test_ds = getattr(tv_datasets, ds_name)(
            root=root, train=False, download=True, transform=transform
        )
        if flatten:
            train_ds = _FlattenDataset(train_ds)
            test_ds = _FlattenDataset(test_ds)
    else:
        to_tensor = transforms.ToTensor()
        transform = (
            transforms.Compose([to_tensor, transforms.Normalize((0.1307,), (0.3081,))])
            if normalize
            else to_tensor
        )
        train_ds = getattr(tv_datasets, ds_name)(
            root=root, train=True, download=True, transform=transform
        )
        test_ds = getattr(tv_datasets, ds_name)(
            root=root, train=False, download=True, transform=transform
        )
        if flatten:
            train_ds = _FlattenDataset(train_ds)
            test_ds = _FlattenDataset(test_ds)

    return train_ds, test_ds


class _FlattenDataset(Dataset):
    """Wraps an image dataset to flatten spatial dimensions."""

    def __init__(self, ds: Dataset) -> None:
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.ds[idx]
        return img.flatten(), label


# ------------------------------------------------------------------
# Continual learning splits
# ------------------------------------------------------------------


def create_split_dataset(
    dataset: Dataset,
    n_tasks: int,
    scenario: str = "class_incremental",
    **kwargs: Any,
) -> List[Tuple[Dataset, Dataset]]:
    """Split a dataset into *n_tasks* for continual learning.

    Args:
        dataset:   Source dataset (must have a ``targets`` attribute or
                   be a ``TensorDataset``).
        n_tasks:   Number of tasks to split into.
        scenario:  Split strategy.  Currently supports:
                   - ``"class_incremental"``: disjoint class sets per task.

    Returns:
        List of ``(train_subset, test_subset)`` per task.
    """
    if scenario == "class_incremental":
        return _class_incremental_split(dataset, n_tasks)
    raise ValueError(f"Unknown scenario '{scenario}'. Supported: class_incremental")


def _class_incremental_split(
    dataset: Dataset, n_tasks: int
) -> List[Tuple[Subset, Subset]]:
    """Split by disjoint class ranges."""
    targets = _get_targets(dataset)
    classes = sorted(set(targets))
    n_classes = len(classes)
    per_task = math.ceil(n_classes / n_tasks)

    splits: List[Tuple[Subset, Subset]] = []
    for t in range(n_tasks):
        start = t * per_task
        end = min(start + per_task, n_classes)
        task_classes = set(classes[start:end])

        indices = [i for i, y in enumerate(targets) if y in task_classes]
        split_point = int(len(indices) * 0.8)
        train_idx = indices[:split_point]
        test_idx = indices[split_point:]

        splits.append((Subset(dataset, train_idx), Subset(dataset, test_idx)))

    return splits


def _get_targets(dataset: Dataset) -> List[int]:
    """Extract target labels from a dataset."""
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if isinstance(dataset, TensorDataset):
        return dataset[1].tolist()  # type: ignore[union-attr]
    # Fallback: iterate
    return [dataset[i][1] for i in range(len(dataset))]  # type: ignore[index]


# ------------------------------------------------------------------
# Permuted MNIST
# ------------------------------------------------------------------


def create_permuted_mnist(
    n_tasks: int = 10,
    root: str = "./data",
) -> List[Tuple[Dataset, Dataset]]:
    """Create permuted-MNIST continual learning tasks.

    Each task uses the full MNIST training/test set but with a
    different random pixel permutation applied to the input features.

    Args:
        n_tasks: Number of permutation tasks.
        root:    Data cache directory.

    Returns:
        List of ``(train_dataset, test_dataset)`` per task.
    """
    train_ds, test_ds = load_dataset("mnist", root=root, normalize=False, flatten=True)

    train_tensors = _dataset_to_tensors(train_ds)
    test_tensors = _dataset_to_tensors(test_ds)

    n_features = train_tensors[0].shape[0]
    rng = np.random.RandomState(42)

    tasks: List[Tuple[TensorDataset, TensorDataset]] = []
    for _ in range(n_tasks):
        perm = rng.permutation(n_features)
        perm_tensor = torch.from_numpy(perm).long()

        train_perms = train_tensors[0][:, perm_tensor]
        test_perms = test_tensors[0][:, perm_tensor]

        tasks.append((
            TensorDataset(train_perms, train_tensors[1]),
            TensorDataset(test_perms, test_tensors[1]),
        ))

    return tasks


def _dataset_to_tensors(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a flattened dataset to (features, labels) tensors."""
    features, labels = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        features.append(x)
        labels.append(y)
    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


# ------------------------------------------------------------------
# Dataset info
# ------------------------------------------------------------------


def get_dataset_info(name: str) -> Dict[str, Any]:
    """Return basic metadata for a supported dataset.

    Args:
        name: Dataset identifier.

    Returns:
        Dict with ``n_features``, ``n_classes``, ``n_train``, ``n_test``,
        ``input_shape`` keys.
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    info_map: Dict[str, Dict[str, Any]] = {
        "mnist": {
            "n_features": 784,
            "n_classes": 10,
            "n_train": 60_000,
            "n_test": 10_000,
            "input_shape": (1, 28, 28),
        },
        "fashion_mnist": {
            "n_features": 784,
            "n_classes": 10,
            "n_train": 60_000,
            "n_test": 10_000,
            "input_shape": (1, 28, 28),
        },
        "cifar10": {
            "n_features": 3072,
            "n_classes": 10,
            "n_train": 50_000,
            "n_test": 10_000,
            "input_shape": (3, 32, 32),
        },
    }

    if name_lower not in info_map:
        raise ValueError(f"Unknown dataset '{name}'. Supported: {sorted(info_map)}")
    return info_map[name_lower]
