from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

DATASET_INFO = {
    "mnist": {"dirname": "MNIST", "mean": 0.1307, "std": 0.3081},
    "fashion_mnist": {"dirname": "FashionMNIST", "mean": 0.2860, "std": 0.3530},
}


def load_dataset(
    dataset: str,
    *,
    train_subset: int,
    test_subset: int,
    seed: int,
    data_dir: str | Path = "data",
    input_dim: int = 784,
    output_dim: int = 10,
):
    if dataset == "synthetic":
        return synthetic_digit_data(train_subset, test_subset, input_dim, output_dim, seed)
    return load_idx_dataset(dataset, train_subset, test_subset, seed, data_dir)


def load_idx_dataset(dataset: str, train_subset: int, test_subset: int, seed: int, data_dir: str | Path):
    info = DATASET_INFO[dataset]
    root = dataset_root(data_dir, info["dirname"])
    x_train_u8 = read_idx_images(root / "train-images-idx3-ubyte")
    y_train_u8 = read_idx_labels(root / "train-labels-idx1-ubyte")
    x_test_u8 = read_idx_images(root / "t10k-images-idx3-ubyte")
    y_test_u8 = read_idx_labels(root / "t10k-labels-idx1-ubyte")
    tr_idx = balanced_indices(y_train_u8, train_subset, seed)
    te_idx = balanced_indices(y_test_u8, test_subset, seed + 17)
    return (
        normalize_images(x_train_u8[tr_idx], info["mean"], info["std"]),
        one_hot(y_train_u8[tr_idx]),
        normalize_images(x_test_u8[te_idx], info["mean"], info["std"]),
        one_hot(y_test_u8[te_idx]),
    )


def dataset_root(data_dir: str | Path, dirname: str) -> Path:
    base = Path(data_dir)
    candidates = [
        base / dirname / "raw",
        base / dirname,
        base / "cache" / dirname / "raw",
        Path(".cache") / "datasets" / dirname / "raw",
    ]
    for candidate in candidates:
        if (candidate / "train-images-idx3-ubyte").is_file() or (candidate / "train-images-idx3-ubyte.gz").is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find {dirname} IDX files. Expected train-images-idx3-ubyte under one of: "
        + ", ".join(str(c) for c in candidates)
    )


def _open_binary(path: Path):
    if path.is_file():
        return path.open("rb")
    gz_path = Path(str(path) + ".gz")
    if gz_path.is_file():
        return gzip.open(gz_path, "rb")
    raise FileNotFoundError(path)


def read_idx_images(path: Path) -> np.ndarray:
    with _open_binary(path) as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError(f"unexpected IDX image magic {magic} at {path}")
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with _open_binary(path) as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError(f"unexpected IDX label magic {magic} at {path}")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


def balanced_indices(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n >= labels.shape[0]:
        idx = np.arange(labels.shape[0], dtype=np.int64)
        rng.shuffle(idx)
        return idx
    classes = np.arange(10)
    base = n // len(classes)
    remainder = n % len(classes)
    out = []
    for ix, cls in enumerate(classes):
        take = base + (1 if ix < remainder else 0)
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)
        out.append(cls_idx[:take])
    idx = np.concatenate(out)
    rng.shuffle(idx)
    return idx.astype(np.int64)


def one_hot(labels: np.ndarray, output_dim: int = 10) -> np.ndarray:
    y = np.zeros((labels.shape[0], output_dim), dtype=np.float32)
    y[np.arange(labels.shape[0]), labels.astype(np.int32)] = 1.0
    return y


def normalize_images(images_u8: np.ndarray, mean: float, std: float) -> np.ndarray:
    x = images_u8.astype(np.float32) / 255.0
    return ((x - mean) / std).astype(np.float32)


def synthetic_digit_data(train_subset: int, test_subset: int, input_dim: int, output_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    templates = rng.normal(size=(output_dim, input_dim)).astype(np.float32)
    templates /= np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-6)

    def make(n: int):
        labels = np.arange(n, dtype=np.int32) % output_dim
        rng.shuffle(labels)
        signal = 8.0 * templates[labels]
        noise = rng.normal(scale=0.75, size=(n, input_dim)).astype(np.float32)
        x = signal + noise
        x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-6)
        return x.astype(np.float32), one_hot(labels, output_dim)

    return (*make(train_subset), *make(test_subset))
