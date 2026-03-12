# data_utils.py
import os
import pickle
import numpy as np
from deepctr_torch.inputs import SparseFeat, DenseFeat
from config import DATA_DIR

_processor_info = None

def load_processor():
    global _processor_info
    if _processor_info is None:
        with open(os.path.join(DATA_DIR, "processor.pkl"), "rb") as f:
            _processor_info = pickle.load(f)
    return _processor_info

def load_part(split: str, part_idx: int, mmap: bool = True):
    path = os.path.join(DATA_DIR, split, f"part_{part_idx:04d}.npz")
    data = np.load(path, mmap_mode="r" if mmap else None)
    y = data["labels"].astype("float32").ravel()
    X = {k: data[k] for k in data.files if k != "labels"}
    return X, y

def get_split_info(split: str):
    proc = load_processor()
    if split == "train":
        return proc["train_parts"], proc["train_samples"]
    if split == "val":
        return proc["val_parts"], proc["val_samples"]
    if split == "test":
        return proc["test_parts"], proc["test_samples"]
    raise ValueError(f"Unknown split: {split}")

def build_feature_columns(proc: dict):
    dense_features = proc["dense_features"]
    sparse_features = proc["sparse_features"]
    missing_indicator_features = proc["missing_indicator_features"]
    vocab_sizes = proc["vocab_sizes"]
    embedding_dim = proc["embedding_dim"]

    columns = []
    for feat in dense_features:
        columns.append(DenseFeat(feat, 1))
    for feat in missing_indicator_features:
        columns.append(DenseFeat(feat + "_missing", 1))
    for feat in sparse_features:
        columns.append(SparseFeat(feat, vocab_sizes[feat], embedding_dim=embedding_dim))

    return columns
