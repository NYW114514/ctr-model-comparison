# train_utils.py
import os
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss

from config import (
    RESULT_DIR, EPOCHS, BATCH_SIZE, DEVICE,
    SHUFFLE, RANDOM_SEED, EMPTY_CACHE_EVERY_N_PARTS, MMAP_LOAD, EVAL_TEST_AT_END
)
from data_utils import load_processor, load_part, get_split_info, build_feature_columns
from model_factory import create_model

def set_all_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _maybe_empty_cache(step: int):
    if DEVICE == "cuda" and EMPTY_CACHE_EVERY_N_PARTS > 0 and (step % EMPTY_CACHE_EVERY_N_PARTS == 0):
        torch.cuda.empty_cache()

def evaluate_on_split(model, split: str, batch_size: int):
    parts, _ = get_split_info(split)
    preds, trues = [], []

    for i in range(parts):
        X, y = load_part(split, i, mmap=True)

        p = model.predict(X, batch_size=batch_size).ravel()

        # 预测长度必须和标签长度一致
        assert len(p) == len(y), (
            f"[{split} part {i:04d}] 预测数量不匹配: pred={len(p)} vs y={len(y)}"
        )

        # 检查数值异常（NaN/Inf）
        if not np.isfinite(p).all():
            bad = np.sum(~np.isfinite(p))
            raise ValueError(f"[{split} part {i:04d}] 预测包含 NaN/Inf: {bad} 个")

        preds.append(p)
        trues.append(y)

        del X, y, p
        _maybe_empty_cache(i + 1) 

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)

    auc = roc_auc_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    return auc, ll


def train_and_evaluate(model_name: str):
    os.makedirs(RESULT_DIR, exist_ok=True)
    set_all_seeds(RANDOM_SEED)

    proc = load_processor()
    feature_columns = build_feature_columns(proc)

    model = create_model(model_name, feature_columns, device=DEVICE)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["auc"])

    train_parts, _ = get_split_info("train")

    best_val_auc = -1.0
    best_epoch = -1
    epoch_results = []

    ckpt_path = os.path.join(RESULT_DIR, f"{model_name.lower()}_best.pth")

    train_start = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ---- Train ----
        for part_idx in range(train_parts):
            X, y = load_part("train", part_idx, mmap=MMAP_LOAD)
            model.fit(
                X, y,
                batch_size=BATCH_SIZE,
                epochs=1,
                verbose=0,
                shuffle=SHUFFLE
            )
            del X, y
            _maybe_empty_cache(part_idx + 1)

        # ---- Validate ----
        val_auc, val_ll = evaluate_on_split(model, "val", batch_size=BATCH_SIZE)

        epoch_time = time.time() - epoch_start
        epoch_results.append({
            "epoch": epoch + 1,
            "val_auc": val_auc,
            "val_logloss": val_ll,
            "epoch_time_sec": float(epoch_time),
        })
        print(
            f"[{model_name}] "
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"AUC={val_auc:.4f} | "
            f"LogLoss={val_ll:.4f} | "
            f"time={epoch_time/60:.1f}min"
        )

        # ---- Checkpoint ----
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)
            print(f"    → 最佳模型已更新 (AUC={best_val_auc:.4f})")

    train_time = time.time() - train_start

    # ---- Load best + final eval ----
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)

    final_val_auc, final_val_ll = evaluate_on_split(model, "val", batch_size=BATCH_SIZE)

    # test
    final_test_auc, final_test_ll = None, None
    if EVAL_TEST_AT_END:
        final_test_auc, final_test_ll = evaluate_on_split(model, "test", batch_size=BATCH_SIZE)

    # 清理显存给下一个模型
    if DEVICE == "cuda":
        del model
        torch.cuda.empty_cache()

    result = {
        "model": model_name,
        "random_seed": RANDOM_SEED,
        "shuffle": SHUFFLE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "train_time_min": float(train_time / 60),
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "final_val_auc": float(final_val_auc),
        "final_val_logloss": float(final_val_ll),
        "final_test_auc": None if final_test_auc is None else float(final_test_auc),
        "final_test_logloss": None if final_test_ll is None else float(final_test_ll),
        "epoch_results": epoch_results,
        "checkpoint_path": ckpt_path,
    }
    return result
