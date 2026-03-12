# train_best_deep.py
# 用最佳参数重新训练7个深度模型，评估val和test
import os
import time
import pickle
import gc
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from data_utils import load_processor, load_part, get_split_info, build_feature_columns
from model_factory_tune import create_model
from config import (
    RESULT_DIR, BATCH_SIZE, DEVICE,
    SHUFFLE, RANDOM_SEED, EMPTY_CACHE_EVERY_N_PARTS, MMAP_LOAD,
)

BEST_RESULT_DIR = os.path.join(RESULT_DIR, "best")

# ========== 各模型最佳参数 ==========
MODEL_BEST_PARAMS = {
    'WDL': {
        'dnn_hidden_units': (2048, 1024, 512),
        'dropout': 0.0,
        'l2_reg_embedding': 0.005,
        'l2_reg_dnn': 0.0001,
        'lr': 0.001,
    },
    'DeepFM': {
        'dnn_hidden_units': (2048, 1024, 512),
        'dropout': 0.0,
        'l2_reg_embedding': 0.01,
        'l2_reg_dnn': 0.001,
        'lr': 0.001,
    },
    'DCN': {
        'dnn_hidden_units': (2048, 1024, 512),
        'dropout': 0.0,
        'l2_reg_embedding': 0.0001,
        'l2_reg_dnn': 0.001,
        'lr': 0.002,
        'cross_num': 4,
    },
    'xDeepFM': {
        'dnn_hidden_units': (2048, 1024, 512),
        'dropout': 0.0,
        'l2_reg_embedding': 0.01,
        'l2_reg_dnn': 0.001,
        'lr': 0.001,
        'cin_layer_size': (512, 256, 128),
    },
    'AutoInt': {
        'dnn_hidden_units': (2048, 1024, 512),
        'dropout': 0.0,
        'l2_reg_embedding': 0.01,
        'l2_reg_dnn': 0.0001,
        'lr': 0.001,
    },
    'PNN': {
        'dnn_hidden_units': (2048, 1024, 512),
        'dropout': 0.0,
        'l2_reg_embedding': 0.001,
        'l2_reg_dnn': 0.0001,
        'lr': 0.002,
    },
    'FiBiNET': {
        'dnn_hidden_units': (2048, 1024, 512),
        'dropout': 0.0,
        'l2_reg_embedding': 0.01,
        'l2_reg_dnn': 0.0001,
        'lr': 0.001,
    },
}

EPOCHS = 5

def set_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _maybe_empty_cache(step):
    if DEVICE == "cuda" and EMPTY_CACHE_EVERY_N_PARTS > 0 and step % EMPTY_CACHE_EVERY_N_PARTS == 0:
        torch.cuda.empty_cache()

def evaluate_on_split(model, split, batch_size):
    parts, _ = get_split_info(split)
    preds, trues = [], []
    for i in range(parts):
        X, y = load_part(split, i, mmap=MMAP_LOAD)
        p = model.predict(X, batch_size=batch_size).ravel()
        preds.append(p)
        trues.append(y)
        del X, y, p
        _maybe_empty_cache(i + 1)
    y_pred = np.clip(np.concatenate(preds), 1e-7, 1-1e-7)
    y_true = np.concatenate(trues)
    return float(roc_auc_score(y_true, y_pred)), float(log_loss(y_true, y_pred))

def train_one(model_name, feature_columns):
    set_all_seeds(RANDOM_SEED)
    params = MODEL_BEST_PARAMS[model_name]

    model = create_model(
        model_name, feature_columns, device=DEVICE,
        dnn_hidden_units=params['dnn_hidden_units'],
        dropout=params['dropout'],
        l2_reg_embedding=params['l2_reg_embedding'],
        l2_reg_dnn=params['l2_reg_dnn'],
        **{k: v for k, v in params.items()
           if k not in ('dnn_hidden_units', 'dropout', 'l2_reg_embedding', 'l2_reg_dnn', 'lr')},
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["auc"])
    for pg in model.optim.param_groups:
        pg['lr'] = params['lr']

    train_parts, _ = get_split_info("train")
    best_val_auc = -1.0
    best_val_ll  = None
    ckpt_path = os.path.join(BEST_RESULT_DIR, f"{model_name.lower()}_best.pth")
    train_start = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        for part_idx in range(train_parts):
            X, y = load_part("train", part_idx, mmap=MMAP_LOAD)
            model.fit(X, y, batch_size=BATCH_SIZE, epochs=1, verbose=0, shuffle=SHUFFLE)
            del X, y
            _maybe_empty_cache(part_idx + 1)

        val_auc, val_ll = evaluate_on_split(model, "val", BATCH_SIZE)
        print(f"  [{model_name}] Epoch {epoch+1}/{EPOCHS} | AUC={val_auc:.4f} | LogLoss={val_ll:.4f} | {(time.time()-epoch_start)/60:.1f}min")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_ll  = val_ll
            torch.save(model.state_dict(), ckpt_path)

    train_time = time.time() - train_start

    # 加载最佳checkpoint评估test
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    test_auc, test_ll = evaluate_on_split(model, "test", BATCH_SIZE)

    print(f"  [{model_name}] Val AUC={best_val_auc:.4f} | Test AUC={test_auc:.4f} | Test LogLoss={test_ll:.4f} | time={train_time/60:.1f}min")

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "params": params,
        "train_time_min": train_time / 60,
        "best_val_auc": best_val_auc,
        "best_val_logloss": best_val_ll,
        "test_auc": test_auc,
        "test_logloss": test_ll,
    }

def main():
    os.makedirs(BEST_RESULT_DIR, exist_ok=True)
    proc = load_processor()
    feature_columns = build_feature_columns(proc)

    summary = []
    for model_name in MODEL_BEST_PARAMS:
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")
        try:
            result = train_one(model_name, feature_columns)
            summary.append(result)
            with open(os.path.join(BEST_RESULT_DIR, f"{model_name.lower()}_best_result.pkl"), "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"[FAILED] {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("最终结果汇总")
    print("="*60)
    for r in sorted(summary, key=lambda x: x["test_auc"], reverse=True):
        print(f"{r['model']:>10} | Val AUC={r['best_val_auc']:.4f} | Test AUC={r['test_auc']:.4f} | Test LogLoss={r['test_logloss']:.4f} | time={r['train_time_min']:.1f}min")

    with open(os.path.join(BEST_RESULT_DIR, "deep_best_summary.pkl"), "wb") as f:
        pickle.dump(summary, f)

if __name__ == "__main__":
    main()