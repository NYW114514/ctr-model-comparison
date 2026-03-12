# train_traditional_best.py
"""
用最佳参数重新训练传统 CTR 模型：LR、FM、FFM
"""

import os
import time
import pickle
import numpy as np
import xlearn as xl
from sklearn.metrics import roc_auc_score, log_loss

# ========== 配置 ==========
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

LIBSVM_DIR = os.path.join(DATA_ROOT, "libsvm_data", "libsvm")
LIBFFM_DIR = os.path.join(DATA_ROOT, "libsvm_data", "libffm")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "best")

MODEL_CONFIGS = {
    "LR": {
        "model_type": "linear",
        "data_dir": LIBSVM_DIR,
        "params": {
            "task": "binary",
            "lr": 1.0,
            "epoch": 10,
            "opt": "ftrl",
            "alpha": 0.1,
            "beta": 1.0,
            "lambda_1": 0.0001,
            "lambda_2": 0.0001,
            "metric": "auc",
        }
    },
    "FM": {
        "model_type": "fm",
        "data_dir": LIBSVM_DIR,
        "params": {
            "task": "binary",
            "lr": 0.2,
            "epoch": 20,
            "k": 4,
            "lambda": 1e-6,
            "metric": "auc",
            "opt": "adagrad",
        }
    },
    "FFM": {
        "model_type": "ffm",
        "data_dir": LIBFFM_DIR,
        "params": {
            "task": "binary",
            "lr": 0.05,
            "epoch": 30,
            "k": 4,
            "lambda": 0.001,
            "metric": "auc",
            "opt": "adagrad",
        }
    },
}

def load_labels(data_dir: str, split: str):
    file_path = os.path.join(data_dir, f"{split}.txt")
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            label = int(line.split()[0])
            labels.append(label)
    return np.array(labels, dtype=np.float32)

def train_model(model_name: str):
    config = MODEL_CONFIGS[model_name]
    model_type = config["model_type"]
    data_dir = config["data_dir"]
    params = config["params"].copy()

    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"{'='*60}")
    print(f"参数: {params}")

    train_path = os.path.join(data_dir, "train.txt")
    val_path   = os.path.join(data_dir, "val.txt")
    test_path  = os.path.join(data_dir, "test.txt")
    model_path = os.path.join(RESULT_DIR, f"{model_name.lower()}_best_model.bin")

    if model_type == "linear":
        model = xl.create_linear()
    elif model_type == "fm":
        model = xl.create_fm()
    elif model_type == "ffm":
        model = xl.create_ffm()

    model.setTrain(train_path)
    model.setValidate(val_path)

    train_start = time.time()
    model.fit(params, model_path)
    train_time = time.time() - train_start
    print(f"训练完成，耗时: {train_time/60:.2f} 分钟")

    # 验证集
    val_pred_path = os.path.join(RESULT_DIR, f"{model_name.lower()}_val_pred.txt")
    model.setTest(val_path)
    model.setSigmoid()
    model.predict(model_path, val_pred_path)
    val_preds  = np.loadtxt(val_pred_path)
    val_labels = load_labels(data_dir, "val")
    val_auc    = roc_auc_score(val_labels, val_preds)
    val_ll     = log_loss(val_labels, val_preds)
    print(f"Val  AUC={val_auc:.4f} | LogLoss={val_ll:.4f}")

    # 测试集
    test_pred_path = os.path.join(RESULT_DIR, f"{model_name.lower()}_test_pred.txt")
    model.setTest(test_path)
    model.setSigmoid()
    model.predict(model_path, test_pred_path)
    test_preds  = np.loadtxt(test_pred_path)
    test_labels = load_labels(data_dir, "test")
    test_auc    = roc_auc_score(test_labels, test_preds)
    test_ll     = log_loss(test_labels, test_preds)
    print(f"Test AUC={test_auc:.4f} | LogLoss={test_ll:.4f}")

    result = {
        "model": model_name,
        "params": params,
        "train_time_min": train_time / 60,
        "val_auc": val_auc,
        "val_logloss": val_ll,
        "test_auc": test_auc,
        "test_logloss": test_ll,
    }
    with open(os.path.join(RESULT_DIR, f"{model_name.lower()}_best_result.pkl"), "wb") as f:
        pickle.dump(result, f)
    return result

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    results = []
    for model_name in ["LR", "FM"]:
        try:
            result = train_model(model_name)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] {model_name} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("汇总")
    print("="*60)
    for r in sorted(results, key=lambda x: x.get("test_auc") or 0, reverse=True):
        print(f"{r['model']:>6} | Val AUC={r['val_auc']:.4f} | Test AUC={r['test_auc']:.4f} | Test LogLoss={r['test_logloss']:.4f} | time={r['train_time_min']:.1f}min")

    with open(os.path.join(RESULT_DIR, "traditional_best_summary.pkl"), "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()