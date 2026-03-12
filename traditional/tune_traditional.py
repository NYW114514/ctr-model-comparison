# tune_traditional.py
import os
import time
import pickle
import itertools
import csv
import numpy as np
import xlearn as xl
from multiprocessing import Process
from sklearn.metrics import roc_auc_score, log_loss

# ========== 路径 ==========
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

LIBSVM_DIR = os.path.join(DATA_ROOT, "libsvm_data", "libsvm")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "tuning")

# ========== 是否保留模型 ==========
# 0: 不保留任何bin
# 1: 只保留最优bin
# K: 保留Top-K的bin
KEEP_TOP_K = 1

# ========== 搜索空间 ==========
LR_SEARCH = {
    "lr":       [1.0],
    "lambda_1": [0.0001],
    "lambda_2": [0.0001],
}

FM_SEARCH = {
    "lr":     [0.2],
    "k":      [4],
    "lambda": [1e-07, 1e-06],
}

# ========== 固定参数 ==========
LR_FIXED = {
    "task": "binary", "epoch": 10, "opt": "ftrl",
    "alpha": 0.1, "beta": 1.0, "metric": "auc",
}
FM_FIXED = {
    "task": "binary", "epoch": 20, "opt": "adagrad", "metric": "auc",
}

def load_labels(data_dir, split):
    labels = []
    with open(os.path.join(data_dir, f"{split}.txt")) as f:
        for line in f:
            labels.append(int(line.split()[0]))
    return np.array(labels, dtype=np.float32)

def _safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except:
        pass

def _worker(model_type, params, data_dir, model_path, result_path):
    """
    子进程执行函数，结果写入result_path
    """
    pred_path = model_path + ".pred"

    try:
        if model_type == "linear":
            m = xl.create_linear()
        else:
            m = xl.create_fm()

        m.setTrain(os.path.join(data_dir, "train.txt"))
        m.setValidate(os.path.join(data_dir, "val.txt"))
        m.fit(params, model_path)

        m.setTest(os.path.join(data_dir, "val.txt"))
        m.setSigmoid()
        m.predict(model_path, pred_path)

        preds = np.loadtxt(pred_path)
        labels = load_labels(data_dir, "val")
        auc = float(roc_auc_score(labels, preds))
        ll  = float(log_loss(labels, preds))

        with open(result_path, "wb") as f:
            pickle.dump({"auc": auc, "ll": ll}, f)

    finally:
        # 尽量清理 pred
        _safe_remove(pred_path)

def run_one(model_type, params, data_dir, model_path):
    """每组单独起子进程，跑完后内存完全释放"""
    result_path = model_path + ".result"

    p = Process(target=_worker, args=(model_type, params, data_dir, model_path, result_path))
    p.start()
    p.join()

    if p.exitcode != 0:
        # 子进程异常时，尽量清理残留 result
        _safe_remove(result_path)
        raise RuntimeError(f"子进程异常退出，exitcode={p.exitcode}")

    if not os.path.exists(result_path):
        raise RuntimeError("子进程未生成结果文件（result_path不存在），可能在fit/predict阶段崩溃。")

    with open(result_path, "rb") as f:
        res = pickle.load(f)

    _safe_remove(result_path)
    return res["auc"], res["ll"]

def _compute_total(model_name, search_space):
    """不建列表，直接算总组合数"""
    if model_name == "LR":
        # 约束：lambda_1 == lambda_2
        # 等价于：选 lr * 选 lambda（只选一次）
        return len(search_space["lr"]) * len(search_space["lambda_1"])
    else:
        total = 1
        for v in search_space.values():
            total *= len(v)
        return total

def grid_search(model_name, model_type, search_space, fixed_params, data_dir):
    keys   = list(search_space.keys())
    values = list(search_space.values())
    total  = _compute_total(model_name, search_space)

    print(f"\n{'='*60}")
    print(f"{model_name} 网格搜索：共 {total} 组")
    print(f"{'='*60}")

    # CSV输出
    csv_path = os.path.join(RESULT_DIR, f"{model_name.lower()}_tuning.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys + ["val_auc", "val_logloss", "time_min", "model_path"])
        writer.writeheader()

        results = []
        idx = 0

        for combo in itertools.product(*values):
            # LR跳过lambda不一致的组合
            if model_name == "LR":
                i1 = keys.index("lambda_1")
                i2 = keys.index("lambda_2")
                if combo[i1] != combo[i2]:
                    continue

            params = dict(fixed_params)
            for k, v in zip(keys, combo):
                params[k] = v

            model_path = os.path.join(RESULT_DIR, f"{model_name.lower()}_tune_{idx}.bin")

            t0 = time.time()
            auc, ll = run_one(model_type, params, data_dir, model_path)
            elapsed = time.time() - t0

            combo_str = ", ".join(f"{k}={v}" for k, v in zip(keys, combo))
            print(f"  [{idx+1:02d}/{total}] {combo_str} | AUC={auc:.4f} | LogLoss={ll:.4f} | {elapsed/60:.1f}min")

            row = dict(zip(keys, combo))
            row.update({
                "val_auc": auc,
                "val_logloss": ll,
                "time_min": round(elapsed / 60, 2),
                "model_path": model_path
            })
            writer.writerow(row)
            csv_file.flush()

            results.append({
                "params": dict(zip(keys, combo)),
                "val_auc": auc,
                "val_logloss": ll,
                "time_min": elapsed / 60,
                "model_path": model_path,
            })
            idx += 1

    results.sort(key=lambda x: x["val_auc"], reverse=True)
    best = results[0]
    print(f"\n  最佳: {best['params']} | AUC={best['val_auc']:.4f}")
    print(f"  CSV已保存: {csv_path}")

    # ===== 按策略清理bin =====
    if KEEP_TOP_K is not None and KEEP_TOP_K >= 0:
        if KEEP_TOP_K == 0:
            keep_set = set()
        else:
            keep_set = set(r["model_path"] for r in results[:KEEP_TOP_K])

        removed = 0
        for r in results:
            mp = r["model_path"]
            if mp not in keep_set:
                _safe_remove(mp)
                removed += 1
        if KEEP_TOP_K == 0:
            print(f"  已清理所有 .bin 模型文件（删除 {removed} 个）")
        else:
            print(f"  已按 Top-{KEEP_TOP_K} 保留 .bin（删除 {removed} 个，保留 {len(keep_set)} 个）")

    return results

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    all_results = {}

    lr_results = grid_search("LR", "linear", LR_SEARCH, LR_FIXED, LIBSVM_DIR)
    all_results["LR"] = lr_results

    fm_results = grid_search("FM", "fm", FM_SEARCH, FM_FIXED, LIBSVM_DIR)
    all_results["FM"] = fm_results

    # 保存完整结果
    with open(os.path.join(RESULT_DIR, "tuning_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    print("\n" + "="*60)
    print("调参完成 - 最佳参数")
    print("="*60)
    for name, res in all_results.items():
        best = res[0]
        print(f"{name}: {best['params']} | AUC={best['val_auc']:.4f}")

if __name__ == "__main__":
    main()