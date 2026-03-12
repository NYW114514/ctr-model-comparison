# train_all_models.py
import os
import pickle
import traceback
from config import RESULT_DIR, MODELS
from train_utils import train_and_evaluate

def save_result(model_name: str, result: dict):
    path = os.path.join(RESULT_DIR, f"{model_name.lower()}_result.pkl")
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return path

if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)

    summary = []
    for name in MODELS:
        print("\n" + "=" * 80)
        print(f"Start: {name}")
        print("=" * 80)
        try:
            res = train_and_evaluate(name)
            summary.append(res)
            p = save_result(name, res)
            print(f"[Saved] {p}")
            print(f"{name}: bestAUC={res['best_val_auc']:.4f}, finalAUC={res['final_val_auc']:.4f}, time={res['train_time_min']:.1f}min")
        except Exception as e:
            print(f"[FAILED] {name}: {e}")
            traceback.print_exc()

    # 汇总
    print("\n" + "=" * 80)
    print("ALL DONE (Summary)")
    print("=" * 80)
    for r in sorted(summary, key=lambda x: x["best_val_auc"], reverse=True):
        print(f"{r['model']:>10} | bestAUC={r['best_val_auc']:.4f} | finalAUC={r['final_val_auc']:.4f} | time={r['train_time_min']:.1f}min")

    # 保存总汇总
    summary_path = os.path.join(RESULT_DIR, "all_models_summary.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)
    print(f"\n[Saved Summary] {summary_path}")
