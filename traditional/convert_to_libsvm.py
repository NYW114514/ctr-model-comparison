# convert_to_libsvm_fast.py
"""
将 .npz 数据转换为 LibSVM 和 LibFFM 格式
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ========== 配置 ==========
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

DATA_DIR = os.path.join(DATA_ROOT, "processed_data_full_optimized")
OUTPUT_DIR = os.path.join(DATA_ROOT, "libsvm_data")

N_WORKERS = max(1, cpu_count() - 2) 

def load_processor():
    with open(os.path.join(DATA_DIR, "processor.pkl"), "rb") as f:
        return pickle.load(f)

def load_part(split: str, part_idx: int):
    path = os.path.join(DATA_DIR, split, f"part_{part_idx:04d}.npz")
    return np.load(path)

def build_feature_mapping(proc):
    """构建特征索引映射"""
    dense_features = proc["dense_features"]
    sparse_features = proc["sparse_features"]
    missing_indicator_features = proc["missing_indicator_features"]
    vocab_sizes = proc["vocab_sizes"]
    
    svm_feat_id = 1
    ffm_feat_id = 1
    ffm_field_id = 1
    
    dense_feat_ids_svm = {}
    dense_feat_ids_ffm = {}
    missing_feat_ids_svm = {}
    missing_feat_ids_ffm = {}
    sparse_feat_offsets_svm = {}
    sparse_feat_offsets_ffm = {}
    field_ids = {}
    
    for feat in dense_features:
        dense_feat_ids_svm[feat] = svm_feat_id
        dense_feat_ids_ffm[feat] = ffm_feat_id
        field_ids[feat] = ffm_field_id
        svm_feat_id += 1
        ffm_feat_id += 1
        ffm_field_id += 1
    
    for feat in missing_indicator_features:
        missing_feat_ids_svm[feat] = svm_feat_id
        missing_feat_ids_ffm[feat] = ffm_feat_id
        field_ids[feat + "_missing"] = ffm_field_id
        svm_feat_id += 1
        ffm_feat_id += 1
        ffm_field_id += 1
    
    for feat in sparse_features:
        sparse_feat_offsets_svm[feat] = svm_feat_id
        sparse_feat_offsets_ffm[feat] = ffm_feat_id
        field_ids[feat] = ffm_field_id
        svm_feat_id += vocab_sizes[feat]
        ffm_feat_id += vocab_sizes[feat]
        ffm_field_id += 1
    
    print(f"特征映射统计:")
    print(f"  - 数值特征: {len(dense_features)} 个")
    print(f"  - 缺失指示特征: {len(missing_indicator_features)} 个")
    print(f"  - 类别特征: {len(sparse_features)} 个")
    print(f"  - LibSVM 特征维度: {svm_feat_id - 1}")
    print(f"  - LibFFM Field 数: {ffm_field_id}")
    
    return {
        "dense_feat_ids_svm": dense_feat_ids_svm,
        "dense_feat_ids_ffm": dense_feat_ids_ffm,
        "missing_feat_ids_svm": missing_feat_ids_svm,
        "missing_feat_ids_ffm": missing_feat_ids_ffm,
        "sparse_feat_offsets_svm": sparse_feat_offsets_svm,
        "sparse_feat_offsets_ffm": sparse_feat_offsets_ffm,
        "field_ids": field_ids,
        "vocab_sizes": vocab_sizes,
        "dense_features": dense_features,
        "sparse_features": sparse_features,
        "missing_indicator_features": missing_indicator_features,
    }

def convert_part_libsvm(data, feat_map):
    """向量化转换单个 part 为 LibSVM 格式"""
    dense_features = feat_map["dense_features"]
    sparse_features = feat_map["sparse_features"]
    missing_indicator_features = feat_map["missing_indicator_features"]
    vocab_sizes = feat_map["vocab_sizes"]
    
    labels = data["labels"].ravel().astype(np.int32)
    n = len(labels)
    
    clamp_counts = {}
    
    # 每行的所有列字符串，用二维列表收集
    # columns[col_idx] = 长度为 n 的字符串数组
    columns = []
    
    # 1. 数值特征
    for feat in dense_features:
        feat_id = feat_map["dense_feat_ids_svm"][feat]
        vals = data[feat].ravel()
        # 生成字符串，0值用空字符串
        col_strs = np.where(
            vals != 0,
            np.char.add(f"{feat_id}:", np.char.mod("%.6g", vals)),
            ""
        )
        columns.append(col_strs)
    
    # 2. 缺失指示特征
    for feat in missing_indicator_features:
        feat_id = feat_map["missing_feat_ids_svm"][feat]
        vals = data[feat + "_missing"].ravel()
        col_strs = np.where(
            vals != 0,
            np.char.add(f"{feat_id}:", np.char.mod("%.6g", vals)),
            ""
        )
        columns.append(col_strs)
    
    # 3. 类别特征
    for feat in sparse_features:
        offset = feat_map["sparse_feat_offsets_svm"][feat]
        vsz = vocab_sizes[feat]
        cat_vals = data[feat].ravel().astype(np.int32)
        
        # 越界统计和 clamp
        invalid_mask = (cat_vals < 0) | (cat_vals >= vsz)
        clamp_counts[feat] = int(np.sum(invalid_mask))
        cat_vals = np.clip(cat_vals, 0, vsz - 1)
        
        feat_ids = offset + cat_vals
        col_strs = np.char.add(np.char.mod("%d", feat_ids), ":1")
        columns.append(col_strs)
    
    # 4. 拼接每行
    # 转置：从 (n_cols, n_samples) 变成逐行处理
    lines = []
    for i in range(n):
        parts = [str(labels[i])]
        for col in columns:
            if col[i]:  # 非空才加
                parts.append(col[i])
        lines.append(" ".join(parts))
    
    return lines, clamp_counts

def convert_part_libffm(data, feat_map):
    """向量化转换单个 part 为 LibFFM 格式"""
    dense_features = feat_map["dense_features"]
    sparse_features = feat_map["sparse_features"]
    missing_indicator_features = feat_map["missing_indicator_features"]
    vocab_sizes = feat_map["vocab_sizes"]
    
    labels = data["labels"].ravel().astype(np.int32)
    n = len(labels)
    
    clamp_counts = {}
    columns = []
    
    # 1. 数值特征
    for feat in dense_features:
        field_id = feat_map["field_ids"][feat]
        feat_id = feat_map["dense_feat_ids_ffm"][feat]
        vals = data[feat].ravel()
        prefix = f"{field_id}:{feat_id}:"
        col_strs = np.where(
            vals != 0,
            np.char.add(prefix, np.char.mod("%.6g", vals)),
            ""
        )
        columns.append(col_strs)
    
    # 2. 缺失指示特征
    for feat in missing_indicator_features:
        field_id = feat_map["field_ids"][feat + "_missing"]
        feat_id = feat_map["missing_feat_ids_ffm"][feat]
        vals = data[feat + "_missing"].ravel()
        prefix = f"{field_id}:{feat_id}:"
        col_strs = np.where(
            vals != 0,
            np.char.add(prefix, np.char.mod("%.6g", vals)),
            ""
        )
        columns.append(col_strs)
    
    # 3. 类别特征
    for feat in sparse_features:
        field_id = feat_map["field_ids"][feat]
        offset = feat_map["sparse_feat_offsets_ffm"][feat]
        vsz = vocab_sizes[feat]
        cat_vals = data[feat].ravel().astype(np.int32)
        
        invalid_mask = (cat_vals < 0) | (cat_vals >= vsz)
        clamp_counts[feat] = int(np.sum(invalid_mask))
        cat_vals = np.clip(cat_vals, 0, vsz - 1)
        
        feat_ids = offset + cat_vals
        # field_id:feat_id:1
        col_strs = np.char.add(
            np.char.add(f"{field_id}:", np.char.mod("%d", feat_ids)),
            ":1"
        )
        columns.append(col_strs)
    
    # 4. 拼接每行
    lines = []
    for i in range(n):
        parts = [str(labels[i])]
        for col in columns:
            if col[i]:
                parts.append(col[i])
        lines.append(" ".join(parts))
    
    return lines, clamp_counts

def process_single_part(args):
    """单个 part 的处理函数（供多进程调用）"""
    split, part_idx, feat_map, fmt, temp_dir = args
    
    data = load_part(split, part_idx)
    
    if fmt == "libsvm":
        lines, clamp_counts = convert_part_libsvm(data, feat_map)
    else:
        lines, clamp_counts = convert_part_libffm(data, feat_map)
    
    # 写入临时文件
    temp_path = os.path.join(temp_dir, f"{split}_{part_idx:04d}.txt")
    with open(temp_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    
    return part_idx, len(lines), clamp_counts

def convert_split_parallel(split: str, n_parts: int, feat_map, fmt: str):
    """多进程并行转换整个 split"""
    out_dir = os.path.join(OUTPUT_DIR, fmt)
    temp_dir = os.path.join(out_dir, "_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    output_path = os.path.join(out_dir, f"{split}.txt")
    
    print(f"\n转换 {split} -> {fmt} 格式 (使用 {N_WORKERS} 进程)...")
    
    # 准备任务参数
    tasks = [(split, i, feat_map, fmt, temp_dir) for i in range(n_parts)]
    
    # 多进程执行
    clamp_counts = {feat: 0 for feat in feat_map["sparse_features"]}
    total_samples = 0
    
    with Pool(N_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_single_part, tasks),
            total=n_parts,
            desc=f"{split}"
        ))
    
    # 统计结果
    for part_idx, n_lines, part_clamps in results:
        total_samples += n_lines
        for feat, cnt in part_clamps.items():
            clamp_counts[feat] += cnt
    
    # 按顺序合并临时文件
    print(f"  合并临时文件...")
    with open(output_path, "w") as out_f:
        for i in range(n_parts):
            temp_path = os.path.join(temp_dir, f"{split}_{i:04d}.txt")
            with open(temp_path, "r") as temp_f:
                out_f.write(temp_f.read())
            os.remove(temp_path)
    
    # 清理临时目录
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    file_size = os.path.getsize(output_path) / (1024**3)
    print(f"  完成: {output_path} ({file_size:.2f} GB, {total_samples:,} 样本)")
    
    # 越界报告
    total_clamps = sum(clamp_counts.values())
    if total_clamps > 0:
        clamp_ratio = total_clamps / (total_samples * len(feat_map["sparse_features"])) * 100
        print(f" 越界统计: {total_clamps:,} 次 ({clamp_ratio:.4f}%)")
        top_clamps = sorted(clamp_counts.items(), key=lambda x: -x[1])[:5]
        for feat, cnt in top_clamps:
            if cnt > 0:
                print(f"     - {feat}: {cnt:,} 次")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("加载 processor...")
    proc = load_processor()
    
    print(f"\n构建特征映射...")
    feat_map = build_feature_mapping(proc)
    
    map_path = os.path.join(OUTPUT_DIR, "feature_mapping.pkl")
    with open(map_path, "wb") as f:
        pickle.dump(feat_map, f)
    print(f"特征映射已保存: {map_path}")
    
    print(f"\n使用 {N_WORKERS} 个进程并行转换")
    
    splits_info = [
        ("train", proc["train_parts"]),
        ("val", proc["val_parts"]),
        ("test", proc["test_parts"]),
    ]
    
    for fmt in ["libsvm", "libffm"]:
        print(f"\n{'='*60}")
        print(f"开始转换 {fmt.upper()} 格式")
        print(f"{'='*60}")
        
        for split, n_parts in splits_info:
            convert_split_parallel(split, n_parts, feat_map, fmt)
    
    print("\n" + "="*60)
    print("全部转换完成!")
    print("="*60)

if __name__ == "__main__":
    main()