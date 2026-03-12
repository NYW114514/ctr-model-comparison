import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import time
import random

# ==================== 固定随机种子====================
random.seed(2026)
np.random.seed(2026)

# ==================== 配置 ====================
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed_data_full_optimized")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")

DENSE_FEATURES = [f'I{i}' for i in range(1, 14) if i != 12]
SPARSE_FEATURES = [f'C{i}' for i in range(1, 27) if i != 22]

# 差异化hash桶
HIGH_CARDINALITY_FEATURES = ['C3', 'C4', 'C12', 'C16', 'C21', 'C24', 'C26']
HASH_BUCKET_SIZES = {
    'C3':  1_500_000,  
    'C12': 1_500_000,   
    'C21': 1_000_000,   
    'C16': 1_000_000, 
    'C4':    300_000,   
    'C24':   100_000,  
    'C26':    50_000,  
}

# 长尾特征需要log变换
LOG_FEATURES = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I13']

RARE_THRESHOLD = 10
EMBEDDING_DIM = 8
CHUNKSIZE = 500_000

print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"随机种子: 2026")
print(f"\n优化配置:")
print(f"  - 对数变换特征: {LOG_FEATURES}")
print(f"  - 差异化Hash桶: {len(HASH_BUCKET_SIZES)} 个特征")

column_names = ['Label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]

# ==================== 稳定快速hash ====================
def stable_hash_to_bucket_series(s: pd.Series, bucket_size: int) -> np.ndarray:
    """稳定且快速的hash"""
    h = pd.util.hash_pandas_object(s, index=False).to_numpy(dtype=np.uint64)
    return (h % bucket_size).astype(np.int32)

# ==================== 第0步：统计总行数 ====================
print("\n[0/4] 统计总行数...")
start_time = time.time()
total_rows = sum(1 for _ in open(TRAIN_FILE, "rb"))
elapsed = time.time() - start_time
print(f"  总行数: {total_rows:,} ({elapsed/60:.1f}分钟)")

train_end = int(total_rows * 0.8)
val_end = int(total_rows * 0.9)

print(f"  训练集: 0 - {train_end:,}")
print(f"  验证集: {train_end:,} - {val_end:,}")
print(f"  测试集: {val_end:,} - {total_rows:,}")

# ==================== 第1步：扫描训练集 ====================
print("\n[1/4] 扫描训练集...")

low_cardinality_features = [f for f in SPARSE_FEATURES if f not in HIGH_CARDINALITY_FEATURES]
sparse_value_counts = {feat: {} for feat in low_cardinality_features}

dense_stats = {feat: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0} for feat in DENSE_FEATURES}
missing_counts = {feat: 0 for feat in DENSE_FEATURES}

row_idx = 0
chunk_iter = pd.read_csv(TRAIN_FILE, sep='\t', header=None, names=column_names, chunksize=CHUNKSIZE)

start_time = time.time()

for chunk_idx, chunk in enumerate(chunk_iter):
    chunk_size = len(chunk)
    
    if row_idx >= train_end:
        break
    if row_idx + chunk_size > train_end:
        chunk = chunk.iloc[:train_end - row_idx]
        chunk_size = len(chunk)
    
    chunk = chunk.drop(columns=['I12', 'C22'])
    
    # 统计低基数类别特征
    for feat in low_cardinality_features:
        col = chunk[feat].fillna('__MISSING__').astype(str)
        vc = col.value_counts()
        for val, cnt in vc.items():
            sparse_value_counts[feat][val] = sparse_value_counts[feat].get(val, 0) + cnt
    
    # 统计数值特征（先log再统计）
    for feat in DENSE_FEATURES:
        col = pd.to_numeric(chunk[feat], errors='coerce')
        
        missing_counts[feat] += col.isnull().sum()
        
        valid = col.notnull()
        if valid.any():
            values = col.loc[valid].values.astype(np.float32)
            
            # === 对长尾特征先做log1p，负值视为缺失 ===
            if feat in LOG_FEATURES:
                # 负值视为缺失
                neg_mask = values < 0
                if neg_mask.any():
                    missing_counts[feat] += neg_mask.sum()
                    values = values[~neg_mask]
                
                if len(values) > 0:
                    values = np.log1p(values)  # 现在values全是非负的
            
            if len(values) > 0:
                dense_stats[feat]['sum'] += float(values.sum())
                dense_stats[feat]['sum_sq'] += float((values ** 2).sum())
                dense_stats[feat]['count'] += int(len(values))
    
    row_idx += chunk_size
    
    if (chunk_idx + 1) % 10 == 0:
        elapsed = time.time() - start_time
        speed = row_idx / elapsed
        progress = row_idx / train_end * 100
        print(f"  已扫描 {row_idx:,}/{train_end:,} ({progress:.1f}%, {speed:.0f}行/秒)")

elapsed = time.time() - start_time
print(f"  扫描完成! {elapsed/60:.1f}分钟")

# ==================== 第2步：构建编码器 ====================
print("\n[2/4] 构建编码器...")

label_encoders = {}
vocab_sizes = {}
mappings = {}
rare_values_dict = {}

for feat in low_cardinality_features:
    rare_values = {val for val, cnt in sparse_value_counts[feat].items() 
                   if cnt < RARE_THRESHOLD}
    rare_values_dict[feat] = rare_values
    
    classes = set(sparse_value_counts[feat].keys()) - rare_values
    classes.add('__RARE__')
    classes = sorted(list(classes))
    
    le = LabelEncoder()
    le.fit(classes)
    label_encoders[feat] = le
    vocab_sizes[feat] = len(classes)
    mappings[feat] = {c: i for i, c in enumerate(le.classes_)}
    
    print(f"  {feat}: vocab_size={vocab_sizes[feat]} (LabelEncoder)")

del sparse_value_counts

# 高基数特征
for feat in HIGH_CARDINALITY_FEATURES:
    vocab_sizes[feat] = HASH_BUCKET_SIZES[feat]
    print(f"  {feat}: vocab_size={HASH_BUCKET_SIZES[feat]:,} (Hash)")

# 数值特征：计算mean/std
feature_mean_std = {}
missing_indicator_features = []

for feat in DENSE_FEATURES:
    count = dense_stats[feat]['count']
    if count > 0:
        mean = dense_stats[feat]['sum'] / count
        var = (dense_stats[feat]['sum_sq'] / count) - (mean ** 2)
        std = np.sqrt(max(var, 0))
        std = max(std, 1e-12)
        feature_mean_std[feat] = {'mean': mean, 'std': std}
    else:
        # 兜底：全缺失特征
        feature_mean_std[feat] = {'mean': 0.0, 'std': 1.0}
        print(f"  警告: {feat} 全缺失，使用默认 mean=0, std=1")
    
    missing_rate = missing_counts[feat] / train_end
    if missing_rate > 0:
        missing_indicator_features.append(feat)
        log_tag = " [log]" if feat in LOG_FEATURES else ""
        print(f"  {feat}: 缺失率={missing_rate:.2%}, mean={feature_mean_std[feat]['mean']:.4f}, std={feature_mean_std[feat]['std']:.4f}{log_tag}")

del dense_stats, missing_counts

# ==================== 第3步：处理并保存 ====================
print("\n[3/4] 处理并分块保存...")

def process_chunk(chunk, split_name, part_idx):
    """处理chunk并保存"""
    
    chunk = chunk.drop(columns=['I12', 'C22'])
    
    # 数值特征
    for feat in DENSE_FEATURES:
        arr = pd.to_numeric(chunk[feat], errors='coerce').to_numpy(dtype=np.float32)
        miss = np.isnan(arr)
        
        if feat in missing_indicator_features:
            chunk[feat + '_missing'] = miss.astype(np.float32)
        
        # === 先log变换（负值视为缺失）===
        if feat in LOG_FEATURES:
            # 把负值视为缺失
            neg_mask = ~miss & (arr < 0)
            if neg_mask.any():
                miss[neg_mask] = True
                arr[neg_mask] = np.nan
                # 同步更新missing indicator
                if feat in missing_indicator_features:
                    chunk[feat + '_missing'] = miss.astype(np.float32)
            
            # 对非缺失值做log1p
            mask = ~miss
            if mask.any():
                arr[mask] = np.log1p(arr[mask])
        
        # === 再标准化 ===
        mean = feature_mean_std[feat]['mean']
        std = feature_mean_std[feat]['std']
        arr[~miss] = (arr[~miss] - mean) / std
        arr[miss] = 0.0
        chunk[feat] = arr
    
    # 类别特征
    for feat in SPARSE_FEATURES:
        col = chunk[feat].fillna('__MISSING__').astype(str)
        
        if feat in HIGH_CARDINALITY_FEATURES:
            chunk[feat] = stable_hash_to_bucket_series(col, HASH_BUCKET_SIZES[feat])
        else:
            rare_mask = col.isin(rare_values_dict[feat])
            col.loc[rare_mask] = '__RARE__'
            rare_idx = mappings[feat].get('__RARE__', 0)
            chunk[feat] = col.map(mappings[feat]).fillna(rare_idx).astype('int32')
    
    # 构造model_input
    model_input = {}
    for feat in DENSE_FEATURES:
        model_input[feat] = chunk[feat].to_numpy(dtype=np.float32)
    for feat in missing_indicator_features:
        model_input[feat + '_missing'] = chunk[feat + '_missing'].to_numpy(dtype=np.float32)
    for feat in SPARSE_FEATURES:
        model_input[feat] = chunk[feat].to_numpy(dtype=np.int32)
    
    labels = chunk['Label'].to_numpy(dtype=np.float32)
    
    # 保存
    save_path = os.path.join(OUTPUT_DIR, split_name, f"part_{part_idx:04d}.npz")
    np.savez_compressed(save_path, labels=labels, **model_input)
    
    return len(labels)

# 处理全部数据
row_idx = 0
train_part_idx = 0
val_part_idx = 0
test_part_idx = 0
train_samples = 0
val_samples = 0
test_samples = 0

chunk_iter = pd.read_csv(TRAIN_FILE, sep='\t', header=None, names=column_names, chunksize=CHUNKSIZE)

start_time = time.time()

for chunk_idx, chunk in enumerate(chunk_iter):
    chunk_size = len(chunk)
    chunk_end = row_idx + chunk_size
    
    if chunk_end <= train_end:
        n = process_chunk(chunk, 'train', train_part_idx)
        train_part_idx += 1
        train_samples += n
    elif row_idx >= val_end:
        n = process_chunk(chunk, 'test', test_part_idx)
        test_part_idx += 1
        test_samples += n
    elif row_idx >= train_end and chunk_end <= val_end:
        n = process_chunk(chunk, 'val', val_part_idx)
        val_part_idx += 1
        val_samples += n
    else:
        if row_idx < train_end < chunk_end:
            split_point = train_end - row_idx
            n = process_chunk(chunk.iloc[:split_point], 'train', train_part_idx)
            train_part_idx += 1
            train_samples += n
            
            if chunk_end <= val_end:
                n = process_chunk(chunk.iloc[split_point:], 'val', val_part_idx)
                val_part_idx += 1
                val_samples += n
            else:
                val_split = val_end - train_end
                n = process_chunk(chunk.iloc[split_point:split_point+val_split], 'val', val_part_idx)
                val_part_idx += 1
                val_samples += n
                n = process_chunk(chunk.iloc[split_point+val_split:], 'test', test_part_idx)
                test_part_idx += 1
                test_samples += n
        elif row_idx < val_end < chunk_end:
            split_point = val_end - row_idx
            n = process_chunk(chunk.iloc[:split_point], 'val', val_part_idx)
            val_part_idx += 1
            val_samples += n
            n = process_chunk(chunk.iloc[split_point:], 'test', test_part_idx)
            test_part_idx += 1
            test_samples += n
    
    row_idx = chunk_end
    
    if (chunk_idx + 1) % 10 == 0:
        elapsed = time.time() - start_time
        speed = row_idx / elapsed
        progress = row_idx / total_rows * 100
        eta = (total_rows - row_idx) / speed / 60 if speed > 0 else 0
        print(f"  已处理 {row_idx:,}/{total_rows:,} ({progress:.1f}%, {speed:.0f}行/秒, 剩余约{eta:.1f}分钟)")

elapsed = time.time() - start_time
print(f"  处理完成! 总耗时 {elapsed/60:.1f}分钟")

# ==================== 第4步：保存元数据 ====================
print("\n[4/4] 保存元数据...")

processor_info = {
    'label_encoders': label_encoders,
    'feature_mean_std': feature_mean_std,
    'vocab_sizes': vocab_sizes,
    'dense_features': DENSE_FEATURES,
    'sparse_features': SPARSE_FEATURES,
    'missing_indicator_features': missing_indicator_features,
    'high_cardinality_features': HIGH_CARDINALITY_FEATURES,
    'low_cardinality_features': low_cardinality_features,
    'hash_bucket_sizes': HASH_BUCKET_SIZES,
    'log_features': LOG_FEATURES,
    'rare_threshold': RARE_THRESHOLD,
    'embedding_dim': EMBEDDING_DIM,
    'random_seed': 2026,
    'train_parts': train_part_idx,
    'val_parts': val_part_idx,
    'test_parts': test_part_idx,
    'train_samples': train_samples,
    'val_samples': val_samples,
    'test_samples': test_samples,
}

with open(os.path.join(OUTPUT_DIR, 'processor.pkl'), 'wb') as f:
    pickle.dump(processor_info, f)

# ==================== 输出总结 ====================
print("\n" + "=" * 80)
print("预处理完成！")
print("=" * 80)
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

print("\n【数据集大小】")
print(f"  训练集: {train_samples:,} 样本, {train_part_idx} 文件")
print(f"  验证集: {val_samples:,} 样本, {val_part_idx} 文件")
print(f"  测试集: {test_samples:,} 样本, {test_part_idx} 文件")

print("\n【特征信息】")
print(f"  数值特征: {len(DENSE_FEATURES)} (其中{len(LOG_FEATURES)}个做log变换)")
print(f"  Missing indicator: {len(missing_indicator_features)}")
print(f"  类别特征: {len(SPARSE_FEATURES)} (其中{len(HIGH_CARDINALITY_FEATURES)}个用差异化Hash)")

print("\n【Embedding参数量估算】")
low_card_params = sum(vocab_sizes[f] * EMBEDDING_DIM for f in low_cardinality_features)
high_card_params = sum(HASH_BUCKET_SIZES[f] * EMBEDDING_DIM for f in HIGH_CARDINALITY_FEATURES)
total_embedding_params = low_card_params + high_card_params

print(f"  低基数特征: {low_card_params:,} ({low_card_params/1e6:.2f}M)")
print(f"  高基数特征: {high_card_params:,} ({high_card_params/1e6:.2f}M)")
print(f"  总Embedding: {total_embedding_params:,} ({total_embedding_params/1e6:.2f}M)")
