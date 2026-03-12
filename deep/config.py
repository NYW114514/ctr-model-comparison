# config.py

# ========== 路径 ==========
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed_data_full_optimized")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

# ========== 训练超参数 ==========
BATCH_SIZE = 10240
EPOCHS = 4

DNN_HIDDEN_UNITS = (256, 128, 64)
DROPOUT = 0.1

L2_REG_LINEAR = 1e-5
L2_REG_EMBEDDING = 1e-5
L2_REG_DNN = 1e-5
L2_REG_CIN = 1e-5       

SHUFFLE = True
RANDOM_SEED = 2026

# ========== 性能调优 ==========
EMPTY_CACHE_EVERY_N_PARTS = 20 

# ========== 设备 ==========
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== 模型列表 ==========
#MODELS = ['WDL', 'DeepFM', 'DCN', 'xDeepFM', 'AutoInt', 'PNN', 'FiBiNET']
MODELS = ['FiBiNET']
MMAP_LOAD = True
EVAL_TEST_AT_END = True
