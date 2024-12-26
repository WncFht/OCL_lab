import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import transformers
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score

# 检查CUDA
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"PyTorch版本: {torch.__version__}")
print(f"GPU数量: {torch.cuda.device_count()}")

# 检查GPU信息
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 测试transformers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
print("\nTokenizer测试成功")