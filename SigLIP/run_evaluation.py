import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用GPU 2
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import logging
from datetime import datetime
import json
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np

from dataset import OCLDataset
from model import OCLModel

def setup_logging():
    """设置日志"""
    # 创建输出目录
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 设置日志格式
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"evaluation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return output_dir, timestamp

def save_results(metrics, output_dir, timestamp):
    """保存评估结果"""
    results = {
        'timestamp': timestamp,
        'metrics': metrics
    }
    
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    # 设置日志和输出目录
    output_dir, timestamp = setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 记录系统信息  
        logger.info("Starting OCL evaluation experiment")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # 加载数据集
        logger.info("Loading dataset...")
        dataset = OCLDataset(
            root_dir='/data/DATA/OCL_DATA/OCL_data',
            split='test',
            top_k_categories=10
        )
        
        # 创建dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=12,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Number of batches: {len(dataloader)}")
        
        # 初始化模型
        logger.info("Initializing model...")
        model = OCLModel()
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 运行评估
        logger.info("Starting evaluation...")
        metrics = model.evaluate(dataloader)
        
        # 保存结果
        save_results(metrics, output_dir, timestamp)
        
        # 打印结果
        logger.info("Evaluation completed")
        logger.info("Final metrics:")
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                logger.info(f"  {metric_name}: {value}")
            else:
                logger.info(f"  {metric_name}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise
    finally:
        # 清理资源
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()