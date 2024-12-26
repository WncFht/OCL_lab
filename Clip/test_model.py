import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

import torch
import logging
from torch.utils.data import DataLoader
import numpy as np
from dataset import OCLDataset
from model import OCLModel

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_results.log')
        ]
    )
    return logging.getLogger('TestProgram')

def setup_gpu():
    """Setup GPU and memory configurations"""
    torch.cuda.empty_cache()
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_data_loader(dataset_path, logger, batch_size=12, num_workers=2):
    """Create dataset and data loader"""
    try:
        # Create dataset
        dataset = OCLDataset(
            root_dir=dataset_path,
            split='test',
            top_k_categories=10
        )

        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return loader

    except Exception as e:
        logger.error(f"Failed to create data loader: {e}")
        raise

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting test program...")

    try:
        # Setup GPU
        device = setup_gpu()
        logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

        dataset_path = '/data/DATA/OCL_DATA/OCL_data' 

        logger.info("Initializing model...")
        model = OCLModel()
        logger.info("Model initialized successfully")

        torch.cuda.empty_cache()

        logger.info("Creating data loader...")
        loader = create_data_loader(dataset_path, logger)
        logger.info(f"Created loader with {len(loader)} batches")

        logger.info("Starting evaluation...")
        metrics = model.evaluate(loader)

        # Print results
        logger.info("Test Results:")
        logger.info("-" * 50)
        logger.info(f"Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
        logger.info(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        logger.info(f"Top-10 Accuracy: {metrics['top10_accuracy']:.4f}")
        logger.info(f"Attribute mAP: {metrics['attribute_map']:.4f}")
        logger.info(f"Affordance mAP: {metrics['affordance_map']:.4f}")
        logger.info("-" * 50)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # 清理内存
        torch.cuda.empty_cache()
        logger.info(f"Final GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

if __name__ == '__main__':
    main()