import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:128,"
    )
    torch.cuda.empty_cache()
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_data_loader(dataset_path, logger, batch_size=12, num_workers=4):
    """Create dataset and data loader"""
    try:
        # Initialize SigLIP model first
        model = OCLModel()
        
        # Create dataset with shared model
        dataset = OCLDataset(
            root_dir=dataset_path,
            split='test',
            top_k_categories=10,
            model=model.model,
            processor=model.processor
        )
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2
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
        
        # Dataset path
        dataset_path = '/data/DATA/OCL_DATA/OCL_data'
        
        # Create data loader
        logger.info("Creating data loader...")
        loader = create_data_loader(dataset_path, logger)
        logger.info(f"Created loader with {len(loader)} batches")
        
        # Initialize model
        logger.info("Initializing model...")
        model = OCLModel()
        logger.info("Model initialized successfully")
        
        # Run evaluation
        logger.info("Starting evaluation...")
        metrics = model.evaluate(loader)
        
        # Print results
        logger.info("Test Results:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        torch.cuda.empty_cache()
        logger.info(f"Final GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

if __name__ == '__main__':
    main()