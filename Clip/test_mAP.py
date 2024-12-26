import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, Subset
from dataset import OCLDataset
from model import OCLModel

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger('DemoProgram')

def create_small_loader(dataset_path, num_samples=100):
    """Create a small dataloader for testing"""
    dataset = OCLDataset(
        root_dir=dataset_path,
        split='test',
        top_k_categories=10
    )
    
    # 只使用前num_samples个样本
    subset_indices = list(range(min(num_samples, len(dataset))))
    subset = Subset(dataset, subset_indices)
    
    loader = DataLoader(
        subset,
        batch_size=4,  # 小批量便于观察
        shuffle=False,
        num_workers=0  # 单进程便于调试
    )
    
    return loader

def demo_evaluation(model, loader, logger):
    """Demo evaluation focusing on category, attribute and affordance prediction"""
    with torch.no_grad():
        batch = next(iter(loader))
        image_features = model.get_embeddings(batch)
        
        # 获取文本特征和名称列表
        cat_text_features = loader.dataset.dataset.text_features.to(model.device)
        categories = loader.dataset.dataset.categories
        attr_names = loader.dataset.dataset.attrs
        aff_dict = loader.dataset.dataset.affs
        
        # 计算相似度
        cat_sim = model.compute_similarity(image_features, cat_text_features)  # 使用CLIP默认温度系数
        attr_sim = model.compute_similarity(image_features, loader.dataset.dataset.attr_text_features.to(model.device))
        aff_sim = model.compute_similarity(image_features, loader.dataset.dataset.aff_text_features.to(model.device))
        
        # 对每个样本展示详细信息
        for sample_idx in range(min(3, len(batch['attributes']))):
            logger.info(f"\n=== Sample {sample_idx + 1} ===")
            
            # 类别分析
            logger.info("\nTop 5 predicted categories with similarities:")
            sample_cat_sim = cat_sim[sample_idx]
            top_cat_indices = torch.topk(sample_cat_sim, 5).indices
            true_cat_id = batch['category_id'][sample_idx].item()
            
            logger.info(f"Ground truth category: {categories[true_cat_id]}")
            for idx in top_cat_indices:
                logger.info(f"Category: {categories[idx]:<15} "
                          f"Similarity: {sample_cat_sim[idx]:>8.4f}")
            
            # 属性分析
            sample_attr_sim = attr_sim[sample_idx]
            top_attr_indices = torch.topk(sample_attr_sim, 10).indices
            logger.info("\nTop 10 predicted attributes with similarities:")
            for idx in top_attr_indices:
                logger.info(f"Attribute: {attr_names[idx]:<15} "
                          f"Similarity: {sample_attr_sim[idx]:>8.4f}")
            
            true_attr_indices = torch.where(batch['attributes'][sample_idx] == 1)[0]
            logger.info("\nGround truth attributes with similarities:")
            for idx in true_attr_indices:
                logger.info(f"Attribute: {attr_names[idx]:<15} "
                          f"Similarity: {sample_attr_sim[idx]:>8.4f}")
            
            # 效能分析
            sample_aff_sim = aff_sim[sample_idx]
            top_aff_indices = torch.topk(sample_aff_sim, 10).indices
            logger.info("\nTop 10 predicted affordances with similarities:")
            for idx in top_aff_indices:
                aff_name = aff_dict[idx]['word'][0] if isinstance(aff_dict[idx]['word'], list) else aff_dict[idx]['word']
                logger.info(f"Affordance: {aff_name:<15} "
                          f"Similarity: {sample_aff_sim[idx]:>8.4f} "
                          f"Definition: {aff_dict[idx]['define'].strip()}")
            
            true_aff_indices = torch.where(batch['affordances'][sample_idx] == 1)[0]
            logger.info("\nGround truth affordances with similarities:")
            for idx in true_aff_indices:
                aff_name = aff_dict[idx]['word'][0] if isinstance(aff_dict[idx]['word'], list) else aff_dict[idx]['word']
                logger.info(f"Affordance: {aff_name:<15} "
                          f"Similarity: {sample_aff_sim[idx]:>8.4f}")

        # 计算指标
        attr_labels = batch['attributes'].to(model.device)
        aff_labels = batch['affordances'].to(model.device)
        cat_accuracy = model.compute_accuracy(cat_sim, batch['category_id'].to(model.device))
        attr_map = model.compute_map(attr_sim, attr_labels)
        aff_map = model.compute_map(aff_sim, aff_labels)
        
        logger.info("\n=== Overall Metrics ===")
        logger.info(f"Category Accuracy: {cat_accuracy:.4f}")
        logger.info(f"Attribute mAP: {attr_map:.4f}")
        logger.info(f"Affordance mAP: {aff_map:.4f}")

def main():
    logger = setup_logging()
    logger.info("Starting demo evaluation...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        dataset_path = '/data/DATA/OCL_DATA/OCL_data'
        
        # 创建小数据集的loader
        loader = create_small_loader(dataset_path)
        logger.info(f"Created small loader with {len(loader)} batches")
        
        # 初始化模型
        model = OCLModel()
        
        # 运行demo评估
        demo_evaluation(model, loader, logger)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()