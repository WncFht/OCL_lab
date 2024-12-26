import os
import torch
from model import OCLModel
from dataset import OCLDataset
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('SingleImageDemo')

def single_image_demo(image_index=0):
    # 设置日志
    logger = setup_logging()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    torch.cuda.empty_cache()
    
    # 初始化数据集和模型
    logger.info("Loading dataset and model...")
    dataset = OCLDataset('/data/DATA/OCL_DATA/OCL_data', split='test')
    model = OCLModel()
    
    # 获取指定索引的样本
    sample = dataset[image_index]
    
    # 转换为batch格式（添加batch维度）
    batch = {
        'image': sample['image'].unsqueeze(0),
        'text_input_ids': sample['text_input_ids'].unsqueeze(0),
        'text_attention_mask': sample['text_attention_mask'].unsqueeze(0),
        'category': sample['category'],
        'category_id': sample['category_id']
    }
    
    # 打印样本信息
    logger.info(f"\nProcessing sample {image_index}")
    logger.info(f"Category: {batch['category']}")
    logger.info(f"Image shape: {batch['image'].shape}")
    logger.info(f"Text input shape: {batch['text_input_ids'].shape}")
    
    # 获取嵌入
    with torch.no_grad():
        # 将数据移到GPU
        image_embeds, text_embeds = model.get_embeddings(batch)
    
    # 打印嵌入维度
    logger.info(f"\nEmbedding dimensions:")
    logger.info(f"Image embedding: {image_embeds.shape}")
    logger.info(f"Text embedding: {text_embeds.shape}")
    
    # 计算余弦相似度
    with torch.no_grad():
        # 归一化嵌入
        image_embeds_norm = torch.nn.functional.normalize(image_embeds, p=2, dim=1)
        text_embeds_norm = torch.nn.functional.normalize(text_embeds, p=2, dim=1)
        
        # 计算余弦相似度
        similarity = torch.sum(image_embeds_norm * text_embeds_norm, dim=1)
    
    # 打印结果
    logger.info("\nResults:")
    logger.info(f"Image category: {batch['category']}")
    logger.info(f"Cosine similarity: {similarity.item():.4f}")

if __name__ == '__main__':
    # 可以通过修改这里的索引来测试不同的图片
    single_image_demo(image_index=1)