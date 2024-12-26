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
    
    # 获取所有类别
    all_categories = list(dataset.top_categories.keys())
    
    # 为每个类别创建文本输入
    text_inputs = []
    for category in all_categories:
        inputs = dataset.tokenizer(
            category,
            padding='max_length',
            max_length=64,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        text_inputs.append({
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        })
    
    # 组合所有文本输入
    batch_text = {
        'input_ids': torch.cat([x['input_ids'] for x in text_inputs]),
        'attention_mask': torch.cat([x['attention_mask'] for x in text_inputs])
    }
    
    # 准备图片输入
    batch = {
        'image': sample['image'].unsqueeze(0),
        'text_input_ids': batch_text['input_ids'].to(model.device),
        'text_attention_mask': batch_text['attention_mask'].to(model.device),
        'category': sample['category'],
        'category_id': sample['category_id']
    }
    
    # 获取嵌入
    with torch.no_grad():
        # 重复图片嵌入以匹配类别数量
        image_embed = model.model(pixel_values=batch['image'].to(model.device)).image_embeds
        image_embed = image_embed.repeat(len(all_categories), 1)
        
        # 获取所有类别的文本嵌入
        text_embeds = model.model(
            input_ids=batch_text['input_ids'].to(model.device),
            attention_mask=batch_text['attention_mask'].to(model.device)
        ).text_embeds
        
        # 计算余弦相似度
        image_embed_norm = torch.nn.functional.normalize(image_embed, p=2, dim=1)
        text_embeds_norm = torch.nn.functional.normalize(text_embeds, p=2, dim=1)
        similarities = torch.sum(image_embed_norm * text_embeds_norm, dim=1)
    
    # 获取排序后的索引
    sorted_indices = torch.argsort(similarities, descending=True)
    
    # 打印结果
    logger.info(f"\nResults for image with true category: {batch['category']}")
    logger.info("\nTop 10 predictions:")
    for i, idx in enumerate(sorted_indices[:10]):
        category = all_categories[idx]
        similarity = similarities[idx].item()
        is_correct = category == batch['category']
        logger.info(f"{i+1}. {category}: {similarity:.4f} {'✓' if is_correct else ''}")
    
    # 计算true category的排名
    true_category_index = all_categories.index(batch['category'])
    true_category_rank = (sorted_indices == true_category_index).nonzero().item() + 1
    
    logger.info(f"\nTrue category rank: {true_category_rank}")
    logger.info(f"Is in top-10: {true_category_rank <= 10}")

if __name__ == '__main__':
    single_image_demo(image_index=0)