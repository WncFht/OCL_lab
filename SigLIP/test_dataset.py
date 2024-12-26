import unittest
import logging
import torch
from dataset import OCLDataset
from torch.utils.data import DataLoader
import os

class TestOCLDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 设置logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger('TestOCLDataset')
        
        # 初始化数据集
        cls.data_root = '/data/DATA/OCL_DATA/OCL_data'
        cls.dataset = OCLDataset(cls.data_root, split='test', top_k_categories=10)
        
    def test_dataset_initialization(self):
        """测试数据集初始化"""
        self.assertTrue(len(self.dataset) > 0)
        self.assertEqual(len(self.dataset.top_categories), 10)
        self.assertTrue(len(self.dataset.obj2id) > 0)
        
        self.logger.info(f"Dataset size: {len(self.dataset)}")
        self.logger.info(f"Number of objects: {len(self.dataset.objs)}")
        self.logger.info(f"Top categories: {list(self.dataset.top_categories.keys())}")
        
    def test_sample_format(self):
        """测试单个样本的格式"""
        sample = self.dataset[0]
        
        # 检查所有键是否存在
        expected_keys = {'image', 'text_input_ids', 'text_attention_mask', 
                        'category', 'category_id', 'attributes', 'affordances'}
        self.assertEqual(set(sample.keys()), expected_keys)
        
        # 检查数据类型和维度
        self.assertIsInstance(sample['image'], torch.Tensor)
        self.assertEqual(sample['image'].shape, (3, 224, 224))
        
        self.assertIsInstance(sample['text_input_ids'], torch.Tensor)
        self.assertIsInstance(sample['text_attention_mask'], torch.Tensor)
        
        self.assertIsInstance(sample['attributes'], torch.Tensor)
        self.assertEqual(len(sample['attributes']), len(self.dataset.attrs))
        
        self.assertIsInstance(sample['affordances'], torch.Tensor)
        
        # 检查类别ID是否有效
        self.assertIsInstance(sample['category_id'], int)
        self.assertTrue(0 <= sample['category_id'] < len(self.dataset.objs))
        
        self.logger.info(f"Sample category: {sample['category']}")
        self.logger.info(f"Text input shape: {sample['text_input_ids'].shape}")
        
    def test_dataloader(self):
        """测试DataLoader"""
        batch_size = 12
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        batch = next(iter(dataloader))
        
        # 检查batch维度
        self.assertEqual(batch['image'].shape[0], batch_size)
        self.assertEqual(batch['text_input_ids'].shape[0], batch_size)
        self.assertEqual(batch['attributes'].shape[0], batch_size)
        self.assertEqual(batch['affordances'].shape[0], batch_size)
        
        # 检查数据类型
        self.assertEqual(batch['image'].dtype, torch.float32)
        self.assertEqual(batch['text_input_ids'].dtype, torch.long)
        self.assertEqual(batch['text_attention_mask'].dtype, torch.long)

if __name__ == '__main__':
    unittest.main()