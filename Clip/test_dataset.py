import unittest
import torch
import logging
import numpy as np
from dataset import OCLDataset
from torch.utils.data import DataLoader
import os

class TestOCLDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger('TestOCLDataset')
        
        # Dataset path
        cls.dataset_path = '/data/DATA/OCL_DATA/OCL_data'
        
        # Initialize dataset
        try:
            cls.dataset = OCLDataset(
                root_dir=cls.dataset_path,
                split='test',
                top_k_categories=10
            )
            
            # Create a small dataloader for testing
            cls.dataloader = DataLoader(
                cls.dataset,
                batch_size=4,
                shuffle=False,
                num_workers=1
            )
            
        except Exception as e:
            cls.logger.error(f"Setup failed: {e}")
            raise
            
    def test_dataset_initialization(self):
        """Test basic dataset properties"""
        self.logger.info("Testing dataset initialization...")
        
        # Check dataset size
        self.logger.info(f"Dataset size: {len(self.dataset)}")
        self.assertGreater(len(self.dataset), 0)
        
        # Check category information
        self.assertIsNotNone(self.dataset.objs)
        self.logger.info(f"Number of objects: {len(self.dataset.objs)}")
        
        # Check top categories
        self.assertEqual(len(self.dataset.top_categories), 10)
        self.logger.info(f"Top categories: {list(self.dataset.top_categories.keys())}")
        
    def test_text_embeddings(self):
        """Test text embeddings computation"""
        self.logger.info("Testing text embeddings...")
        
        # Check category embeddings
        self.assertTrue(hasattr(self.dataset, 'text_features'))
        self.assertEqual(len(self.dataset.text_features), 10)
        
        # Check attribute embeddings
        self.assertTrue(hasattr(self.dataset, 'attr_text_features'))
        self.assertEqual(len(self.dataset.attr_text_features), len(self.dataset.attrs))
        
        # Check affordance embeddings
        self.assertTrue(hasattr(self.dataset, 'aff_text_features'))
        self.assertEqual(len(self.dataset.aff_text_features), len(self.dataset.affs))
        
        # Check normalization
        self.assertTrue(torch.allclose(
            torch.norm(self.dataset.text_features, dim=1),
            torch.ones(len(self.dataset.text_features)),
            atol=1e-5
        ))
        
    def test_sample_loading(self):
        """Test sample loading and format"""
        self.logger.info("Testing sample loading...")
        
        # Get a sample
        sample = next(iter(self.dataloader))
        
        # Check sample contents
        self.assertIn('image', sample)
        self.assertIn('category', sample)
        self.assertIn('category_id', sample)
        self.assertIn('attributes', sample)
        self.assertIn('affordances', sample)
        
        # Check sample formats
        self.assertEqual(sample['image'].shape[0], 4)  # batch size
        self.assertEqual(sample['image'].shape[1], 3)  # channels
        self.assertEqual(sample['image'].shape[2], 224)  # height
        self.assertEqual(sample['image'].shape[3], 224)  # width
        
        # Check label shapes
        self.assertEqual(len(sample['category_id']), 4)  # batch size
        self.assertEqual(sample['attributes'].shape[1], len(self.dataset.attrs))
        self.assertEqual(sample['affordances'].shape[1], len(self.dataset.affs))
        
        # Print a sample for inspection
        self.logger.info(f"Sample category: {sample['category'][0]}")
        
        # Check if there are any valid labels
        self.assertTrue(torch.any(sample['attributes'] == 1))
        self.assertTrue(torch.any(sample['affordances'] == 1))
        
if __name__ == '__main__':
    unittest.main()