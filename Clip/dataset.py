import torch
from torch.utils.data import Dataset
import clip
from PIL import Image
import os
import json
import pickle
import numpy as np
import logging
from collections import Counter

class OCLDataset(Dataset):
    def __init__(self, root_dir, split='test', top_k_categories=10):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.root_dir = root_dir
        self.split = split
        self.top_k_categories = top_k_categories
        
        # Initialize CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        
        torch.cuda.empty_cache()

        # Load annotations
        pkl_dir = os.path.join(root_dir, "data/resources")
        pkl_path = os.path.join(pkl_dir, f"OCL_annot_{split}.pkl")
        
        self.logger.info(f"Loading annotations from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.annotations = pickle.load(f)
            
        # Load class information
        self.load_class_info(pkl_dir)
        
        # Filter top k categories
        self.filter_top_categories()

    def precompute_text_embeddings(self):
        """Precompute text embeddings for categories, attributes and affordances"""
        self.logger.info("Precomputing text embeddings...")
        
        with torch.no_grad():
            # 1. Category text features
            categories = list(self.top_categories.keys())
            category_texts = [f"a photo of a {category}" for category in categories]
            # category_texts = [f"{category}" for category in categories]
            category_tokens = clip.tokenize(category_texts).to(self.device)
            category_features = self.model.encode_text(category_tokens)
            self.text_features = category_features / category_features.norm(dim=-1, keepdim=True)
            
            # 2. Attribute text features
            attribute_texts = [f"a {attr} object" for attr in self.attrs]
            # attribute_texts = [f"an object that is {attr}" for attr in self.attrs]
            # attribute_texts = [f"{attr}" for attr in self.attrs]
            attribute_tokens = clip.tokenize(attribute_texts).to(self.device)
            attribute_features = self.model.encode_text(attribute_tokens)
            self.attr_text_features = attribute_features / attribute_features.norm(dim=-1, keepdim=True)
            
            # 3. Affordance text features
            affordance_texts = [f"an object that can {aff}" for aff in self.affs]
            # affordance_texts = [f"{aff}" for aff in self.affs]
            affordance_tokens = clip.tokenize(affordance_texts).to(self.device)
            affordance_features = self.model.encode_text(affordance_tokens)
            self.aff_text_features = affordance_features / affordance_features.norm(dim=-1, keepdim=True)
        
        # Log shapes for verification
        self.logger.info(f"Precomputed feature shapes:")
        self.logger.info(f"  Categories: {self.text_features.shape}")
        self.logger.info(f"  Attributes: {self.attr_text_features.shape}")
        self.logger.info(f"  Affordances: {self.aff_text_features.shape}")
        
        # Store category information
        self.categories = categories
        
    def load_class_info(self, pkl_dir):
        """Load class information"""
        def load_class_json(name):
            path = os.path.join(pkl_dir, f"OCL_class_{name}.json")
            with open(path, 'r') as f:
                return json.load(f)
                
        self.attrs = load_class_json("attribute")
        self.objs = load_class_json("object")
        self.affs = load_class_json("affordance")
        
        self.obj2id = {obj: idx for idx, obj in enumerate(self.objs)}
        
        matrix_path = os.path.join(pkl_dir, 'category_aff_matrix.json')
        with open(matrix_path, 'r') as f:
            aff_matrix = json.load(f)
            self.aff_matrix = np.array(aff_matrix["aff_matrix"])
            
    def filter_top_categories(self):
        """Filter top k categories based on sample count"""
        # Count samples per category
        category_counts = {}
        for ann in self.annotations:
            for obj in ann['objects']:
                category = obj['obj']
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Get top k categories
        self.top_categories = dict(Counter(category_counts).most_common(self.top_k_categories))
        self.logger.info(f"Selected top {len(self.top_categories)} categories")
        
        # Filter annotations
        self.filtered_annotations = []
        for ann in self.annotations:
            filtered_objects = [obj for obj in ann['objects'] if obj['obj'] in self.top_categories]
            if filtered_objects:
                ann = ann.copy()
                ann['objects'] = filtered_objects
                self.filtered_annotations.append(ann)
                
        self.logger.info(f"Filtered dataset contains {len(self.filtered_annotations)} images")

        # Create category to index mapping for the filtered categories
        self.filtered_obj2id = {obj: idx for idx, obj in enumerate(self.top_categories.keys())}
        
        # After filtering, precompute text embeddings
        self.precompute_text_embeddings()
        
    def __len__(self):
        return len(self.filtered_annotations)
        
    def __getitem__(self, idx):
        ann = self.filtered_annotations[idx]
        
        # Load and preprocess image
        img_path = os.path.join(self.root_dir, "data", ann["name"])
        image = Image.open(img_path).convert('RGB')
        
        # Handle large images
        if max(image.size) > 1800:
            w, h = image.size
            image = image.resize((w//2, h//2))
                
        # Apply CLIP preprocessing
        image = self.preprocess(image)
        
        # Get first object's category
        obj = ann['objects'][0]
        category = obj['obj']
        
        # Get attribute and affordance labels
        attr = torch.zeros(len(self.attrs))
        attr[obj['attr']] = 1
        
        aff = torch.zeros(self.aff_matrix.shape[1])
        aff[obj['aff']] = 1
        
        return {
            'image': image,
            'category': category,
            'category_id': self.filtered_obj2id[category],
            'attributes': attr,
            'affordances': aff
        }