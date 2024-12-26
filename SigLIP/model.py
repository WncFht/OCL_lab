import torch
import torch.nn.functional as F
import logging
import numpy as np
from sklearn.metrics import average_precision_score
from transformers import SiglipProcessor, SiglipModel

class OCLModel:
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger.info(f'Initializing SigLIP model {model_name} on {self.device}')
        
        try:
            self.model = SiglipModel.from_pretrained(model_name)
            self.processor = SiglipProcessor.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Freeze parameters
            self._freeze_parameters()
            
        except Exception as e:
            self.logger.error(f'Failed to initialize model: {e}')
            raise
            
    def _freeze_parameters(self):
        """Freeze model parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
            
    def get_embeddings(self, batch):
        """Get image embeddings for a batch"""
        try:
            # Move images to device
            images = batch['image'].to(self.device)
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            with torch.no_grad():
                # Get image features
                image_features = self.model.get_image_features(pixel_values=images)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                self.logger.debug(f"Image features shape: {image_features.shape}")
                
            return image_features
            
        except Exception as e:
            self.logger.error(f'Forward pass failed: {e}')
            raise
            
    def compute_similarity(self, image_features, text_features):
        """Compute similarity between image and text features"""
        # Compute similarity matrix using sigmoid instead of cosine
        similarity = torch.mm(image_features, text_features.t())
        return similarity 
        
    def compute_accuracy(self, similarity, targets, k=1):
        """Compute top-k classification accuracy"""
        batch_size = targets.size(0)
        
        # Get top-k predictions
        _, pred_indices = similarity.topk(k, dim=1)
        correct = pred_indices.eq(targets.view(-1, 1).expand_as(pred_indices))
        
        # Calculate accuracy
        correct_k = correct[:, :k].float().sum(dim=1)
        accuracy = correct_k.sum().item() / batch_size
        
        self.logger.debug(f"Batch top-{k} accuracy: {accuracy:.4f}")
        
        return accuracy


    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate model performance"""
        total_accuracy = 0
        total_samples = 0
        total_top5_accuracy = 0
        total_top10_accuracy = 0
        all_attr_labels = []
        all_aff_labels = []
        all_attr_preds = []
        all_aff_preds = []
        
        # Get features from dataset
        text_features = dataloader.dataset.text_features.to(self.device)
        attr_text_features = dataloader.dataset.attr_text_features.to(self.device)
        aff_text_features = dataloader.dataset.aff_text_features.to(self.device)
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Get image embeddings
                image_features = self.get_embeddings(batch)
                
                # 1. Compute category classification accuracy
                category_sim = self.compute_similarity(image_features, text_features)
                targets = batch['category_id'].to(self.device)

                accuracy = self.compute_accuracy(category_sim, targets, k=1)
                top5_accuracy = self.compute_accuracy(category_sim, targets, k=5)
                top10_accuracy = self.compute_accuracy(category_sim, targets, k=10)

                batch_size = targets.size(0)
                total_accuracy += accuracy * batch_size
                total_top5_accuracy += top5_accuracy * batch_size
                total_top10_accuracy += top10_accuracy * batch_size
                total_samples += batch_size
                
                # 2. Compute attribute predictions
                attr_sim = self.compute_similarity(image_features, attr_text_features)
                attr_preds = torch.sigmoid(attr_sim)
                
                # 3. Compute affordance predictions
                aff_sim = self.compute_similarity(image_features, aff_text_features)
                aff_preds = torch.sigmoid(aff_sim)
                
                # Store labels and predictions
                all_attr_labels.append(batch['attributes'].to(self.device))
                all_aff_labels.append(batch['affordances'].to(self.device))
                all_attr_preds.append(attr_preds)
                all_aff_preds.append(aff_preds)

                if (batch_idx + 1) % 50 == 0:
                    self.logger.info(
                        f"Processed {batch_idx + 1}/{len(dataloader)} batches. "
                        f"Running top-1: {total_accuracy/total_samples:.4f}, "
                        f"top-5: {total_top5_accuracy/total_samples:.4f}, "
                        f"top-10: {total_top10_accuracy/total_samples:.4f}"
                    )
                        
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        try:
            # Combine all batch data
            all_attr_labels = torch.cat(all_attr_labels, dim=0)
            all_aff_labels = torch.cat(all_aff_labels, dim=0)
            all_attr_preds = torch.cat(all_attr_preds, dim=0)
            all_aff_preds = torch.cat(all_aff_preds, dim=0)
            
            # Calculate mAP scores
            attr_map = self.compute_map(all_attr_preds, all_attr_labels)
            aff_map = self.compute_map(all_aff_preds, all_aff_labels)
            
            # Calculate final metrics
            metrics = {
                'top1_accuracy': total_accuracy / total_samples if total_samples > 0 else 0.0,
                'top5_accuracy': total_top5_accuracy / total_samples if total_samples > 0 else 0.0,
                'top10_accuracy': total_top10_accuracy / total_samples if total_samples > 0 else 0.0,
                'attribute_map': attr_map,
                'affordance_map': aff_map
            }
            
            self.logger.info(f"Evaluation Results:")
            self.logger.info(f"  Total samples processed: {total_samples}")
            self.logger.info(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
            self.logger.info(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
            self.logger.info(f"  Top-10 Accuracy: {metrics['top10_accuracy']:.4f}")
            self.logger.info(f"  Attribute mAP: {metrics['attribute_map']:.4f}")
            self.logger.info(f"  Affordance mAP: {metrics['affordance_map']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in final computation: {e}")
            raise

    def compute_map(self, predictions, labels):
        """Compute mean Average Precision"""
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
            
        assert predictions.shape == labels.shape, \
            f"Predictions shape {predictions.shape} != labels shape {labels.shape}"
        
        self.logger.debug(f"Computing mAP for shape: {predictions.shape}")
        
        aps = []
        for i in range(predictions.shape[1]):
            if labels[:, i].sum() > 0:  # Only compute for classes with positive samples
                try:
                    ap = average_precision_score(labels[:, i], predictions[:, i])
                    if not np.isnan(ap):
                        aps.append(ap)
                except Exception as e:
                    self.logger.error(f"Error computing AP for class {i}: {e}")
                    continue
        
        if not aps:
            self.logger.warning("No valid AP scores computed")
            return 0.0
        
        mean_ap = np.mean(aps)
        self.logger.debug(f"Computed mAP: {mean_ap:.4f} from {len(aps)} classes")
        return mean_ap