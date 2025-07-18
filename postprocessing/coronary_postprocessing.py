"""
Coronary artery segmentation post-processing module
Add this to: postprocessing/coronary_postprocessing.py
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_opening, binary_dilation
from skimage import morphology, measure
from skimage.morphology import ball
import torch
from typing import Tuple, Optional


class CoronaryPostProcessor:
    """
    Post-processing for coronary artery segmentation to improve connectivity and smoothness
    """
    
    def __init__(self, 
                 min_component_size: int = 50,
                 closing_radius: int = 2,
                 opening_radius: int = 1,
                 max_gap_distance: int = 3):
        """
        Initialize post-processor
        
        Args:
            min_component_size: Minimum size for connected components
            closing_radius: Radius for morphological closing
            opening_radius: Radius for morphological opening
            max_gap_distance: Maximum distance for gap filling
        """
        self.min_component_size = min_component_size
        self.closing_radius = closing_radius
        self.opening_radius = opening_radius
        self.max_gap_distance = max_gap_distance
    
    def remove_small_components(self, binary_mask: np.ndarray) -> np.ndarray:
        """Remove small disconnected components"""
        labeled_mask = measure.label(binary_mask)
        properties = measure.regionprops(labeled_mask)
        
        filtered_mask = np.zeros_like(binary_mask)
        for prop in properties:
            if prop.area >= self.min_component_size:
                filtered_mask[labeled_mask == prop.label] = 1
        
        return filtered_mask.astype(bool)
    
    def morphological_operations(self, binary_mask: np.ndarray, 
                                operation: str = "closing") -> np.ndarray:
        """Apply morphological operations"""
        if operation == "closing":
            struct_elem = ball(self.closing_radius)
            return binary_closing(binary_mask, structure=struct_elem)
        elif operation == "opening":
            struct_elem = ball(self.opening_radius)
            return binary_opening(binary_mask, structure=struct_elem)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def fill_gaps(self, binary_mask: np.ndarray) -> np.ndarray:
        """Fill small gaps using distance transform"""
        distance_transform = ndimage.distance_transform_edt(~binary_mask)
        gap_mask = distance_transform <= self.max_gap_distance
        return binary_mask | gap_mask
    
    def basic_postprocessing(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Basic post-processing pipeline:
        1. Morphological closing
        2. Remove small components  
        3. Morphological opening
        """
        # Step 1: Connect nearby structures
        processed = self.morphological_operations(binary_mask, "closing")
        
        # Step 2: Remove small components
        processed = self.remove_small_components(processed)
        
        # Step 3: Smooth result
        processed = self.morphological_operations(processed, "opening")
        
        return processed.astype(bool)
    
    def advanced_postprocessing(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Advanced post-processing pipeline:
        1. Basic morphological closing
        2. Distance-based gap filling
        3. Remove small components
        4. Final smoothing
        """
        # Step 1: Basic closing
        processed = self.morphological_operations(binary_mask, "closing")
        
        # Step 2: Fill gaps using distance transform
        processed = self.fill_gaps(processed)
        
        # Step 3: Remove small components
        processed = self.remove_small_components(processed)
        
        # Step 4: Final smoothing
        processed = self.morphological_operations(processed, "opening")
        
        return processed.astype(bool)
    
    def process(self, binary_mask: np.ndarray, method: str = "advanced") -> np.ndarray:
        """
        Apply post-processing
        
        Args:
            binary_mask: Input binary segmentation mask
            method: "basic" or "advanced"
            
        Returns:
            Post-processed binary mask
        """
        if method == "basic":
            return self.basic_postprocessing(binary_mask)
        elif method == "advanced":
            return self.advanced_postprocessing(binary_mask)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'basic' or 'advanced'")


def calculate_dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate Dice coefficient"""
    intersection = (pred * gt).sum()
    dice = (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)
    return float(dice)


def calculate_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate 95th percentile Hausdorff distance"""
    from scipy.ndimage import distance_transform_edt as edt
    
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 95.0
    
    dt_pred = edt(1 - pred)
    dt_gt = edt(1 - gt)
    
    dist_pred_to_gt = dt_pred[gt > 0]
    dist_gt_to_pred = dt_gt[pred > 0]
    
    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    
    return float(np.percentile(all_distances, 95))


def create_comparison_overlay(img: np.ndarray, 
                            gt: np.ndarray, 
                            pred_orig: np.ndarray,
                            pred_proc: np.ndarray,
                            alpha: float = 0.6) -> np.ndarray:
    """
    Create comparison overlay
    GT=green, Original=red, Processed=blue
    """
    img_norm = (img - img.min()) / (img.ptp() + 1e-8)
    rgb = np.stack([img_norm] * 3, axis=-1)
    
    # Apply colors with transparency
    rgb[gt > 0, 1] = alpha * 1.0 + (1-alpha) * rgb[gt > 0, 1]  # GT in green
    rgb[pred_orig > 0, 0] = alpha * 1.0 + (1-alpha) * rgb[pred_orig > 0, 0]  # Original in red  
    rgb[pred_proc > 0, 2] = alpha * 1.0 + (1-alpha) * rgb[pred_proc > 0, 2]  # Processed in blue
    
    return (rgb * 255).astype(np.uint8)
