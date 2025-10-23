#!/usr/bin/env python3
"""
Confusion matrix computation utilities for bird call detection evaluation.

This module provides functions to compute confusion matrices from detection results
and ground truth labels in CSV format.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_detections_csv(csv_path: str) -> pd.DataFrame:
    """
    Load detections from CSV file.
    
    Args:
        csv_path: Path to the detections CSV file
        
    Returns:
        DataFrame with columns: Filename, Start Time (s), End Time (s), Low Freq (Hz), High Freq (Hz), Species eBird Code
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_columns = ['Filename', 'Start Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Species eBird Code']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_columns}")
    
    return df


def load_labels_csv(csv_path: str) -> pd.DataFrame:
    """
    Load ground truth labels from CSV file.
    
    Args:
        csv_path: Path to the labels CSV file
        
    Returns:
        DataFrame with columns: Filename, Start Time (s), End Time (s), Low Freq (Hz), High Freq (Hz), Species eBird Code
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_columns = ['Filename', 'Start Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Species eBird Code']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_columns}")
    
    return df


def compute_confusion_matrix(detections_df: pd.DataFrame, labels_df: pd.DataFrame, 
                           iou_threshold: float = 0.5, time_tolerance: float = 0.5) -> Tuple[np.ndarray, List[str]]:
    """
    Compute confusion matrix for bird call detections.
    
    Args:
        detections_df: DataFrame with detection results
        labels_df: DataFrame with ground truth labels
        iou_threshold: IoU threshold for considering detections as matches (default: 0.5)
        time_tolerance: Time tolerance in seconds for matching detections (default: 0.5)
        
    Returns:
        Tuple of (confusion_matrix, class_labels)
    """
    # Get all unique species from both detections and labels
    detection_species = set(detections_df['Species eBird Code'].unique())
    label_species = set(labels_df['Species eBird Code'].unique())
    all_species = sorted(list(detection_species.union(label_species)))
    
    # Add background class as the last class
    all_species.append('background')
    
    # Create mapping from species to index
    species_to_idx = {species: idx for idx, species in enumerate(all_species)}
    n_classes = len(all_species)
    background_idx = n_classes - 1  # Background is always the last class
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Normalize filenames by removing extensions for matching
    detections_df_copy = detections_df.copy()
    labels_df_copy = labels_df.copy()
    
    # Remove file extensions for matching
    detections_df_copy['Filename_Base'] = detections_df_copy['Filename'].str.replace(r'\.[^.]+$', '', regex=True)
    labels_df_copy['Filename_Base'] = labels_df_copy['Filename'].str.replace(r'\.[^.]+$', '', regex=True)
    
    # Group by base filename for efficient processing
    detection_groups = detections_df_copy.groupby('Filename_Base')
    label_groups = labels_df_copy.groupby('Filename_Base')
    
    # Process each file
    for filename_base in detection_groups.groups.keys():
        if filename_base not in label_groups.groups:
            # File has detections but no ground truth - all detections are false positives
            file_detections = detection_groups.get_group(filename_base)
            for _, detection in file_detections.iterrows():
                pred_idx = species_to_idx[detection['Species eBird Code']]
                confusion_matrix[background_idx, pred_idx] += 1  # [true=background, pred=species]
            continue
        
        file_detections = detection_groups.get_group(filename_base)
        file_labels = label_groups.get_group(filename_base)
        
        # Convert to list of dictionaries for easier processing
        detections = file_detections.to_dict('records')
        labels = file_labels.to_dict('records')
        
        # Track which labels have been matched
        matched_labels = set()
        
        # For each detection, find the best matching label
        for detection in detections:
            best_match = None
            best_iou = 0
            
            for label_idx, label in enumerate(labels):
                if label_idx in matched_labels:
                    continue
                
                # Check if species match
                if detection['Species eBird Code'] != label['Species eBird Code']:
                    continue
                
                # Compute temporal IoU
                iou = compute_temporal_iou(
                    detection['Start Time (s)'], detection['End Time (s)'],
                    label['Start Time (s)'], label['End Time (s)']
                )
                
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match = label_idx
            
            if best_match is not None:
                # True positive
                pred_idx = species_to_idx[detection['Species eBird Code']]
                confusion_matrix[pred_idx, pred_idx] += 1  # [true_idx, pred_idx]
                matched_labels.add(best_match)
            else:
                # False positive
                pred_idx = species_to_idx[detection['Species eBird Code']]
                confusion_matrix[background_idx, pred_idx] += 1  # [true=background, pred=species]
        
        # Remaining unmatched labels are false negatives
        for label_idx, label in enumerate(labels):
            if label_idx not in matched_labels:
                true_idx = species_to_idx[label['Species eBird Code']]
                confusion_matrix[true_idx, background_idx] += 1  # [true=species, pred=background]
    
    # Add true negatives (correctly identified background)
    # This is computed as the total audio duration minus all detected and labeled regions
    confusion_matrix = add_true_negatives(confusion_matrix, detections_df, labels_df, background_idx)
    
    return confusion_matrix, all_species


def add_true_negatives(confusion_matrix: np.ndarray, detections_df: pd.DataFrame, 
                      labels_df: pd.DataFrame, background_idx: int) -> np.ndarray:
    """
    Add true negatives to the confusion matrix.
    
    True negatives represent correctly identified background regions (no bird calls detected
    where no bird calls were present). This is approximated by considering the total audio
    duration and subtracting all detected and labeled regions.
    
    Args:
        confusion_matrix: Current confusion matrix
        detections_df: DataFrame with detection results
        labels_df: DataFrame with ground truth labels
        background_idx: Index of the background class
        
    Returns:
        Updated confusion matrix with true negatives
    """
    # Get all unique files
    all_files = set(detections_df['Filename'].str.replace(r'\.[^.]+$', '', regex=True)).union(
                set(labels_df['Filename'].str.replace(r'\.[^.]+$', '', regex=True)))
    
    total_true_negatives = 0
    
    for filename_base in all_files:
        # Get detections and labels for this file
        file_detections = detections_df[detections_df['Filename'].str.replace(r'\.[^.]+$', '', regex=True) == filename_base]
        file_labels = labels_df[labels_df['Filename'].str.replace(r'\.[^.]+$', '', regex=True) == filename_base]
        
        # Estimate total audio duration (this is a rough approximation)
        # In practice, you might want to get actual audio file durations
        max_time = 0
        
        # Find maximum time from detections and labels
        if not file_detections.empty:
            max_time = max(max_time, file_detections['End Time (s)'].max())
        if not file_labels.empty:
            max_time = max(max_time, file_labels['End Time (s)'].max())
        
        # If no detections or labels, assume a default duration
        if max_time == 0:
            max_time = 60.0  # Default 1 minute
        
        # Calculate total labeled and detected time
        total_labeled_time = 0
        total_detected_time = 0
        
        for _, label in file_labels.iterrows():
            total_labeled_time += label['End Time (s)'] - label['Start Time (s)']
        
        for _, detection in file_detections.iterrows():
            total_detected_time += detection['End Time (s)'] - detection['Start Time (s)']
        
        # True negatives = total time - max(labeled_time, detected_time)
        # This is a conservative estimate
        true_negative_time = max_time - max(total_labeled_time, total_detected_time)
        
        # Convert to a reasonable count (assuming 1-second intervals)
        true_negatives_count = max(0, int(true_negative_time))
        total_true_negatives += true_negatives_count
    
    # Add true negatives to the confusion matrix
    confusion_matrix[background_idx, background_idx] = total_true_negatives
    
    return confusion_matrix


def compute_temporal_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """
    Compute temporal IoU between two time intervals.
    
    Args:
        start1, end1: Start and end times of first interval
        start2, end2: Start and end times of second interval
        
    Returns:
        Temporal IoU value between 0 and 1
    """
    # Compute intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    
    if intersection_start >= intersection_end:
        return 0.0
    
    intersection = intersection_end - intersection_start
    
    # Compute union
    union = (end1 - start1) + (end2 - start2) - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def normalize_confusion_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize confusion matrix by row (true labels).
    
    Args:
        confusion_matrix: Raw confusion matrix
        
    Returns:
        Normalized confusion matrix
    """
    # Avoid division by zero
    row_sums = confusion_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    
    return confusion_matrix.astype(float) / row_sums[:, np.newaxis]


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_labels: List[str], 
                         title: str = "Confusion Matrix", save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 10)):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        confusion_matrix: Confusion matrix to plot
        class_labels: List of class labels
        title: Title for the plot
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(confusion_matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title(title)
    plt.xlabel('Predicted Species')
    plt.ylabel('True Species')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {save_path}")
    
    plt.show()


def print_confusion_matrix_summary(confusion_matrix: np.ndarray, class_labels: List[str]):
    """
    Print a summary of the confusion matrix.
    
    Args:
        confusion_matrix: Normalized confusion matrix
        class_labels: List of class labels
    """
    print("\n" + "="*80)
    print("CONFUSION MATRIX SUMMARY")
    print("="*80)
    
    n_classes = len(class_labels)
    
    # Print matrix
    print(f"\nNormalized Confusion Matrix ({n_classes}x{n_classes}):")
    print("-" * 60)
    
    # Header
    header = f"{'True\\Pred':<12}"
    for label in class_labels:
        header += f"{label:<8}"
    print(header)
    print("-" * 60)
    
    # Rows
    for i, true_label in enumerate(class_labels):
        row = f"{true_label:<12}"
        for j in range(n_classes):
            row += f"{confusion_matrix[i, j]:<8.3f}"
        print(row)
    
    # Summary statistics
    print("\n" + "-" * 60)
    print("SUMMARY STATISTICS:")
    print("-" * 60)
    
    # Per-class precision and recall
    for i, species in enumerate(class_labels):
        if species != 'background':  # Skip the background class
            # Precision = TP / (TP + FP)
            # FP = all predictions of this species - TP
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Recall = TP / (TP + FN)
            # FN = all true instances of this species - TP
            fn = confusion_matrix[i, :].sum() - tp
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"{species:<12}: Precision={precision:.3f}, Recall={recall:.3f}")
    
    # Background class metrics
    if 'background' in class_labels:
        bg_idx = class_labels.index('background')
        bg_tp = confusion_matrix[bg_idx, bg_idx]
        bg_fp = confusion_matrix[:, bg_idx].sum() - bg_tp  # All predictions as background - TP
        bg_fn = confusion_matrix[bg_idx, :].sum() - bg_tp  # All true background - TP
        
        bg_precision = bg_tp / (bg_tp + bg_fp) if (bg_tp + bg_fp) > 0 else 0
        bg_recall = bg_tp / (bg_tp + bg_fn) if (bg_tp + bg_fn) > 0 else 0
        
        print(f"{'background':<12}: Precision={bg_precision:.3f}, Recall={bg_recall:.3f}")
    
    # Overall accuracy
    total_correct = np.trace(confusion_matrix)
    total_samples = np.sum(confusion_matrix)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.3f}")
