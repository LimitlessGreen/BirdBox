#!/usr/bin/env python3
"""
Evaluate bird call detection results against ground truth labels.

This script computes confusion matrices and F-beta scores for bird call detection
evaluation, comparing detection results against ground truth labels.

Usage:
    python inference/evaluate_detections.py --labels labels.csv --detections detections.csv --output results
    python inference/evaluate_detections.py --labels labels.csv --detections detections.csv --output results --beta 2.0
    python inference/evaluate_detections.py --labels labels.csv --detections detections.csv --output results --iou-threshold 0.3
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import evaluation utilities
from inference.utils.confusion_matrix import (
    load_detections_csv, load_labels_csv, compute_confusion_matrix,
    normalize_confusion_matrix, plot_confusion_matrix, print_confusion_matrix_summary
)
from inference.utils.f_beta_score import (
    compute_f_beta_scores, compute_macro_f_beta_score, compute_weighted_f_beta_score,
    compute_micro_f_beta_score, print_f_beta_summary, save_f_beta_results
)


class DetectionEvaluator:
    """
    Evaluator for bird call detection results.
    
    This class handles the complete evaluation pipeline from loading data
    to computing and saving evaluation metrics.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize the detection evaluator.
        
        Args:
            iou_threshold: IoU threshold for considering detections as matches
        """
        self.iou_threshold = iou_threshold
        
        print(f"Initialized evaluator with IoU threshold: {iou_threshold}")
        print("Using 2D IoU (time-frequency)")
    
    def load_data(self, labels_path: str, detections_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load ground truth labels and detection results.
        
        Args:
            labels_path: Path to the labels CSV file
            detections_path: Path to the detections CSV file
            
        Returns:
            Tuple of (labels_df, detections_df)
        """
        print(f"\nLoading ground truth labels from: {labels_path}")
        labels_df = load_labels_csv(labels_path)
        print(f"Loaded {len(labels_df)} ground truth labels")
        
        print(f"\nLoading detection results from: {detections_path}")
        detections_df = load_detections_csv(detections_path)
        print(f"Loaded {len(detections_df)} detections")
        
        # Print data summary
        print(f"\nData Summary:")
        print(f"  Ground truth files: {labels_df['Filename'].nunique()}")
        print(f"  Detection files: {detections_df['Filename'].nunique()}")
        print(f"  Ground truth species: {sorted(labels_df['Species eBird Code'].unique())}")
        print(f"  Detection species: {sorted(detections_df['Species eBird Code'].unique())}")
        
        return labels_df, detections_df
    
    def compute_metrics(self, labels_df: pd.DataFrame, detections_df: pd.DataFrame) -> Dict:
        """
        Compute evaluation metrics.
        
        Args:
            labels_df: Ground truth labels DataFrame
            detections_df: Detection results DataFrame
            
        Returns:
            Dictionary containing all computed metrics
        """
        print(f"\nComputing confusion matrix...")
        confusion_matrix, class_labels = compute_confusion_matrix(
            detections_df, labels_df, 
            iou_threshold=self.iou_threshold
        )
        
        print(f"Computed confusion matrix: {confusion_matrix.shape}")
        print(f"Classes: {class_labels}")
        
        # Normalize confusion matrix
        normalized_matrix = normalize_confusion_matrix(confusion_matrix)
        
        return {
            'confusion_matrix': confusion_matrix,
            'normalized_confusion_matrix': normalized_matrix,
            'class_labels': class_labels
        }
    
    def compute_f_beta_scores(self, metrics: Dict, beta: float) -> Dict:
        """
        Compute F-beta scores from the confusion matrix.
        
        Args:
            metrics: Dictionary containing confusion matrix and class labels
            beta: Beta parameter for F-beta score
            
        Returns:
            Dictionary containing F-beta scores
        """
        print(f"\nComputing F-{beta} scores...")
        
        confusion_matrix = metrics['normalized_confusion_matrix']
        class_labels = metrics['class_labels']
        
        # Compute per-class F-beta scores
        f_beta_scores = compute_f_beta_scores(confusion_matrix, class_labels, beta)
        
        # Compute overall F-beta scores
        macro_f_beta = compute_macro_f_beta_score(f_beta_scores, beta)
        weighted_f_beta = compute_weighted_f_beta_score(confusion_matrix, class_labels, beta)
        micro_f_beta = compute_micro_f_beta_score(confusion_matrix, class_labels, beta)
        
        return {
            'f_beta_scores': f_beta_scores,
            'macro_f_beta': macro_f_beta,
            'weighted_f_beta': weighted_f_beta,
            'micro_f_beta': micro_f_beta,
            'beta': beta
        }
    
    def save_results(self, metrics: Dict, f_beta_results: Dict, output_path: str):
        """
        Save evaluation results to files.
        
        Args:
            metrics: Dictionary containing confusion matrix metrics
            f_beta_results: Dictionary containing F-beta scores
            output_path: Base path for output files
        """
        output_path_obj = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_dir = output_path_obj.parent / f"{output_path_obj.stem}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save confusion matrix
        confusion_matrix_path = str(output_dir / "confusion_matrix.npy")
        np.save(confusion_matrix_path, metrics['confusion_matrix'])
        print(f"\nSaved confusion matrix to: {confusion_matrix_path}")
        
        # Save normalized confusion matrix
        normalized_matrix_path = str(output_dir / "normalized_confusion_matrix.npy")
        np.save(normalized_matrix_path, metrics['normalized_confusion_matrix'])
        print(f"Saved normalized confusion matrix to: {normalized_matrix_path}")
        
        # Save class labels
        class_labels_path = str(output_dir / "class_labels.json")
        with open(class_labels_path, 'w') as f:
            json.dump(metrics['class_labels'], f, indent=2)
        print(f"Saved class labels to: {class_labels_path}")
        
        # Save F-beta results
        f_beta_path = str(output_dir / f"f_{f_beta_results['beta']}_scores.csv")
        save_f_beta_results(
            f_beta_results['f_beta_scores'],
            f_beta_results['macro_f_beta'],
            f_beta_results['weighted_f_beta'],
            f_beta_results['micro_f_beta'],
            f_beta_results['beta'],
            f_beta_path
        )
        
        # Save complete evaluation summary
        summary_path = str(output_dir / "evaluation_summary.json")
        summary = {
            'evaluation_parameters': {
                'iou_threshold': self.iou_threshold,
                'beta': f_beta_results['beta']
            },
            'data_summary': {
                'n_classes': len(metrics['class_labels']),
                'class_labels': metrics['class_labels']
            },
            'overall_metrics': {
                'macro_f_beta': f_beta_results['macro_f_beta'],
                'weighted_f_beta': f_beta_results['weighted_f_beta'],
                'micro_f_beta': f_beta_results['micro_f_beta']
            },
            'per_class_metrics': f_beta_results['f_beta_scores']
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved evaluation summary to: {summary_path}")
    
    def plot_results(self, metrics: Dict, output_path: str):
        """
        Generate and save confusion matrix plots.
        
        Args:
            metrics: Dictionary containing confusion matrix metrics
            output_path: Base path for output files
        """
        output_path_obj = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_dir = output_path_obj.parent / f"{output_path_obj.stem}"
        
        # Plot normalized confusion matrix
        plot_path = str(output_dir / "confusion_matrix.png")
        plot_confusion_matrix(
            metrics['normalized_confusion_matrix'],
            metrics['class_labels'],
            title=f"Normalized Confusion Matrix (IoU ≥ {self.iou_threshold})",
            save_path=plot_path,
            figsize=(12, 10)
        )
    
    def print_summary(self, metrics: Dict, f_beta_results: Dict):
        """
        Print evaluation summary to console.
        
        Args:
            metrics: Dictionary containing confusion matrix metrics
            f_beta_results: Dictionary containing F-beta scores
        """
        # Print confusion matrix summary
        print_confusion_matrix_summary(
            metrics['normalized_confusion_matrix'],
            metrics['class_labels']
        )
        
        # Print F-beta summary
        print_f_beta_summary(
            f_beta_results['f_beta_scores'],
            f_beta_results['macro_f_beta'],
            f_beta_results['weighted_f_beta'],
            f_beta_results['micro_f_beta'],
            f_beta_results['beta']
        )
    
    def evaluate(self, labels_path: str, detections_path: str, output_path: str, 
                beta: float = 1.0, plot: bool = True) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Args:
            labels_path: Path to the labels CSV file
            detections_path: Path to the detections CSV file
            output_path: Base path for output files
            beta: Beta parameter for F-beta score
            plot: Whether to generate plots
            
        Returns:
            Dictionary containing all evaluation results
        """
        print("="*80)
        print("BIRD CALL DETECTION EVALUATION")
        print("="*80)
        
        # Load data
        labels_df, detections_df = self.load_data(labels_path, detections_path)
        
        # Compute metrics
        metrics = self.compute_metrics(labels_df, detections_df)
        
        # Compute F-beta scores
        f_beta_results = self.compute_f_beta_scores(metrics, beta)
        
        # Save results
        self.save_results(metrics, f_beta_results, output_path)
        
        # Generate plots
        if plot:
            self.plot_results(metrics, output_path)
        
        # Print summary
        self.print_summary(metrics, f_beta_results)
        
        return {
            'metrics': metrics,
            'f_beta_results': f_beta_results
        }


def ensure_output_directory(output_path: str) -> bool:
    """
    Ensure the output directory exists, creating it automatically if needed.
    
    Args:
        output_path: The output path (may be a file path)
        
    Returns:
        True if directory exists or was created successfully, False if creation failed
    """
    if not output_path:
        return True  # No output path specified, nothing to check
    
    output_dir = Path(output_path).parent
    
    # If the directory already exists, we're good
    if output_dir.exists():
        return True
    
    # Directory doesn't exist, create it automatically
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")
        return True
    except Exception as e:
        print(f"✗ Error creating directory: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bird call detection results against ground truth labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with F1-score
  python inference/evaluate_detections.py --labels labels.csv --detections detections.csv --output results
  
  # Evaluation with F2-score (emphasizes recall)
  python inference/evaluate_detections.py --labels labels.csv --detections detections.csv --output results --beta 2.0
  
  # Evaluation with custom IoU threshold (more lenient)
  python inference/evaluate_detections.py --labels labels.csv --detections detections.csv --output results --iou-threshold 0.3
  
  # Evaluation without plots
  python inference/evaluate_detections.py --labels labels.csv --detections detections.csv --output results --no-plot
  
        """
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Path to the ground truth labels CSV file'
    )
    
    parser.add_argument(
        '--detections',
        type=str,
        required=True,
        help='Path to the detection results CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Base path for output files'
    )
    
    parser.add_argument(
        '--beta',
        type=float,
        default=1.0,
        help='Beta parameter for F-beta score (default: 1.0 for F1-score)'
    )
    
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for considering detections as matches (default: 0.5)'
    )
    
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating confusion matrix plots'
    )
    
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.labels).exists():
        print(f"Error: Labels file not found: {args.labels}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.detections).exists():
        print(f"Error: Detections file not found: {args.detections}", file=sys.stderr)
        sys.exit(1)
    
    # Ensure output directory exists
    if not ensure_output_directory(args.output):
        sys.exit(1)
    
    # Create evaluator
    evaluator = DetectionEvaluator(
        iou_threshold=args.iou_threshold
    )
    
    # Run evaluation
    try:
        results = evaluator.evaluate(
            labels_path=args.labels,
            detections_path=args.detections,
            output_path=args.output,
            beta=args.beta,
            plot=not args.no_plot
        )
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
