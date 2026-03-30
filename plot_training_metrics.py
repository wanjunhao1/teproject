#!/usr/bin/env python3
"""
Training Metrics Visualization Script

This script reads TensorBoard logs and generates visualizations for:
1. MLU (Maximum Link Utilization) over time
2. Deadline miss rate over time
3. Coflow CCT (Completion Time) comparison across strategies

Usage:
    python plot_training_metrics.py [--logdir LOGDIR] [--output OUTPUT_DIR]

Options:
    --logdir LOGDIR       Directory containing TensorBoard logs (default: ./runs)
    --output OUTPUT_DIR   Directory to save plots (default: ./plots)
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Training Metrics Visualization Script')
    parser.add_argument('--logdir', type=str, default='./runs',
                        help='Directory containing TensorBoard logs')
    parser.add_argument('--output', type=str, default='./plots',
                        help='Directory to save plots')
    return parser.parse_args()


def load_tensorboard_data(logdir):
    """Load TensorBoard log data"""
    runs = []
    
    # Find all run directories
    if os.path.exists(logdir):
        for run_name in os.listdir(logdir):
            run_path = os.path.join(logdir, run_name)
            if os.path.isdir(run_path):
                event_acc = EventAccumulator(run_path)
                event_acc.Reload()
                
                # Check if this run has the required tags
                tags = event_acc.Tags()['scalars']
                if any('MLU' in tag or 'Deadline' in tag or 'CCT' in tag for tag in tags):
                    runs.append((run_name, event_acc))
    
    return runs


def extract_scalar_data(event_acc, tag):
    """Extract scalar data from TensorBoard event accumulator"""
    try:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except Exception:
        return [], []


def plot_mlu_over_time(runs, output_dir):
    """Plot MLU over time"""
    plt.figure(figsize=(12, 6))
    
    for run_name, event_acc in runs:
        steps, values = extract_scalar_data(event_acc, 'Metrics/MLU')
        if steps and values:
            plt.plot(steps, values, label=run_name, alpha=0.7)
    
    plt.title('Maximum Link Utilization (MLU) Over Training Steps', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('MLU', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'mlu_over_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved MLU plot to: {output_path}")
    plt.close()


def plot_deadline_miss_rate(runs, output_dir):
    """Plot Deadline Miss Rate over time"""
    plt.figure(figsize=(12, 6))
    
    for run_name, event_acc in runs:
        steps, values = extract_scalar_data(event_acc, 'Episode/Deadline_Miss_Rate')
        if steps and values:
            plt.plot(steps, values, label=run_name, alpha=0.7)
    
    plt.title('Deadline Miss Rate Over Training Steps', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Deadline Miss Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'deadline_miss_rate.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Deadline Miss Rate plot to: {output_path}")
    plt.close()


def plot_coflow_cct_comparison(runs, output_dir):
    """Plot Coflow CCT comparison across strategies"""
    plt.figure(figsize=(12, 6))
    
    for run_name, event_acc in runs:
        steps, values = extract_scalar_data(event_acc, 'Episode/Avg_Coflow_CCT')
        if steps and values:
            plt.plot(steps, values, label=run_name, alpha=0.7)
    
    plt.title('Average Coflow Completion Time (CCT) Over Training Steps', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Average CCT (time slots)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'coflow_cct_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Coflow CCT comparison plot to: {output_path}")
    plt.close()


def plot_change_cost(runs, output_dir):
    """Plot Change Cost over time"""
    plt.figure(figsize=(12, 6))
    
    for run_name, event_acc in runs:
        steps, values = extract_scalar_data(event_acc, 'Metrics/Change_Cost')
        if steps and values:
            plt.plot(steps, values, label=run_name, alpha=0.7)
    
    plt.title('Change Cost Over Training Steps', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Change Cost', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'change_cost.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Change Cost plot to: {output_path}")
    plt.close()


def plot_metrics_summary(runs, output_dir):
    """Plot summary of all metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # MLU
    for run_name, event_acc in runs:
        steps, values = extract_scalar_data(event_acc, 'Metrics/MLU')
        if steps and values:
            axes[0, 0].plot(steps, values, label=run_name, alpha=0.7)
    axes[0, 0].set_title('Maximum Link Utilization (MLU)')
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('MLU')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Deadline Miss Rate
    for run_name, event_acc in runs:
        steps, values = extract_scalar_data(event_acc, 'Episode/Deadline_Miss_Rate')
        if steps and values:
            axes[0, 1].plot(steps, values, label=run_name, alpha=0.7)
    axes[0, 1].set_title('Deadline Miss Rate')
    axes[0, 1].set_xlabel('Training Steps')
    axes[0, 1].set_ylabel('Deadline Miss Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Coflow CCT
    for run_name, event_acc in runs:
        steps, values = extract_scalar_data(event_acc, 'Episode/Avg_Coflow_CCT')
        if steps and values:
            axes[1, 0].plot(steps, values, label=run_name, alpha=0.7)
    axes[1, 0].set_title('Average Coflow CCT')
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_ylabel('CCT (time slots)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Change Cost
    for run_name, event_acc in runs:
        steps, values = extract_scalar_data(event_acc, 'Metrics/Change_Cost')
        if steps and values:
            axes[1, 1].plot(steps, values, label=run_name, alpha=0.7)
    axes[1, 1].set_title('Change Cost')
    axes[1, 1].set_xlabel('Training Steps')
    axes[1, 1].set_ylabel('Change Cost')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.01), ncol=min(3, len(runs)))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    output_path = os.path.join(output_dir, 'metrics_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics summary plot to: {output_path}")
    plt.close()


def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load TensorBoard data
    runs = load_tensorboard_data(args.logdir)
    
    if not runs:
        print(f"No valid TensorBoard logs found in {args.logdir}")
        print("Please run the training first to generate logs")
        return
    
    print(f"Found {len(runs)} runs with valid metrics")
    for run_name, _ in runs:
        print(f"  - {run_name}")
    
    # Generate plots
    plot_mlu_over_time(runs, args.output)
    plot_deadline_miss_rate(runs, args.output)
    plot_coflow_cct_comparison(runs, args.output)
    plot_change_cost(runs, args.output)
    plot_metrics_summary(runs, args.output)
    
    print(f"\nAll plots saved to: {args.output}")


if __name__ == '__main__':
    main()
