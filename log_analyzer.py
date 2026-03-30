#!/usr/bin/env python3
"""
Log Analyzer and Visualization Script

This script analyzes training logs and generates visualizations for:
1. MLU (Maximum Link Utilization) over time
2. Deadline miss rate over time
3. Coflow CCT over time
4. Change cost over time
5. Comparison between PPO and SPF strategies

Usage:
    python log_analyzer.py [--ppo-log PPO_LOG] [--spf-log SPF_LOG] [--output OUTPUT_DIR]

Options:
    --ppo-log PPO_LOG     Path to PPO training log file (default: training.log)
    --spf-log SPF_LOG     Path to SPF training log file (default: spf_training.log)
    --output OUTPUT_DIR   Directory to save plots (default: ./plots)
"""

import os
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Log Analyzer and Visualization Script')
    parser.add_argument('--ppo-log', type=str, default='training.log',
                        help='Path to PPO training log file')
    parser.add_argument('--spf-log', type=str, default='spf_training.log',
                        help='Path to SPF training log file')
    parser.add_argument('--output', type=str, default='./plots',
                        help='Directory to save plots')
    return parser.parse_args()


def parse_log_file(log_path):
    """Parse training log file and extract metrics"""
    if not os.path.exists(log_path):
        print(f"Warning: Log file {log_path} not found")
        return None
    
    steps = []
    mlu_values = []
    deadline_penalty_values = []
    coflow_cct_values = []
    # New admission ratios
    adm_dl_values = []
    adm_cf_values = []
    adm_bk_values = []
    deadline_miss_rates = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse update lines
            update_match = re.search(r'update=\s*(\d+)\s+steps=\s*(\d+)\s+mlu=(\d+\.\d+)\s+deadline_pen=(\d+\.\d+)\s+coflow_cct=(\d+\.\d+)\s+adm_dl=(\d+\.\d+)\s+adm_cf=(\d+\.\d+)\s+adm_bk=(\d+\.\d+)', line)
            
            # Fallback for old logs (if they didn't have the 3 adm ratios)
            if not update_match:
                update_match_old = re.search(r'update=\s*(\d+)\s+steps=\s*(\d+)\s+mlu=(\d+\.\d+)\s+deadline_pen=(\d+\.\d+)\s+coflow_cct=(\d+\.\d+)\s+change=(\d+\.\d+)', line)
                if update_match_old:
                    step = int(update_match_old.group(2))
                    mlu = float(update_match_old.group(3))
                    deadline_penalty = float(update_match_old.group(4))
                    coflow_cct = float(update_match_old.group(5))
                    
                    steps.append(step)
                    mlu_values.append(mlu)
                    deadline_penalty_values.append(deadline_penalty)
                    coflow_cct_values.append(coflow_cct)
                    adm_dl_values.append(0.0)
                    adm_cf_values.append(0.0)
                    adm_bk_values.append(0.0)
            else:
                step = int(update_match.group(2))
                mlu = float(update_match.group(3))
                deadline_penalty = float(update_match.group(4))
                coflow_cct = float(update_match.group(5))
                adm_dl = float(update_match.group(6))
                adm_cf = float(update_match.group(7))
                adm_bk = float(update_match.group(8))
                
                steps.append(step)
                mlu_values.append(mlu)
                deadline_penalty_values.append(deadline_penalty)
                coflow_cct_values.append(coflow_cct)
                adm_dl_values.append(adm_dl)
                adm_cf_values.append(adm_cf)
                adm_bk_values.append(adm_bk)
            
            # Parse episode deadline miss rate
            miss_rate_match = re.search(r'Episode deadline miss rate: (\d+\.\d+)', line)
            if miss_rate_match:
                miss_rate = float(miss_rate_match.group(1))
                deadline_miss_rates.append(miss_rate)
    
    if not steps:
        print(f"Warning: No metrics found in log file {log_path}")
        return None
    
    return {
        'steps': steps,
        'mlu': mlu_values,
        'deadline_penalty': deadline_penalty_values,
        'coflow_cct': coflow_cct_values,
        'adm_dl': adm_dl_values,
        'adm_cf': adm_cf_values,
        'adm_bk': adm_bk_values,
        'deadline_miss_rates': deadline_miss_rates
    }


def smooth_data(data, window_size=10):
    """Smooth data using moving average"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_metric_comparison(ppo_data, spf_data, metric_name, metric_label, output_dir):
    """Plot comparison of a metric between PPO and SPF"""
    plt.figure(figsize=(12, 6))
    
    if ppo_data:
        ppo_steps = ppo_data['steps']
        ppo_values = ppo_data[metric_name]
        ppo_smooth = smooth_data(ppo_values)
        ppo_smooth_steps = ppo_steps[:len(ppo_smooth)]
        plt.plot(ppo_smooth_steps, ppo_smooth, label='PPO', alpha=0.7)
    
    if spf_data:
        spf_steps = spf_data['steps']
        spf_values = spf_data[metric_name]
        spf_smooth = smooth_data(spf_values)
        spf_smooth_steps = spf_steps[:len(spf_smooth)]
        plt.plot(spf_smooth_steps, spf_smooth, label='SPF', alpha=0.7)
    
    plt.title(f'{metric_label} Comparison: PPO vs SPF', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{metric_name}_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {metric_name} comparison plot to: {output_path}")
    plt.close()


def plot_deadline_miss_rate(ppo_data, spf_data, output_dir):
    """Plot deadline miss rate comparison"""
    plt.figure(figsize=(12, 6))
    
    if ppo_data and ppo_data['deadline_miss_rates']:
        ppo_rates = ppo_data['deadline_miss_rates']
        ppo_smooth = smooth_data(ppo_rates)
        ppo_steps = range(len(ppo_smooth))
        plt.plot(ppo_steps, ppo_smooth, label='PPO', alpha=0.7)
    
    if spf_data and spf_data['deadline_miss_rates']:
        spf_rates = spf_data['deadline_miss_rates']
        spf_smooth = smooth_data(spf_rates)
        spf_steps = range(len(spf_smooth))
        plt.plot(spf_steps, spf_smooth, label='SPF', alpha=0.7)
    
    plt.title('Deadline Miss Rate Comparison: PPO vs SPF', fontsize=14)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Deadline Miss Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'deadline_miss_rate_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved deadline miss rate comparison plot to: {output_path}")
    plt.close()


def plot_all_metrics(ppo_data, spf_data, output_dir):
    """Plot all metrics in a single figure"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = [
        ('mlu', 'Maximum Link Utilization'),
        ('deadline_penalty', 'Deadline Penalty'),
        ('coflow_cct', 'Coflow CCT'),
        ('change_cost', 'Change Cost')
    ]
    
    for i, (metric_name, metric_label) in enumerate(metrics):
        row = i // 2
        col = i % 2
        
        if ppo_data:
            ppo_steps = ppo_data['steps']
            ppo_values = ppo_data[metric_name]
            ppo_smooth = smooth_data(ppo_values)
            ppo_smooth_steps = ppo_steps[:len(ppo_smooth)]
            axes[row, col].plot(ppo_smooth_steps, ppo_smooth, label='PPO', alpha=0.7)
        
        if spf_data:
            spf_steps = spf_data['steps']
            spf_values = spf_data[metric_name]
            spf_smooth = smooth_data(spf_values)
            spf_smooth_steps = spf_steps[:len(spf_smooth)]
            axes[row, col].plot(spf_smooth_steps, spf_smooth, label='SPF', alpha=0.7)
        
        axes[row, col].set_title(metric_label)
        axes[row, col].set_xlabel('Training Steps')
        axes[row, col].set_ylabel(metric_label)
        axes[row, col].grid(True, alpha=0.3)
    
    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.01), ncol=2)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    output_path = os.path.join(output_dir, 'all_metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved all metrics comparison plot to: {output_path}")
    plt.close()


def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Parse log files
    ppo_data = parse_log_file(args.ppo_log)
    spf_data = parse_log_file(args.spf_log)
    
    if not ppo_data and not spf_data:
        print("Error: No valid log files found")
        return
    
    # Generate comparison plots
    plot_metric_comparison(ppo_data, spf_data, 'mlu', 'Maximum Link Utilization', args.output)
    plot_metric_comparison(ppo_data, spf_data, 'deadline_penalty', 'Deadline Penalty', args.output)
    plot_metric_comparison(ppo_data, spf_data, 'coflow_cct', 'Coflow CCT', args.output)
    plot_metric_comparison(ppo_data, spf_data, 'change_cost', 'Change Cost', args.output)
    plot_deadline_miss_rate(ppo_data, spf_data, args.output)
    plot_all_metrics(ppo_data, spf_data, args.output)
    
    print(f"\nAll plots saved to: {args.output}")


if __name__ == '__main__':
    main()
