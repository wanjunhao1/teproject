import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def smooth_data(data, window_size=5):
    """Smooth data using moving average"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def generate_dashboard(log_path='training.log', output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    steps = []
    mlu = []
    deadline_pen = []
    coflow_cct = []
    adm_dl = []
    adm_cf = []
    adm_bk = []
    
    ep_miss_rates = []
    
    # Parse log file
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Parse metrics
            match = re.search(r'update=\s*(\d+)\s+steps=\s*(\d+)\s+mlu=(\d+\.\d+)\s+deadline_pen=(\d+\.\d+)\s+coflow_cct=(\d+\.\d+)\s+adm_dl=(\d+\.\d+)\s+adm_cf=(\d+\.\d+)\s+adm_bk=(\d+\.\d+)', line)
            if match:
                steps.append(int(match.group(2)))
                mlu.append(float(match.group(3)))
                deadline_pen.append(float(match.group(4)))
                coflow_cct.append(float(match.group(5)))
                adm_dl.append(float(match.group(6)))
                adm_cf.append(float(match.group(7)))
                adm_bk.append(float(match.group(8)))
            
            # Parse miss rate
            miss_match = re.search(r'Episode deadline miss rate: (\d+\.\d+)', line)
            if miss_match:
                ep_miss_rates.append(float(miss_match.group(1)))
                
    if not steps:
        print("No matched metrics found in log! Make sure the format is correct.")
        return
        
    print(f"Extracted {len(steps)} data points from {log_path}.")
    
    # Enable matplotlib style
    plt.style.use('ggplot')
    
    # 1. Plot MLU and Deadline Penalty
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    ax1.plot(steps[:len(smooth_data(mlu))], smooth_data(mlu), 'g-', label='MLU (Left Axis)', linewidth=2)
    ax2.plot(steps[:len(smooth_data(deadline_pen))], smooth_data(deadline_pen), 'r-', label='Deadline Penalty (Right Axis)', linewidth=2)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Max Link Utilization (MLU)', color='g')
    ax2.set_ylabel('Deadline Penalty Score', color='r')
    plt.title('Network Congestion vs Delay Penalty Evolution')
    
    # Merge legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    plt.savefig(os.path.join(output_dir, '1_Network_Health.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Plot Admission Ratios (The "Three-Way Ticket" Behavior)
    plt.figure(figsize=(10, 5))
    plt.plot(steps[:len(smooth_data(adm_dl))], smooth_data(adm_dl), label='Deadline Flows', color='red', linewidth=2.5)
    plt.plot(steps[:len(smooth_data(adm_cf))], smooth_data(adm_cf), label='Coflows', color='orange', linewidth=2)
    plt.plot(steps[:len(smooth_data(adm_bk))], smooth_data(adm_bk), label='Bulk Flows', color='blue', linewidth=2)
    plt.title('Agent Admission Control Strategy over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Acceptance Ratio')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig(os.path.join(output_dir, '2_Admission_Strategy.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Plot Deadline Miss Rate over Episodes
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(ep_miss_rates)), ep_miss_rates, alpha=0.3, color='purple', label='Raw Episode Miss Rate')
    if len(ep_miss_rates) > 10:
        plt.plot(range(len(smooth_data(ep_miss_rates, 10))), smooth_data(ep_miss_rates, 10), color='purple', linewidth=2, label='Smoothed Trend (Window=10)')
    plt.title('Deadline Miss Rate across Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Miss Rate (%)')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig(os.path.join(output_dir, '3_Deadline_Miss_Rate.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print("Charts generated successfully in the 'plots' folder!")

if __name__ == "__main__":
    generate_dashboard()
