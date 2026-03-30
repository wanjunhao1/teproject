import re
import matplotlib.pyplot as plt
import os

def parse_log_file(log_path):
    steps = []
    mlu = []
    deadline_pen = []
    coflow_cct = []
    
    with open(log_path, 'r') as f:
        for line in f:
            if "update=" in line and "steps=" in line:
                # Extract step
                step_match = re.search(r"steps=\s*(\d+)", line)
                mlu_match = re.search(r"mlu=([0-9.]+)", line)
                pen_match = re.search(r"deadline_pen=([0-9.]+)", line)
                cct_match = re.search(r"coflow_cct=([0-9.]+)", line)
                
                if step_match and mlu_match and pen_match and cct_match:
                    steps.append(int(step_match.group(1)))
                    mlu.append(float(mlu_match.group(1)))
                    deadline_pen.append(float(pen_match.group(1)))
                    coflow_cct.append(float(cct_match.group(1)))
                    
    return steps, mlu, deadline_pen, coflow_cct

def plot_metrics(log_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    steps, mlu, pen, cct = parse_log_file(log_path)
    
    if not steps:
        print("No training data found in log.")
        return

    # Plot MLU
    plt.figure(figsize=(10, 5))
    plt.plot(steps, mlu, label='MLU', color='blue')
    plt.title('MLU over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Maximum Link Utilization')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'MLU_over_time.png'))
    plt.close()

    # Plot Deadline Penalty
    plt.figure(figsize=(10, 5))
    plt.plot(steps, pen, label='Deadline Penalty', color='red')
    plt.title('Deadline Penalty over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Penalty')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'Deadline_Penalty_over_time.png'))
    plt.close()

    # Plot Coflow CCT
    plt.figure(figsize=(10, 5))
    plt.plot(steps, cct, label='Coflow CCT', color='green')
    plt.title('Average Coflow CCT over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('CCT')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'Coflow_CCT_over_time.png'))
    plt.close()
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    plot_metrics("training.log", "plots")
