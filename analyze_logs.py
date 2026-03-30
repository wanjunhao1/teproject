import re
import os

def analyze_logs(logfile, prefix):
    if not os.path.exists(logfile):
        print(f"File {logfile} not found.")
        return
        
    metrics = {'mlu':[], 'deadline_pen':[], 'coflow_cct':[], 'adm_dl':[], 'adm_cf':[], 'adm_bk':[]}
    with open(logfile) as f:
        for l in f:
            if 'update=' in l:
                for k in metrics:
                    m = re.search(f"{k}=([0-9.]+)", l)
                    if m:
                        metrics[k].append(float(m.group(1)))
                        
    print(f"=== {prefix} Graph Analysis ===")
    for k, v in metrics.items():
        if v:
            avg_first_10 = sum(v[:10])/abs(len(v[:10])+1e-9)
            avg_last_10 = sum(v[-10:])/abs(len(v[-10:])+1e-9)
            print(f'{k:<12}: Start={v[0]:.3f} | End={v[-1]:.3f} | Min={min(v):.3f} | Max={max(v):.3f} | First10Avg={avg_first_10:.3f} -> Last10Avg={avg_last_10:.3f}')
    print()

analyze_logs('training.log', "PPO Agent Training")
analyze_logs('spf_training.log', "SPF (Shortest Path First) Baseline")
