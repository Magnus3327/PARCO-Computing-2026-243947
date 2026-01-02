import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    # Verify command line arguments
    if len(sys.argv) < 3:
        print("Usage: python plot_weak_metrics.py <json_path> <output_directory>")
        sys.exit(1)

    json_path = sys.argv[1]
    output_dir = sys.argv[2]

    # Check if the JSON file exists
    if not os.path.exists(json_path):
        print(f"Error: The file {json_path} does not exist.")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    results = data.get('results', [])
    rows = []
    for r in results:
        # Filter: Only "Generated" matrices for Weak Scaling
        if r['matrix']['name'] == "Generated" and not r.get('errors'):
            rows.append({
                'p': r['mpi']['processes'],
                't_kernel': r['statistics']['kernel_p90_ms'],
                't_total': r['statistics']['kernel_p90_ms'] + r['timings_ms']['communication']
            })

    if not rows:
        print("No valid 'Generated' matrix results found in the JSON file.")
        sys.exit(1)

    # Sort by number of processes
    df = pd.DataFrame(rows).sort_values('p')
    p_vals = df['p'].values
    
    # Baseline T1 (Performance at the smallest p, usually 1)
    t1_total = df.iloc[0]['t_total']
    t1_kernel = df.iloc[0]['t_kernel']

    # --- Metrics Calculation ---
    # Weak Efficiency: E = T1 / Tp
    eff_total = t1_total / df['t_total'].values
    eff_kernel = t1_kernel / df['t_kernel'].values

    # Scaled Speedup (Gustafson): S = p * (T1 / Tp)
    s_total = p_vals * eff_total
    s_kernel = p_vals * eff_kernel

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 1. SCALED SPEEDUP PLOT
    ax1.set_title("Weak Scaling: Scaled Speedup (Gustafson's Law)", fontsize=14)
    ax1.set_xlabel('Number of Processes (p)', fontsize=12)
    ax1.set_ylabel('Scaled Speedup ($p \cdot T_1 / T_p$)', fontsize=12)
    
    ax1.plot(p_vals, s_total, marker='o', linestyle='-', color='tab:blue', label='Total Scaled Speedup', linewidth=2)
    ax1.plot(p_vals, s_kernel, marker='s', linestyle='--', color='tab:orange', label='Kernel Scaled Speedup', alpha=0.7)
    
    # Ideal line for Speedup: S = p
    ax1.plot(p_vals, p_vals, color='grey', linestyle=':', linewidth=2, label='Ideal Scaled Speedup (S=p)')

    # 2. WEAK EFFICIENCY PLOT
    ax2.set_title('Weak Scaling: Efficiency', fontsize=14)
    ax2.set_xlabel('Number of Processes (p)', fontsize=12)
    ax2.set_ylabel('Efficiency ($T_1 / T_p$)', fontsize=12)
    
    ax2.plot(p_vals, eff_total, marker='o', linestyle='-', color='tab:blue', label='Total Efficiency', linewidth=2)
    ax2.plot(p_vals, eff_kernel, marker='s', linestyle='--', color='tab:orange', label='Kernel Efficiency', alpha=0.7)
    
    # Ideal line for Efficiency: E = 1.0
    ax2.axhline(y=1.0, color='grey', linestyle=':', linewidth=2, label='Ideal Efficiency (E=1.0)')

    # Formatting
    for ax in [ax1, ax2]:
        ax.set_xscale('log', base=2)
        ax.set_xticks(p_vals)
        ax.set_xticklabels([str(p) for p in p_vals])
        ax.minorticks_off()
        ax.grid(True, which="both", ls="-", alpha=0.3)
        ax.legend(fontsize='medium')

    ax2.set_ylim(0, 1.1)

    plt.tight_layout()

    # Save as PDF
    output_path = os.path.join(output_dir, 'weak_scaling_metrics.pdf')
    plt.savefig(output_path)
    print(f"Weak scaling speedup and efficiency saved to: {output_path}")

if __name__ == "__main__":
    main()