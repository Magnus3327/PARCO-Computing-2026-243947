import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    # Verify command line arguments
    if len(sys.argv) < 3:
        print("Usage: python plot_weak_scaling.py <json_path> <output_directory>")
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
        # Filter only for "Generated" matrices and skip entries with errors
        if r['matrix']['name'] == "Generated" and not r.get('errors'):
            rows.append({
                'p': r['mpi']['processes'],
                't_kernel': r['statistics']['kernel_p90_ms'],
                't_comm': r['timings_ms']['communication'],
                't_total': r['statistics']['kernel_p90_ms'] + r['timings_ms']['communication']
            })

    if not rows:
        print("No valid 'Generated' matrix results found in the JSON file.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values('p')

    plt.figure(figsize=(10, 7))

    p_vals = df['p'].values
    t_total = df['t_total'].values
    t_kernel = df['t_kernel'].values

    # Baseline for Ideal Scaling (Slope = 0)
    # We take the total time at the smallest number of processes (usually p=1)
    t1_total = df.iloc[0]['t_total']

    # 1. Total Time: Solid line with circles
    plt.plot(p_vals, t_total, marker='o', linestyle='-', color='tab:blue', 
             label='Total (Communication + Kernel)', linewidth=2)
    
    # 2. Kernel Time: Dashed line with squares
    plt.plot(p_vals, t_kernel, marker='s', linestyle='--', color='tab:orange', 
             label='Kernel Time (Computation Only)', alpha=0.8)

    # 3. Ideal Scaling (Horizontal Line, Slope = 0)
    plt.axhline(y=t1_total, color='grey', linestyle=':', linewidth=2, 
                label=f'Ideal Weak Scaling')

    # X-axis configuration (log scale for cores)
    plt.xscale('log', base=2)
    plt.xticks(p_vals, labels=[str(p) for p in p_vals])
    plt.minorticks_off()

    # Y-axis (using log scale due to the significant overhead at high core counts)
    plt.yscale('log')

    plt.xlabel('Number of Processes (p)')
    plt.ylabel('Execution Time (ms)')
    plt.title('Weak Scaling')
    plt.grid(True, which="both", ls="-", alpha=0.3)

    # Legend
    plt.legend(loc='upper left', fontsize='medium')
    plt.tight_layout()

    # Save as PDF
    output_path = os.path.join(output_dir, 'weak_scaling_analysis.pdf')
    plt.savefig(output_path)
    print(f"Weak Scaling graph successfully saved to: {output_path}")

if __name__ == "__main__":
    main()