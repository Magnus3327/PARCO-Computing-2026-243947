import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    # Verify command line arguments
    if len(sys.argv) < 3:
        print("Usage: python plot_strong_metrics.py <json_path> <output_directory>")
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
        # Filter: Exclude "Generated" matrices and skip errors
        if r['matrix']['name'] != "Generated" and not r.get('errors'):
            rows.append({
                'matrix': r['matrix']['name'].split('/')[-1],
                'p': r['mpi']['processes'],
                't_kernel': r['statistics']['kernel_p90_ms'],
                't_total': r['statistics']['kernel_p90_ms'] + r['timings_ms']['communication']
            })

    if not rows:
        print("No valid real-matrix results found in the JSON file.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    matrices = sorted(df['matrix'].unique())
    core_counts = sorted(df['p'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(matrices)))

    # Create a figure with two subplots (Side-by-Side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- 1. SPEEDUP PLOT ---
    ax1.set_title('Strong Scaling: Speedup ($S = T_1 / T_p$)', fontsize=14)
    ax1.set_xlabel('Number of Processes (p)', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)

    # --- 2. EFFICIENCY PLOT ---
    ax2.set_title('Strong Scaling: Efficiency ($E = S / p$)', fontsize=14)
    ax2.set_xlabel('Number of Processes (p)', fontsize=12)
    ax2.set_ylabel('Efficiency', fontsize=12)

    for i, m_name in enumerate(matrices):
        subset = df[df['matrix'] == m_name].sort_values('p')
        color = colors[i]
        p = subset['p'].values
        
        # Baseline T1 values (Performance at 1 process)
        t1_total = subset[subset['p'] == 1]['t_total'].values[0]
        t1_kernel = subset[subset['p'] == 1]['t_kernel'].values[0]
        
        # Calculate Metrics
        s_total = t1_total / subset['t_total'].values
        s_kernel = t1_kernel / subset['t_kernel'].values
        e_total = s_total / p
        e_kernel = s_kernel / p
        
        # Plot Speedup
        ax1.loglog(p, s_total, marker='o', linestyle='-', color=color, 
                   label=f'{m_name} (Total)', linewidth=2)
        ax1.loglog(p, s_kernel, marker='s', linestyle='--', color=color, 
                   label=f'{m_name} (Kernel)', alpha=0.7)
        
        # Plot Efficiency
        ax2.plot(p, e_total, marker='o', linestyle='-', color=color, 
                 label=f'{m_name} (Total)', linewidth=2)
        ax2.plot(p, e_kernel, marker='s', linestyle='--', color=color, 
                 label=f'{m_name} (Kernel)', alpha=0.7)

    # Add Ideal Reference Lines
    ideal_p = np.array([1, max(core_counts)])
    ax1.loglog(ideal_p, ideal_p, color='grey', linestyle=':', linewidth=2, label='Ideal for plot')
    ax2.axhline(y=1.0, color='grey', linestyle=':', linewidth=2)

    # Aesthetics and Grids
    ax1.set_xticks(core_counts)
    ax1.set_xticklabels([str(c) for c in core_counts])
    ax1.minorticks_off()
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    ax2.set_xscale('log', base=2)
    ax2.set_xticks(core_counts)
    ax2.set_xticklabels([str(c) for c in core_counts])
    ax2.grid(True, which="both", ls="-", alpha=0.3)

    fig.legend(
        handles=ax1.get_legend_handles_labels()[0],
        loc='upper center',
        ncol=3,
        fontsize='medium'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])  

    # Save as PDF
    output_path = os.path.join(output_dir, 'strong_scaling_metrics.pdf')
    plt.savefig(output_path)
    print(f"Speedup and Efficiency graphs saved to: {output_path}")

if __name__ == "__main__":
    main()