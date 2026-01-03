import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    # Verify that the necessary parameters were passed
    if len(sys.argv) < 3:
        print("Usage: python script.py <json_path> <output_directory>")
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
        # Skip entries containing errors
        if r.get('errors'):
            continue
            
        rows.append({
            'matrix': r['matrix']['name'].split('/')[-1],
            'p': r['mpi']['processes'],
            't_kernel': r['statistics']['kernel_p90_ms'],
            't_comm': r['timings_ms']['communication'],
            't_total': r['statistics']['kernel_p90_ms'] + r['timings_ms']['communication']
        })

    if not rows:
        print("No valid results found in the JSON file.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Filter: exclude "Generated" matrices for Strong Scaling analysis
    strong_matrices = df[df['matrix'] != 'Generated']['matrix'].unique()

    plt.figure(figsize=(12, 8))

    # Define colors for matrix consistency
    colors = plt.cm.tab10(np.linspace(0, 1, len(strong_matrices)))
    core_counts = sorted(df['p'].unique())

    ideal_label_added = False

    for i, m_name in enumerate(strong_matrices):
        subset = df[df['matrix'] == m_name].sort_values('p')
        color = colors[i]
        
        # Calculate specific ideal scaling (T1 / p)
        # We ensure p=1 exists for the baseline
        t1_subset = subset[subset['p'] == 1]
        if t1_subset.empty:
            continue
            
        t1_total = t1_subset['t_total'].values[0]
        ideal_t = t1_total / subset['p']
        
        # 1. Total Time: Solid line with circles
        plt.loglog(subset['p'], subset['t_total'], marker='o', linestyle='-', 
                   color=color, label=f'{m_name} (Total)', linewidth=2)
        
        # 2. Kernel Time: Dashed line with squares
        plt.loglog(subset['p'], subset['t_kernel'], marker='s', linestyle='--', 
                   color=color, label=f'{m_name} (Kernel)', alpha=0.8)
        
        # 3. Ideals in grey: One per matrix, but only one entry in the legend
        if not ideal_label_added:
            plt.loglog(subset['p'], ideal_t, color='grey', linestyle=':', 
                       linewidth=1.5, alpha=0.6, label='Ideal Scaling (Slope -1)')
            ideal_label_added = True
        else:
            plt.loglog(subset['p'], ideal_t, color='grey', linestyle=':', 
                       linewidth=1.5, alpha=0.6)

    # X-axis configuration
    plt.xticks(core_counts, labels=[str(c) for c in core_counts])
    plt.minorticks_off()

    plt.xlabel('Number of Processes (p)')
    plt.ylabel('Execution Time (ms)')
    plt.title('Strong Scaling: Real Performance vs. Ideal References')
    plt.grid(True, which="major", ls="-", alpha=0.3)

    # Legend positioning
    plt.legend(loc='lower left', fontsize='small')
    plt.tight_layout()

    # Save as PDF in the specified directory
    output_path = os.path.join(output_dir, 'strong_scaling_analysis.pdf')
    plt.savefig(output_path)
    print(f"Graph successfully saved to: {output_path}")

if __name__ == "__main__":
    main()