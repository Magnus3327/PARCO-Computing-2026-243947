import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_performance.py <json_path> <output_directory>")
        sys.exit(1)

    json_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    results = data.get('results', [])
    rows = []
    for r in results:
        if r.get('errors'): continue
        rows.append({
            'matrix': r['matrix']['name'].split('/')[-1],
            'p': r['mpi']['processes'],
            'gflops': r['statistics']['gflops']
        })

    df = pd.DataFrame(rows)
    plt.rcParams.update({'font.size': 12, 'grid.alpha': 0.3})
    plt.figure(figsize=(12, 9))

    all_p = sorted(df['p'].unique())
    matrices = sorted(df['matrix'].unique())

    # Reference for the ideal line entry in the legend
    ideal_line_proxy = None

    for m in matrices:
        subset = df[df['matrix'] == m].sort_values('p')
        
        if m == "Generated":
            color = 'grey'
            label = "Generated (Weak Scaling)"
            marker = 's'
        else:
            color = None 
            label = f"{m} (Strong Scaling)"
            marker = 'o'
        
        # Plot Measured Data
        line, = plt.plot(subset['p'], subset['gflops'], marker=marker, markersize=8, linewidth=2.5, label=label, color=color)
        
        # Ideal Scaling Logic: GFLOPS(p) = p * GFLOPS(1)
        p_min = subset['p'].min()
        g_min = subset[subset['p'] == p_min]['gflops'].values[0]
        ideal_p = np.array(all_p)
        ideal_g = (ideal_p / p_min) * g_min
        
        # Save one instance for the legend reference
        ideal_line_proxy, = plt.plot(ideal_p, ideal_g, linestyle='--', color=line.get_color(), alpha=0.3)

    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xticks(all_p, [str(p) for p in all_p])
    plt.grid(True, which="both", ls="-")

    plt.title('GFLOP/s Strong & Weak Scaling Analysis', fontweight='bold')
    plt.ylabel('Performance (GFLOP/s)', fontweight='bold')

    plt.xlabel('Number of MPI Processes (p)', fontsize=14)

    # LEGEND CUSTOMIZATION
    from matplotlib.lines import Line2D
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Add a custom entry to explain the dashed lines
    custom_lines = [Line2D([0], [0], color='black', linestyle='--', alpha=0.5)]
    custom_labels = ["Ideal Scaling ($p \\times GFLOPS_{p=1}$)"]
    
    plt.legend(handles + custom_lines, labels + custom_labels, loc='upper left', fontsize=10, frameon=True, title="Scaling Results")

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'spmv_unified_performance.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Graph saved to: {output_path}")

if __name__ == "__main__":
    main()