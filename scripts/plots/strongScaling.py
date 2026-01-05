import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def main():
    # Verify required parameters
    if len(sys.argv) < 3:
        print("Usage: python script.py <json_path> <output_directory>")
        sys.exit(1)

    json_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(json_path):
        print(f"Error: The file {json_path} does not exist.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data from JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    # Extract and process results
    results = data.get('results', [])
    rows = []
    for r in results:
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
    strong_matrices = df[df['matrix'] != 'Generated']['matrix'].unique()

    # Setup plot
    plt.figure(figsize=(14, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, len(strong_matrices)))
    core_counts = sorted(df['p'].unique())

    for i, m_name in enumerate(strong_matrices):
        subset = df[df['matrix'] == m_name].sort_values('p')
        color = colors[i]

        # Baseline for ideal scaling (T1)
        t1_subset = subset[subset['p'] == 1]
        if t1_subset.empty:
            print(f"Warning: No p=1 data for matrix {m_name}. Skipping ideal line.")
            continue

        t1_total = t1_subset['t_total'].values[0]
        ideal_t = t1_total / subset['p']

        # Plot total performance (solid line with circles)
        plt.loglog(subset['p'], subset['t_total'], marker='o', linestyle='-',
                   color=color, label=f'{m_name} (Total)', linewidth=2.5)

        # Plot kernel performance without communication (dashed line with squares)
        plt.loglog(subset['p'], subset['t_kernel'], marker='s', linestyle='--',
                   color=color, label=f'{m_name} (Kernel)', markersize=4)

        # Plot ideal line (dotted line without marker)
        plt.loglog(subset['p'], ideal_t, color=color, linestyle=':',
                   linewidth=1.5, alpha=0.6)

    # Configure axes and labels
    plt.xticks(core_counts, labels=[str(c) for c in core_counts])
    plt.minorticks_off()

    plt.xlabel('Number of Processes (p)', fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontweight='bold')
    plt.title('Strong Scaling', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # External legend to preserve plot space
    plt.legend(loc='lower left', borderaxespad=0.)

    plt.tight_layout()

    # Save output
    output_path = os.path.join(output_dir, 'strong_scaling_analysis.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Graph successfully saved to: {output_path}")


if __name__ == "__main__":
    main()