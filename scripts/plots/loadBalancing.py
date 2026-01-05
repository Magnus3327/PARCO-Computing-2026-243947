import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <json_path> <output_directory>")
        sys.exit(1)

    json_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(json_path):
        print(f"Error: file {json_path} does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Load JSON
    # -----------------------------
    with open(json_path, 'r') as f:
        data = json.load(f)

    rows = []
    for r in data.get('results', []):
        if r.get('errors'):
            continue
        
        raw_name = r['matrix']['name']
        is_generated = "Generated" in raw_name
        display_name = "Generated (Weak)" if is_generated else raw_name.split('/')[-1]

        nnz = r['mpi']['nnz_per_rank']
        ghost = r['mpi']['ghost_entries_per_rank']

        rows.append({
            'matrix': display_name,
            'is_generated': is_generated,
            'p': r['mpi']['processes'],
            'nnz_min_avg': nnz['min'] / nnz['avg'] * 100,
            'nnz_max_avg': nnz['max'] / nnz['avg'] * 100,
            'ghost_ratio': ghost['avg'] / nnz['avg'] * 100
        })

    df = pd.DataFrame(rows)

    # Sorting to keep real matrices together and Generated last
    matrices = sorted(df['matrix'].unique(), key=lambda x: ("Generated" in x, x))
    processes = sorted(df['p'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(matrices)))

    # FIGURE: 2 SUBPLOTS SIDE-BY-SIDE
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # LEFT: LOAD BALANCE (Min/Avg and Max/Avg)
    ax1.set_title("Load Balance: Min & Max vs Average NNZ", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Number of Processes (p)", fontsize=12)
    ax1.set_ylabel("Ratio to Average (%)", fontsize=12)

    for i, m in enumerate(matrices):
        sub = df[df['matrix'] == m].sort_values('p')
        is_gen = sub['is_generated'].iloc[0]
        
        linestyle = '--' if is_gen else '-'
        color = colors[i]
        
        # Plot Min/Avg
        ax1.plot(sub['p'], sub['nnz_min_avg'], marker='o', linestyle=linestyle,
                 color=color, linewidth=1.5, alpha=0.7, label=f"{m} min/avg")
        # Plot Max/Avg
        ax1.plot(sub['p'], sub['nnz_max_avg'], marker='s', linestyle=linestyle,
                 color=color, linewidth=2, alpha=1.0, label=f"{m} max/avg")

    ax1.axhline(100, color='black', linewidth=1, linestyle=':', alpha=0.8)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(processes)
    ax1.set_xticklabels(processes)
    ax1.grid(True, which="both", alpha=0.3)

    # RIGHT: COMMUNICATION OVERHEAD
    ax2.set_title("Communication Overhead (Ghost / NNZ)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Number of Processes (p)", fontsize=12)
    ax2.set_ylabel("Ghost / NNZ (%)", fontsize=12)

    for i, m in enumerate(matrices):
        sub = df[df['matrix'] == m].sort_values('p')
        is_gen = sub['is_generated'].iloc[0]
        
        linestyle = '--' if is_gen else '-'
        marker = 'D' if is_gen else 'o'

        ax2.plot(sub['p'], sub['ghost_ratio'], marker=marker, linestyle=linestyle,
                 color=colors[i], linewidth=2, label=m)

    ax2.set_xscale('log', base=2)
    ax2.set_xticks(processes)
    ax2.set_xticklabels(processes)
    ax2.grid(True, which="both", alpha=0.3)

    # LEGEND HANDLING

    fig.legend(handles=ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0],
               loc='upper center',
               ncol=3,
               fontsize='medium')

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    out = os.path.join(output_dir, "detailed_load_balance_and_comm.pdf")
    plt.savefig(out, dpi=300)
    plt.close(fig)

    print(f"Detailed plot with Min/Max/Avg and Generated matrices saved to: {out}")

if __name__ == "__main__":
    main()