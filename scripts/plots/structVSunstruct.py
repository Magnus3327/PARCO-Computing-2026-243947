import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input_json> <output_directory>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_path, 'r') as f:
        data = json.load(f)

    # Filter real matrices (excluding 'Generated')
    results = [r for r in data['results'] if r['matrix']['name'] != "Generated"]
    processes = sorted(set(r['mpi']['processes'] for r in results if r['mpi']['processes'] > 1))
    
    # Categorization 
    categories = {
        'thermal2': 'Structured (Grid/FEM)',
        'Flan_1565': 'Structured (Grid/FEM)',
        'soc-LiveJournal1': 'Unstructured (Social Network)',
        'cit-Patents': 'Unstructured (Citation Network)'
    }

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(11, 6))

    for mat_id, cat_name in categories.items():
        p_list = []
        ratio_list = []
        
        for p in processes:
            # Check for matrix name match (handling potential path prefixes)
            res = next((r for r in results if r['mpi']['processes'] == p and mat_id in r['matrix']['name']), None)
            if res:
                nnz_avg = res['mpi']['nnz_per_rank']['avg']
                ghost_avg = res['mpi']['ghost_entries_per_rank']['avg']
                # Calculate the percentage of communication volume vs local work
                ratio = (ghost_avg / nnz_avg) * 100 
                p_list.append(p)
                ratio_list.append(ratio)
        
        if p_list:
            # Use different markers to distinguish types easily
            marker = 's' if 'Structured' in cat_name else 'o'
            ax.plot(p_list, ratio_list, marker=marker, label=f"{mat_id} [{cat_name}]", linewidth=2)

    ax.set_xscale('log', base=2)
    ax.set_xticks(processes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_xlabel('Number of Processes (MPI Ranks)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Communication Volume / NNZ (%)', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Matrix Topology on Communication Overhead', fontsize=14, pad=15)
    
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(loc='upper left', fontsize='small')
    
    # Technical annotation for the report
    ax.text(0.02, -0.15, "* Communication Overhead = (Avg Ghost Entries / Avg NNZ per rank) * 100", 
            transform=ax.transAxes, fontsize=9, style='italic')

    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'topology_comparison_ratio.png')
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print(f"Topology comparison plot saved to: {save_path}")

if __name__ == "__main__":
    main()