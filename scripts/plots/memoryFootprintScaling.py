import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    # Usual structure to take arguments from the command line
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input_json> <output_directory>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    results = data['results']
    processes = sorted(set(r['mpi']['processes'] for r in results))
    
    # Get unique matrix display names
    # We use a set to avoid duplicates, then sort: real matrices first, Generated last
    mat_names = sorted(set(
        "Generated" if "Generated" in r['matrix']['name'] 
        else os.path.basename(r['matrix']['name']).replace('.mtx', '') 
        for r in results
    ), key=lambda x: (x == "Generated", x))

    # Setup white plot style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))

    for mat in mat_names:
        p_coords = []
        m_coords = []
        
        for p in processes:
            # Check if this matrix name matches the result
            # Logic handles both "Generated" string and real matrix filenames
            res = next((r for r in results if r['mpi']['processes'] == p 
                        and (mat in r['matrix']['name'] or (mat == "Generated" and "Generated" in r['matrix']['name']))), None)
            if res:
                p_coords.append(p)
                # Convert bytes to MegaBytes
                m_coords.append(res['memory']['bytes_per_rank'] / (1024 * 1024))
        
        if p_coords:
            # APPLY CUSTOM STYLING
            if mat == "Generated":
                ax.plot(p_coords, m_coords, color='grey', linestyle='--', 
                        marker='s', label=mat, linewidth=1.5, alpha=0.7)
            else:
                ax.plot(p_coords, m_coords, marker='o', label=mat, 
                        linewidth=2, markersize=7)

    # Log-Log scale configuration 
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(processes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    # Labels and titles
    ax.set_xlabel('Number of Processes (MPI Ranks)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Memory per Rank (MB)', fontsize=11, fontweight='bold')
    ax.set_title('Memory Footprint per Process', fontsize=14, pad=15)
    
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(title="Matrices", fontsize='small')

    plt.tight_layout()
    
    # Save the file and close the figure (don't open UI)
    save_path = os.path.join(output_dir, 'memory_footprint_scaling.png')
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print(f"Memory Footprint plot saved to: {save_path}")

if __name__ == "__main__":
    main()