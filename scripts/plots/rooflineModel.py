import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def extract_best_roofline_points(par_json):
    # Pick highest GFLOPS run per matrix
    best_points = {}
    for result in par_json['results']:
        matrix = result['matrix']['name']
        stats = result.get('statistics90', {})
        gflops = stats.get('GFLOPS')
        arith_intensity = stats.get('Arithmetic_intensity')
        scenario = result.get('scenario', {})
        sched = scenario.get('scheduling_type','unknown')
        chunk = scenario.get('chunk_size','unknown')
        threads = scenario.get('threads','?')
        if gflops is None or arith_intensity is None:
            continue
        if matrix not in best_points or gflops > best_points[matrix]['gflops']:
            best_points[matrix] = {
                'gflops': gflops,
                'intensity': arith_intensity,
                'sched': sched,
                'chunk': chunk,
                'threads': threads
            }
    return best_points

def plot_roofline_clean(parallel_file, output_folder, mem_bw, peak_flops):
    par_json = load_json(parallel_file)
    best_points = extract_best_roofline_points(par_json)

    plt.figure(figsize=(10,7))
    OI_range = np.logspace(-2, 2, 300)
    theory = np.minimum(OI_range * mem_bw, peak_flops)
    plt.plot(OI_range, theory, color='blue', linewidth=2, zorder=2)

    ridge = peak_flops / mem_bw
    plt.axvline(ridge, color='black', linewidth=1.5, linestyle='-', alpha=0.6, zorder=1)
    plt.annotate(f'Ridge: OI={ridge:.2f}',
                 xy=(ridge, ridge*mem_bw*0.8),
                 xytext=(ridge*3, ridge*mem_bw*0.7),
                 arrowprops=None,
                 fontsize=10, color='dimgray', ha='left', fontweight='bold')

    plt.axhline(peak_flops, color='gray', linestyle='--', alpha=0.5, zorder=1)
    plt.annotate(f'Peak {peak_flops:.0f} GFLOPS',
                 xy=(OI_range[-1]/2, peak_flops*0.98),
                 xytext=(OI_range[-1]/18, peak_flops*0.65),
                 arrowprops=None,
                 fontsize=10, color='red', ha='left', fontweight='bold')

    plt.plot([OI_range[0], ridge], [OI_range[0]*mem_bw, ridge*mem_bw], color='gray', lw=1.5, linestyle='-')
    plt.annotate(f'Bandwidth {mem_bw} GB/s',
        xy=(OI_range[0]*10, OI_range[0]*mem_bw*10),
        xytext=(OI_range[0]*2, mem_bw/15),
        fontsize=9, color='gray', ha='left')

    # Plot best configs per matrix, label config near dot, smaller font
    matrix_colors = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#A020F0','#F0E442','#56B4E9']
    for i, (mat, p) in enumerate(best_points.items()):
        color = matrix_colors[i % len(matrix_colors)]
        config = f"{p['sched']}, chunk={p['chunk']}, thread={p['threads']}"
        plt.scatter(p['intensity'], p['gflops'], s=120, marker='o', color=color, edgecolors='black', label=mat, zorder=5)
        plt.text(p['intensity'] * 1.12, p['gflops'] * 1.02, config,
                 fontsize=9, color=color, ha='left', va='bottom', fontweight='normal')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=10)
    plt.ylabel('Measured Performance (GFLOPS)', fontsize=10)
    plt.title('Roofline Model using Best SpMV Run ', fontsize=11)
    plt.grid(True, which='both', linestyle=':')
    plt.legend(loc='lower right', fontsize=9)
    plt.tick_params(axis='both', labelsize=9)
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, "roofline_spmv_bestpoints_config.png"))
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python plot_roofline_spmv_best.py <parallel.json> <output_folder> <MEM_BW_GBps> <PEAK_FLOPS_GFLOPS>")
        sys.exit(1)
    parallel_file = sys.argv[1]
    output_folder = sys.argv[2]
    mem_bw = float(sys.argv[3])
    peak_flops = float(sys.argv[4])
    plot_roofline_clean(parallel_file, output_folder, mem_bw, peak_flops)
