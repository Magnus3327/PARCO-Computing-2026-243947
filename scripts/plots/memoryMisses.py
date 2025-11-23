import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_perf_file(filepath):
    """Extracts miss rates (%) plus elapsed time and config string from perf file."""
    with open(filepath) as f:
        text = f.read()
    def extract_int(pattern):
        m = re.search(pattern, text)
        return int(m.group(1).replace(",", "")) if m else None
    def extract_float(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None
    l1_loads = extract_int(r'([0-9,]+)\s+L1-dcache-loads')
    l1_misses = extract_int(r'([0-9,]+)\s+L1-dcache-load-misses')
    llc_loads = extract_int(r'([0-9,]+)\s+LLC-loads')
    llc_misses = extract_int(r'([0-9,]+)\s+LLC-load-misses')
    cache_refs = extract_int(r'([0-9,]+)\s+cache-references')
    cache_misses = extract_int(r'([0-9,]+)\s+cache-misses')
    elapsed = extract_float(r'([0-9.]+) seconds time elapsed')
    base = os.path.basename(filepath)
    if "seq_perf_" in base:
        config = "sequential"
    else:
        conf_match = re.search(r'(best|median|worst)_T(\d+)_S([\w]+)_C(\d+)', base)
        if conf_match:
            _, threads, sched, chunk = conf_match.groups()
            config = f"{sched}, chunk={chunk}, th={threads}"
        else:
            config = "unknown"
    def rate(nmiss, nref):
        if nmiss is not None and nref and nref > 0:
            return 100.0 * nmiss / nref
        return np.nan
    l1_rate = rate(l1_misses, l1_loads)
    llc_rate = rate(llc_misses, llc_loads)
    cache_rate = rate(cache_misses, cache_refs)
    return l1_rate, llc_rate, cache_rate, config, elapsed

def gather_matrix_files(root):
    """Collects only matrices where BOTH seq and best file are present."""
    matrices = {}
    par_dir = os.path.join(root, "parallel")
    seq_dir = os.path.join(root, "sequential")
    # Find all "best" files among parallel ones
    for fname in os.listdir(par_dir):
        mat_match = re.match(r'perf_(.+?)\.mtx_best_.*\.txt', fname)
        if mat_match:
            matrix = mat_match.group(1)
            if matrix not in matrices:
                matrices[matrix] = {"best_file": None, "seq_file": None}
            matrices[matrix]["best_file"] = os.path.join(par_dir, fname)
    for fname in os.listdir(seq_dir):
        mat_match = re.match(r'seq_perf_(.+?)\.mtx\.txt', fname)
        if mat_match:
            matrix = mat_match.group(1)
            if matrix not in matrices:
                matrices[matrix] = {"best_file": None, "seq_file": None}
            matrices[matrix]["seq_file"] = os.path.join(seq_dir, fname)
    # Remove matrices without both files
    matrices = {m: f for m, f in matrices.items() if f["best_file"] and f["seq_file"]}
    return matrices

def plot_best_vs_seq_misses(matrices, outfolder):
    """Single plot with only best parallel and sequential configs per matrix."""
    matrix_colors = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#A020F0','#F0E442','#56B4E9']
    miss_types = ['L1 Miss Rate (%)','LLC Miss Rate (%)','Cache Miss Rate (%)']
    miss_markers = ['o','s','D']
    font_size = 10

    plot_data = []
    labels = []
    for i, (matrix, files) in enumerate(sorted(matrices.items())):
        color = matrix_colors[i % len(matrix_colors)]
        # Parse sequential
        l1, llc, cm, conf, _ = parse_perf_file(files["seq_file"])
        plot_data.append({'matrix':matrix, 'config':conf, 'rates':[l1, llc, cm], 'color':color})
        labels.append(f"{matrix}\n{conf}")
        # Parse best
        l1, llc, cm, conf, _ = parse_perf_file(files["best_file"])
        plot_data.append({'matrix':matrix, 'config':conf, 'rates':[l1, llc, cm], 'color':color})
        labels.append(f"{matrix}\n{conf}")

    xticks = np.arange(len(plot_data))
    fig, ax = plt.subplots(figsize=(max(10,len(plot_data)//2),6))
    for j, (miss_type, marker) in enumerate(zip(miss_types, miss_markers)):
        ax.scatter(xticks, [d['rates'][j] for d in plot_data],
                   c=[d['color'] for d in plot_data], marker=marker, s=90, label=miss_type, edgecolors='black')
    ax.legend(title="Miss Type", fontsize=font_size, title_fontsize=font_size+1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=font_size-1)
    ax.set_ylabel('Miss Rate (%)', fontsize=font_size+2)
    ax.set_ylim(0, 100)
    ax.set_title('Cache & Memory Miss Percentage (Seq vs Best Config Only)', fontsize=font_size+5)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    plt.tight_layout()
    os.makedirs(outfolder, exist_ok=True)
    fig.savefig(os.path.join(outfolder, "seq_best_percentmiss_scatter.png"))

def main(perf_dir, outfolder):
    matrices = gather_matrix_files(perf_dir)
    if not matrices:
        print(f"No matrices with both seq and best files.")
    plot_best_vs_seq_misses(matrices, outfolder)
    print(f"Plot saved in: {outfolder}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python plot_seq_best_percentmiss.py <perf_folder> <output_folder>")
        sys.exit(1)
    perf_dir = sys.argv[1]
    outfolder = sys.argv[2]
    main(perf_dir, outfolder)
