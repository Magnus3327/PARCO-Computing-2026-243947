import json
import matplotlib.pyplot as plt
import os
import sys

# Speedup plot (linear axes): speedup vs number of threads.
# Shows best speedup per thread count for each matrix in color with marker.
# Annotates the maximum speedup point with scheduling type and chunk size to the right.

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def extract_sequential_durations(data):
    seq_durations = {}
    for result in data['results']:
        name = result['matrix']['name']
        duration = result['statistics90']['duration_ms']
        seq_durations[name] = duration
    return seq_durations

def extract_best_speedup_and_config(seq_json, par_json):
    seq_durations = extract_sequential_durations(seq_json)
    best_by_matrix = {}
    for result in par_json['results']:
        name = result['matrix']['name']
        scenario = result['scenario']
        threads = scenario['threads']
        scheduling = scenario['scheduling_type'] if 'scheduling_type' in scenario else 'unknown'
        chunk_size = scenario['chunk_size'] if 'chunk_size' in scenario else 'unknown'
        duration = result['statistics90']['duration_ms']
        if name not in seq_durations:
            continue
        speedup = seq_durations[name] / duration if duration > 0 else 0
        if name not in best_by_matrix:
            best_by_matrix[name] = {}
        if threads not in best_by_matrix[name] or (speedup > best_by_matrix[name][threads][0]):
            best_by_matrix[name][threads] = (speedup, scheduling, chunk_size)
    return best_by_matrix

def plot_speedup_annotate_right(sequential_file, parallel_file, output_folder):
    seq_json = load_json(sequential_file)
    par_json = load_json(parallel_file)

    best_by_matrix = extract_best_speedup_and_config(seq_json, par_json)

    plt.figure(figsize=(12, 8))
    color_palette = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c', '#984ea3']

    for idx, (matrix, vals_by_thread) in enumerate(best_by_matrix.items()):
        threads = sorted(vals_by_thread.keys())
        if 1 not in threads:
            threads = [1] + threads
        speedups = []
        scheds = []
        chunks = []
        for t in threads:
            if t == 1:
                speedups.append(1.0)
                scheds.append('-')
                chunks.append('-')
            else:
                speedups.append(vals_by_thread[t][0])
                scheds.append(vals_by_thread[t][1])
                chunks.append(vals_by_thread[t][2])
        color = color_palette[idx % len(color_palette)]
        plt.plot(threads, speedups, marker='o', color=color, label=matrix, linewidth=2)
        # Mark max speedup with downward-pointing triangle
        max_idx = speedups.index(max(speedups))
        max_thread = threads[max_idx]
        max_speedup = speedups[max_idx]
        max_sched = scheds[max_idx]
        max_chunk = chunks[max_idx]
        plt.scatter([max_thread], [max_speedup], color=color, marker='v', s=160, edgecolors='k', zorder=10)
        
        # Annotate to the right of the point, write max values, speedup have 2 decimal places
        annotation = f"speedUp:{max_speedup:.2f} ,scheduling: {max_sched}, chunkSize: {max_chunk}"
        plt.annotate(
            annotation,
            xy=(max_thread, max_speedup),
            xytext=(max_thread, max_speedup + 0.4 ),
            fontsize=9,
            color=color,
            ha='left',
            va='center'
        )

    plt.xlabel('Number of Threads (1 = Sequential)')
    plt.ylabel('Speedup (Sequential Time / Parallel Time)')
    plt.title('Speedup vs Threads per Matrix\nShowing Best Speedup Configuration Annotations')
    plt.grid(True, linestyle=':')
    plt.xticks([1,2,4,8,16,32], ["1","2","4","8","16","32"])
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, 'spmv_speedup.png'))
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python plot_speedup.py <sequential.json> <parallel.json> <output_folder>")
        sys.exit(1)
    sequential_file = sys.argv[1]
    parallel_file = sys.argv[2]
    output_folder = sys.argv[3]
    plot_speedup_annotate_right(sequential_file, parallel_file, output_folder)