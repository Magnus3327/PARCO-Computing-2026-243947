# ========================================
# Scheduling & Chunk Analysis with Dynamic Output Folder/File per Matrix
#
# - Reads sequential and parallel JSONs.
# - Plots speedup vs threads, color = scheduling, marker = chunk size.
# - Sequential point included (thread 1).
# - Output folder and file naming are dynamic based on matrix_name.
# ========================================

import json
import matplotlib.pyplot as plt
import os
import sys

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_sequential_duration(seq_json, matrix_name):
    for result in seq_json['results']:
        if result['matrix']['name'] == matrix_name:
            return result['statistics90']['duration_ms']
    return None

def get_parallel_runs_for_matrix(par_json, matrix_name):
    runs = []
    for result in par_json['results']:
        if result['matrix']['name'] == matrix_name:
            scenario = result['scenario']
            threads = scenario['threads']
            scheduling = scenario.get('scheduling_type', 'unknown')
            chunk_size = scenario.get('chunk_size', 'unknown')
            duration = result['statistics90']['duration_ms']
            runs.append({
                'threads': threads,
                'scheduling': scheduling,
                'chunk_size': chunk_size,
                'duration': duration
            })
    return runs

COLORBLIND_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#56B4E9', '#CC79A7', '#E69F00', '#000000']
MARKER_LIST = ['o', 's', '^', 'D', 'P', '*', 'X', 'v']

# Updated: allows output folder + dynamic file/title based on matrix_name
def plot_sched_chunk_matrix(matrix_name, seq_file, par_file, output_folder):
    seq_json = load_json(seq_file)
    par_json = load_json(par_file)
    seq_duration = get_sequential_duration(seq_json, matrix_name)
    if seq_duration is None:
        print("Sequential timing not found for selected matrix.")
        sys.exit(1)
    runs = get_parallel_runs_for_matrix(par_json, matrix_name)
    if not runs:
        print("No parallel experimental data for selected matrix.")
        sys.exit(1)

    plt.figure(figsize=(14,8))

    # Build mapping for scheduling/colors and chunk/markers
    sched_types = sorted(set(run['scheduling'] for run in runs))
    color_map = {sched: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)] for i, sched in enumerate(sched_types)}
    chunk_sizes = sorted(set(run['chunk_size'] for run in runs))
    marker_map = {chunk: MARKER_LIST[i % len(MARKER_LIST)] for i, chunk in enumerate(chunk_sizes)}

    # Plot each curve for combo (scheduling, chunk)
    for sched in sched_types:
        sched_runs = [run for run in runs if run['scheduling'] == sched]
        for chunk in chunk_sizes:
            combo_runs = [run for run in sched_runs if run['chunk_size'] == chunk]
            if not combo_runs:
                continue
            combo_runs = sorted(combo_runs, key=lambda r: r['threads'])
            threads = [r['threads'] for r in combo_runs]
            speedups = [seq_duration / r['duration'] if r['duration'] > 0 else 0 for r in combo_runs]

            # Always start each curve at thread=1, speedup=1
            threads = [1] + threads
            speedups = [1.0] + speedups

            color = color_map[sched]
            marker = marker_map[chunk]
            label = f"{sched}, chunk={chunk}"
            plt.scatter([1], [1.0], color='black', marker='o', s=120, zorder=10)
            plt.plot(threads, speedups, color=color, marker=marker, label=label, linewidth=2)

    # Make folder if needed
    os.makedirs(output_folder, exist_ok=True)
    # Format matrix name for output (safe filename)
    safe_matrix = matrix_name.replace(".mtx","").replace("/", "_").replace(" ", "_")
    output_file = f"{output_folder}/speedup_sched_chunk_{safe_matrix}.png"
    plot_title = f"Speedup for matrix '{matrix_name}' by Scheduling & Chunk Size"

    plt.xlabel("Number of Threads (1 = sequential)")
    plt.ylabel("Speedup (Sequential / Parallel Time)")
    plt.title(plot_title)
    plt.xticks([1,2,4,8,16,32], ["1","2","4","8","16","32"])
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)

# --- Accept output folder as third argument
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python plot_sched_chunk_matrix.py <matrix_name> <sequential.json> <parallel.json> <output_folder>")
        sys.exit(1)
    matrix_name = sys.argv[1]
    seq_file = sys.argv[2]
    par_file = sys.argv[3]
    output_folder = sys.argv[4]
    plot_sched_chunk_matrix(matrix_name, seq_file, par_file, output_folder)
